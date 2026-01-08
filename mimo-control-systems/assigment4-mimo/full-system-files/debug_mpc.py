#!/usr/bin/env python3
import argparse
import math
import signal
import subprocess
import threading
import time

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry, Path
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

from utils.mpc_utils import euler_from_quaternion


class MPCDebugger:
    def __init__(self, launch_pkg: str, world_launch: str, startup_sleep: float, auto_launch: bool):
        self.launch_pkg = launch_pkg
        self.world_launch = world_launch
        self.startup_sleep = startup_sleep
        self.auto_launch = auto_launch
        self.launch_proc = None

        if self.auto_launch:
            self.launch_proc = self._spawn_launch()

        rospy.init_node("mpc_debugger", anonymous=True)

        # Allow ROS params to override the CLI defaults
        self.launch_pkg = rospy.get_param("~launch_pkg", self.launch_pkg)
        self.world_launch = rospy.get_param("~launch_file", self.world_launch)
        self.startup_sleep = rospy.get_param("~startup_sleep", self.startup_sleep)
        self.auto_launch = rospy.get_param("~auto_launch", self.auto_launch)

        if self.auto_launch and self.launch_proc is None:
            self.launch_proc = self._spawn_launch()

        rospy.on_shutdown(self._shutdown_processes)

        self.state_lock = threading.Lock()
        self.heading_err = 0.0
        self.distance_err = 0.0
        self.cmd = TwistStamped()
        self.odom = Odometry()
        self.path = Path()
        self.heading_seq = 0
        self.distance_seq = 0
        self.last_reset_time = rospy.Time(0)
        self.episode = 0

        self.distance_limit = rospy.get_param("~distance_limit", 0.21)
        self.stuck_speed_limit = rospy.get_param("~stuck_speed_limit", 0.15)
        self.cmd_lin_min = rospy.get_param("~cmd_min_linear_for_reset", 0.25)
        self.max_pitch_roll = rospy.get_param("~max_pitch_roll", 0.01)
        self.reset_cooldown = rospy.Duration(rospy.get_param("~reset_cooldown", 5.0))
        self.print_period = rospy.get_param("~print_period", 0.5)
        self.reset_data_timeout = rospy.get_param("~reset_data_timeout", 1.0)

        rospy.Subscriber("/terrasentia/heading_error", Float32MultiArray, self.heading_cb, queue_size=10)
        rospy.Subscriber("/terrasentia/distance_error", Float32MultiArray, self.distance_cb, queue_size=10)
        rospy.Subscriber("/terrasentia/cmd_vel", TwistStamped, self.cmd_cb, queue_size=10)
        rospy.Subscriber("/terrasentia/ground_truth", Odometry, self.odom_cb, queue_size=10)
        rospy.Subscriber("/terrasentia/path", Path, self.path_cb, queue_size=10)

        self.reset_srv = rospy.ServiceProxy("/gazebo/reset_world", Empty)
        self.pause_srv = rospy.ServiceProxy("/gazebo/pause_physics", Empty)
        self.unpause_srv = rospy.ServiceProxy("/gazebo/unpause_physics", Empty)

        self._wait_for_services()

        self.print_timer = rospy.Timer(rospy.Duration(self.print_period), self.print_status)

    def _spawn_launch(self):
        proc = subprocess.Popen(
            ["roslaunch", self.launch_pkg, self.world_launch],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
        )
        time.sleep(self.startup_sleep)
        return proc

    def _wait_for_services(self):
        services = ["/gazebo/reset_world", "/gazebo/pause_physics", "/gazebo/unpause_physics"]
        for service in services:
            try:
                rospy.wait_for_service(service, timeout=self.startup_sleep)
            except rospy.ROSException:
                rospy.logwarn("[mpc_debug] service %s unavailable", service)

    def _shutdown_processes(self):
        if self.launch_proc and self.launch_proc.poll() is None:
            self.launch_proc.send_signal(signal.SIGINT)
            try:
                self.launch_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.launch_proc.kill()

    def heading_cb(self, msg):
        if msg.data:
            with self.state_lock:
                self.heading_seq += 1
                self.heading_err = float(msg.data[0])

    def distance_cb(self, msg):
        if msg.data:
            with self.state_lock:
                self.distance_seq += 1
                self.distance_err = float(msg.data[0])

    def cmd_cb(self, msg):
        with self.state_lock:
            self.cmd = msg

    def odom_cb(self, msg):
        with self.state_lock:
            self.odom = msg

    def path_cb(self, msg):
        with self.state_lock:
            self.path = msg

    def print_status(self, _):
        with self.state_lock:
            lin_vel = self.odom.twist.twist.linear.x
            ang_vel = self.odom.twist.twist.angular.z
            cmd_lin = self.cmd.twist.linear.x
            cmd_ang = self.cmd.twist.angular.z
            q = self.odom.pose.pose.orientation
            roll, pitch, yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
            heading = self.heading_err
            distance = self.distance_err
            pose_count = len(self.path.poses)
            frame_id = self.path.header.frame_id if self.path.header.frame_id else "?"
            first_wp = self.path.poses[0].pose.position if pose_count >= 1 else None
            if pose_count >= 2:
                p0 = self.path.poses[0].pose.position
                p1 = self.path.poses[1].pose.position
                path_heading = math.atan2(p1.y - p0.y, p1.x - p0.x)
            else:
                path_heading = float("nan")
            if not math.isnan(path_heading):
                heading_diff = (heading - path_heading + math.pi) % (2 * math.pi) - math.pi
            else:
                heading_diff = float("nan")

        rospy.loginfo(
            "[mpc_debug] ep=%d vel=(%.3f, %.3f) cmd=(%.3f, %.3f) eul=(%.3f, %.3f, %.3f) "
            "err=(%.3f, %.3f) path=%d frame=%s path_heading=%.3f heading_diff=%.3f first_wp=(%.3f, %.3f)",
            self.episode,
            lin_vel,
            ang_vel,
            cmd_lin,
            cmd_ang,
            roll,
            pitch,
            yaw,
            heading,
            distance,
            pose_count,
            frame_id,
            path_heading,
            heading_diff,
            first_wp.x if first_wp else float("nan"),
            first_wp.y if first_wp else float("nan"),
        )

        if self.should_reset(lin_vel, cmd_lin, distance, pitch, roll):
            self.reset_sim("auto")

    def should_reset(self, lin_vel, cmd_lin, distance, pitch, roll):
        now = rospy.Time.now()
        if now - self.last_reset_time < self.reset_cooldown:
            return False

        if abs(distance) > self.distance_limit:
            rospy.logwarn("[mpc_debug] distance breach (%.3f)", distance)
            return True

        if abs(lin_vel) < self.stuck_speed_limit and abs(cmd_lin) > self.cmd_lin_min:
            rospy.logwarn("[mpc_debug] robot stuck (vel=%.3f cmd=%.3f)", lin_vel, cmd_lin)
            return True

        if abs(pitch) > self.max_pitch_roll or abs(roll) > self.max_pitch_roll:
            rospy.logwarn("[mpc_debug] excessive tilt (pitch=%.4f roll=%.4f)", pitch, roll)
            return True

        return False

    def reset_sim(self, reason):
        rospy.logwarn("[mpc_debug] reset triggered (%s)", reason)
        with self.state_lock:
            prev_heading_seq = self.heading_seq
            prev_distance_seq = self.distance_seq
        try:
            self.unpause_srv()
        except rospy.ServiceException:
            pass
        try:
            self.reset_srv()
        except rospy.ServiceException as err:
            rospy.logerr("reset_world failed: %s", err)
            return
        try:
            self.pause_srv()
        except rospy.ServiceException:
            pass
        try:
            self.unpause_srv()
        except rospy.ServiceException:
            pass
        deadline = time.time() + self.reset_data_timeout
        while time.time() < deadline:
            with self.state_lock:
                heading_ready = self.heading_seq != prev_heading_seq
                distance_ready = self.distance_seq != prev_distance_seq
            if heading_ready and distance_ready:
                break
            rospy.sleep(0.05)
        else:
            rospy.logwarn("[mpc_debug] reset wait timed out waiting for fresh errors")
        
        with self.state_lock:
            # rossleep to ensure reset time is updated after reset
            rospy.sleep(0.5)
            self.last_reset_time = rospy.Time.now()
            self.episode += 1

    def spin(self):
        rospy.loginfo("[mpc_debug] ready")
        rospy.spin()


def parse_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--launch-pkg", default="terra_gazebo")
    parser.add_argument("--launch-file", default="terra_gazebo_mpc.launch")
    parser.add_argument("--startup-sleep", type=float, default=6.0)
    parser.add_argument("--no-auto-launch", action="store_true")
    args, _ = parser.parse_known_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    auto_launch = not args.no_auto_launch
    try:
        debugger = MPCDebugger(
            launch_pkg=args.launch_pkg,
            world_launch=args.launch_file,
            startup_sleep=args.startup_sleep,
            auto_launch=auto_launch,
        )
        debugger.spin()
    except rospy.ROSInterruptException:
        pass
