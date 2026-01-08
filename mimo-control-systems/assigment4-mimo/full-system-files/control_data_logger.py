#!/usr/bin/env python3
"""
Data Logger for Robust Control Analysis

Logs control system data to CSV for offline μ-synthesis and performance analysis.
Captures: time, errors, commands, velocities, and derived metrics.

Usage:
    rosrun terra_mpc control_data_logger.py

Output:
    Creates timestamped CSV files in ~/.ros/control_logs/
"""

import os
import csv
import math
import threading
from datetime import datetime

import rospy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Float32MultiArray


def euler_from_quaternion(x, y, z, w):
    """Convert quaternion to euler angles (roll, pitch, yaw)."""
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)

    t2 = 2.0 * (w * y - z * x)
    t2 = max(-1.0, min(1.0, t2))
    pitch = math.asin(t2)

    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)

    return roll, pitch, yaw


class ControlDataLogger:
    def __init__(self):
        rospy.init_node('control_data_logger', anonymous=True)
        
        # Parameters
        self.log_rate = rospy.get_param('~log_rate', 10.0)  # Hz
        self.log_dir = rospy.get_param('~log_dir', os.path.expanduser('~/.ros/control_logs'))
        
        # Create log directory
        os.makedirs(self.log_dir, exist_ok=True)
        
        # Generate unique filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.csv_path = os.path.join(self.log_dir, f'control_data_{timestamp}.csv')
        
        # State variables (protected by lock)
        self.lock = threading.Lock()
        self.heading_error = 0.0
        self.distance_error = 0.0
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0
        self.vel_lin = 0.0
        self.vel_ang = 0.0
        self.pose_x = 0.0
        self.pose_y = 0.0
        self.yaw = 0.0
        self.roll = 0.0
        self.pitch = 0.0
        
        # Timing
        self.start_time = None
        self.sample_count = 0
        
        # Subscribers
        rospy.Subscriber('/terrasentia/heading_error', Float32MultiArray, 
                         self.heading_cb, queue_size=10)
        rospy.Subscriber('/terrasentia/distance_error', Float32MultiArray, 
                         self.distance_cb, queue_size=10)
        rospy.Subscriber('/terrasentia/cmd_vel', TwistStamped, 
                         self.cmd_cb, queue_size=10)
        rospy.Subscriber('/terrasentia/ground_truth', Odometry, 
                         self.odom_cb, queue_size=10)
        
        # Open CSV file and write header
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            'time_s',           # Elapsed time in seconds
            'heading_err_rad',  # Heading error (theta)
            'distance_err_m',   # Lateral error (y)
            'cmd_lin_mps',      # Commanded linear velocity
            'cmd_ang_radps',    # Commanded angular velocity
            'vel_lin_mps',      # Actual linear velocity
            'vel_ang_radps',    # Actual angular velocity
            'pose_x_m',         # Robot X position (world frame)
            'pose_y_m',         # Robot Y position (world frame)
            'yaw_rad',          # Robot yaw (world frame)
            'roll_rad',         # Robot roll
            'pitch_rad',        # Robot pitch
            'tracking_err_m',   # Combined tracking error norm
        ])
        
        rospy.loginfo(f'[ControlDataLogger] Logging to: {self.csv_path}')
        
        # Start logging timer
        self.log_timer = rospy.Timer(rospy.Duration(1.0 / self.log_rate), self.log_data)
        
        rospy.on_shutdown(self.shutdown)
    
    def heading_cb(self, msg):
        if msg.data:
            with self.lock:
                self.heading_error = float(msg.data[0])
    
    def distance_cb(self, msg):
        if msg.data:
            with self.lock:
                self.distance_error = float(msg.data[0])
    
    def cmd_cb(self, msg):
        with self.lock:
            self.cmd_lin = msg.twist.linear.x
            self.cmd_ang = msg.twist.angular.z
    
    def odom_cb(self, msg):
        with self.lock:
            self.pose_x = msg.pose.pose.position.x
            self.pose_y = msg.pose.pose.position.y
            self.vel_lin = msg.twist.twist.linear.x
            self.vel_ang = msg.twist.twist.angular.z
            q = msg.pose.pose.orientation
            self.roll, self.pitch, self.yaw = euler_from_quaternion(q.x, q.y, q.z, q.w)
    
    def log_data(self, event):
        if self.start_time is None:
            self.start_time = rospy.Time.now()
        
        elapsed = (rospy.Time.now() - self.start_time).to_sec()
        
        with self.lock:
            tracking_err = math.sqrt(self.heading_error**2 + self.distance_error**2)
            
            row = [
                f'{elapsed:.3f}',
                f'{self.heading_error:.6f}',
                f'{self.distance_error:.6f}',
                f'{self.cmd_lin:.6f}',
                f'{self.cmd_ang:.6f}',
                f'{self.vel_lin:.6f}',
                f'{self.vel_ang:.6f}',
                f'{self.pose_x:.6f}',
                f'{self.pose_y:.6f}',
                f'{self.yaw:.6f}',
                f'{self.roll:.6f}',
                f'{self.pitch:.6f}',
                f'{tracking_err:.6f}',
            ]
        
        self.csv_writer.writerow(row)
        self.sample_count += 1
        
        # Periodic status update
        if self.sample_count % 50 == 0:
            rospy.loginfo(f'[ControlDataLogger] Logged {self.sample_count} samples ({elapsed:.1f}s)')
    
    def shutdown(self):
        rospy.loginfo(f'[ControlDataLogger] Shutting down. Total samples: {self.sample_count}')
        self.csv_file.close()
        rospy.loginfo(f'[ControlDataLogger] Data saved to: {self.csv_path}')
    
    def spin(self):
        rospy.spin()


def main():
    try:
        logger = ControlDataLogger()
        logger.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
