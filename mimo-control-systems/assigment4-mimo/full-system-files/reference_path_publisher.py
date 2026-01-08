import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import numpy as np
from std_msgs.msg import Float32MultiArray

class PathPublisher:
    def __init__(self, num_points=10):
        self.deviation = 0
        self.angle = 0
        self.num_points = num_points

        # Initialize the node
        rospy.init_node('path_publisher')

        heading_topic = rospy.get_param('~heading_topic', '/terrasentia/heading_error')
        distance_topic = rospy.get_param('~distance_topic', '/terrasentia/distance_error')
        self.rate_hz = rospy.get_param('~publish_rate', 10.0)

        # Create a publisher
        self.pub = rospy.Publisher('/terrasentia/path', Path, queue_size=10)

        # Create subscribers
        rospy.Subscriber(heading_topic, Float32MultiArray, self.update_angle)
        rospy.Subscriber(distance_topic, Float32MultiArray, self.update_deviation)

    def update_angle(self, msg):
        if not msg.data:
            return
        self.angle = float(msg.data[0])

    def update_deviation(self, msg):
        if not msg.data:
            return
        self.deviation = float(msg.data[0])

    def generate_path(self):
        path = Path()
        path.header.frame_id = 'base_footprint'

        for i in range(self.num_points):
            pose = PoseStamped()
            pose.pose.position.x = i
            pose.pose.position.y = self.deviation

            x_rotated = pose.pose.position.x * np.cos(self.angle) - pose.pose.position.y * np.sin(self.angle)
            y_rotated = pose.pose.position.x * np.sin(self.angle) + pose.pose.position.y * np.cos(self.angle)

            pose.pose.position.x = x_rotated
            pose.pose.position.y = y_rotated 

            pose.pose.orientation.x = 0
            pose.pose.orientation.y = 0
            pose.pose.orientation.z = 0
            pose.pose.orientation.w = 1

            path.poses.append(pose)

        return path

    def publish_path(self):
        rate = rospy.Rate(self.rate_hz)

        while not rospy.is_shutdown():
            path = self.generate_path()
            self.pub.publish(path)
            rate.sleep()

if __name__ == '__main__':
    path_publisher = PathPublisher()
    path_publisher.publish_path()
