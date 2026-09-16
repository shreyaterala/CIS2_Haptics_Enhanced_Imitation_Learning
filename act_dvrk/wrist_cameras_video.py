#!/usr/bin/env python3
#
# wrist_cameras_video_new.py
#
# Captures frames from two USB wrist cameras and publishes them as standard
# sensor_msgs/Image ROS 2 topics so they are recorded inline with all other
# data by `ros2 bag record`. No separate .mp4 or timestamp file is produced.
#
# Topics published:
#   /wrist/cam1/image_raw   (sensor_msgs/msg/Image, rgb8)
#   /wrist/cam2/image_raw   (sensor_msgs/msg/Image, rgb8)
#
# Usage:
#   python wrist_cameras_video_new.py [--cam1 0] [--cam2 2]

import argparse
import sys

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vidgear.gears import VideoGear


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class WristCameraNode(Node):

    def __init__(self, cam1_channel: int, cam2_channel: int, fps: float):
        super().__init__("wrist_cameras")

        self.pub1 = None
        self.pub2 = None
        self.stream1 = None
        self.stream2 = None

        self._init_camera("cam1", cam1_channel, "/wrist/cam1/image_raw")
        self._init_camera("cam2", cam2_channel, "/wrist/cam2/image_raw")

        if not self.stream1 and not self.stream2:
            self.get_logger().error("No cameras successfully detected! Exiting.")
            raise SystemExit

        period = 1.0 / fps
        self.timer = self.create_timer(period, self.timer_cb)
        self.get_logger().info(f"Publishing at {fps:.1f} Hz")

    def _init_camera(self, name, channel, topic):
        if channel < 0:
            self.get_logger().info(f"{name} disabled via channel={channel}")
            return
        try:
            stream = VideoGear(source=channel, logging=False).start()
            if stream.read() is not None:
                if name == "cam1":
                    self.stream1 = stream
                    self.pub1 = self.create_publisher(Image, topic, 10)
                else:
                    self.stream2 = stream
                    self.pub2 = self.create_publisher(Image, topic, 10)
                self.get_logger().info(f"Successfully started {name} on channel {channel}")
            else:
                stream.stop()
                self.get_logger().warn(f"{name} (channel {channel}) returned None frame. Disabling.")
        except Exception as e:
            self.get_logger().warn(f"Failed to initialize {name} on channel {channel}: {e}")

    def _to_ros_image(self, frame, topic_name: str) -> Image:
        """Convert an OpenCV BGR frame to a sensor_msgs/Image."""
        
        # Convert BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = topic_name
        msg.height = frame_rgb.shape[0]
        msg.width  = frame_rgb.shape[1]
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = frame_rgb.shape[1] * 3
        msg.data = frame_rgb.tobytes()
        return msg

    def timer_cb(self):
        if self.stream1:
            frame1 = self.stream1.read()
            if frame1 is not None:
                self.pub1.publish(self._to_ros_image(frame1, "wrist_cam1"))
                cv2.imshow("Wrist Cam 1", frame1)
            else:
                 self.get_logger().warn("cam1 dropped frame")

        if self.stream2:
            frame2 = self.stream2.read()
            if frame2 is not None:
                self.pub2.publish(self._to_ros_image(frame2, "wrist_cam2"))
                cv2.imshow("Wrist Cam 2", frame2)
            else:
                 self.get_logger().warn("cam2 dropped frame")

        if cv2.waitKey(1) & 0xFF == ord("q"):
            raise SystemExit

    def destroy_node(self):
        if self.stream1:
            self.stream1.stop()
        if self.stream2:
            self.stream2.stop()
        cv2.destroyAllWindows()
        super().destroy_node()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Publish wrist camera feeds as ROS 2 Image topics."
    )
    parser.add_argument("--cam1", type=int, default=0,  help="V4L2 index for wrist cam 1 (-1 to disable)")
    parser.add_argument("--cam2", type=int, default=2,  help="V4L2 index for wrist cam 2 (-1 to disable)")
    parser.add_argument("--fps",  type=float, default=30.0, help="Publishing rate in Hz")
    args = parser.parse_args()

    rclpy.init(args=sys.argv[:1])   # don't pass argparse args to RCL
    node = WristCameraNode(args.cam1, args.cam2, args.fps)

    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()