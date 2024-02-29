#!/home/hushouyue/miniconda3/envs/glasses/bin/python
   
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge 
import cv2
import numpy as np

class ImagePublisher(Node):
  def __init__(self):

    super().__init__('image_publisher')
    self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
       
    timer_period = 0.1  
       
    self.timer = self.create_timer(timer_period, self.timer_callback)
    self.cap = np.random.randint(255, size=(2,2,3),dtype=np.uint8)

    self.br = CvBridge()
    
  def timer_callback(self):
    frame = np.random.randint(255, size=(900,800,3),dtype=np.uint8)
           
    self.publisher_.publish(self.br.cv2_to_imgmsg(frame, encoding='rgb8'))

    self.get_logger().info('Publishing video frame')
   
def main(args=None):
  rclpy.init(args=args)

  image_publisher = ImagePublisher()

  rclpy.spin(image_publisher)
  image_publisher.destroy_node()
  rclpy.shutdown()
   
if __name__ == '__main__':
  main()