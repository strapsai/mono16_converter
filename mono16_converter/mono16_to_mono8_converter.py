import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Mono16ToMono8Converter(Node):
    def __init__(self):
        super().__init__('mono16_to_mono8_converter')
        self.get_logger().info('Starting Enhanced Mono16 to Mono8 Converter')
        
        # Initialize parameters
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_grid_size_x', 8)
        self.declare_parameter('clahe_grid_size_y', 8)
        self.declare_parameter('bilateral_d', 9)
        self.declare_parameter('bilateral_sigma_color', 75)
        self.declare_parameter('bilateral_sigma_space', 75)
        self.declare_parameter('input_topic', 'image_raw')
        self.declare_parameter('output_topic', 'image_raw/mono8')
        self.declare_parameter('target_fps', 2.0)
        
        # Get parameters
        clahe_clip_limit = self.get_parameter('clahe_clip_limit').value
        grid_x = self.get_parameter('clahe_grid_size_x').value
        grid_y = self.get_parameter('clahe_grid_size_y').value
        self.bilateral_d = self.get_parameter('bilateral_d').value
        self.bilateral_sigma_color = self.get_parameter('bilateral_sigma_color').value
        self.bilateral_sigma_space = self.get_parameter('bilateral_sigma_space').value
        self.input_topic = self.get_parameter('input_topic').value
        self.output_topic = self.get_parameter('output_topic').value
        self.target_fps = self.get_parameter('target_fps').value
        
        # Initialize image processing components
        self.bridge = CvBridge()
        self.clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, 
                                     tileGridSize=(grid_x, grid_y))
        
        # Create publisher with default QoS
        self.publisher = self.create_publisher(
            Image,
            self.output_topic,
            10)  # Queue size of 10
        
        # Create a permanent subscription instead of temporary ones
        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10)  # Queue size of 10
        
        # Store the latest image
        self.latest_image = None
        
        # Create a timer for periodic processing at the desired rate
        process_period = 1.0 / self.target_fps  # Convert fps to seconds
        self.timer = self.create_timer(process_period, self.process_and_publish)
        
        self.get_logger().info(f'Subscribed to: {self.input_topic}')
        self.get_logger().info(f'Publishing to: {self.output_topic} at {self.target_fps} fps')
        
        # Add a debug message to verify the node is running
        self.get_logger().info('Node initialized successfully')

    def image_callback(self, msg):
        """Store the latest image message"""
        self.latest_image = msg

    def intensity_binding(self, image):
        """Optimize contrast by clipping to 1% and 99% percentiles"""
        flat_img = image.ravel()
        k1 = int(0.01 * flat_img.size)
        k99 = int(0.99 * flat_img.size)
        p1 = np.partition(flat_img, k1)[k1]
        p99 = np.partition(flat_img, k99)[k99]
        return np.clip(image, p1, p99)

    def normalize_to_uint8(self, image):
        """Efficiently convert to 8-bit with full dynamic range"""
        img_min = image.min()
        img_max = image.max()
        if img_max == img_min:
            return np.zeros(image.shape, dtype=np.uint8)
        return ((image - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)

    def process_image(self, image):
        """Process mono16 image with enhanced contrast focused on human recognition"""
        try:
            # Step 1: Apply more aggressive dynamic range optimization
            # Find meaningful percentiles for robust contrast stretching
            p_low = np.percentile(image, 1)  # 1st percentile
            p_high = np.percentile(image, 99)  # 99th percentile
            
            # Ensure we have a valid range
            if p_high <= p_low:
                p_high = p_low + 1
            
            # Apply contrast stretching to utilize full 8-bit range
            adapted = np.clip((image - p_low) * 255.0 / (p_high - p_low), 0, 255).astype(np.uint8)
            
            # Step 2: Apply stronger CLAHE for better local contrast
            # This helps distinguish humans from backgrounds
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(adapted)
            
            # Step 3: Apply customized sharpening specifically tuned for human features
            # Create kernel for edge detection - focusing on human-sized features
            kernel_sharpen = np.array([[-1, -1, -1],
                                       [-1, 9, -1],
                                       [-1, -1, -1]])
            
            # Apply sharpening filter - enhances edges significantly
            sharpened = cv2.filter2D(clahe_img, -1, kernel_sharpen)
            
            # Step 4: Apply very minimal noise reduction that preserves critical details
            # Use a conservative median filter with small kernel size
            # This removes speckle noise without blurring edges
            final_img = cv2.medianBlur(sharpened, 3)
            
            return final_img
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            # Fallback to basic normalization
            return self.normalize_to_uint8(image)

    def process_and_publish(self):
        """Process and publish the latest image at the timer rate"""
        if self.latest_image is None:
            self.get_logger().warn('No image received yet')
            return
            
        try:
            # Convert ROS Image message to OpenCV image
            cv_img = self.bridge.imgmsg_to_cv2(self.latest_image, 'mono16')
            
            # Process the image with enhanced contrast
            enhanced_img = self.process_image(cv_img)
            
            # Convert back to ROS Image message and publish
            ros_img = self.bridge.cv2_to_imgmsg(enhanced_img, 'mono8')
            ros_img.header = self.latest_image.header  # Preserve timestamp and frame_id
            self.publisher.publish(ros_img)
            
        except Exception as e:
            self.get_logger().error(f'Error in process_and_publish: {str(e)}')

def main(args=None):
    rclpy.init(args=args)
    node = Mono16ToMono8Converter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

