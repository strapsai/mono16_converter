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
        """Process mono16 image with enhanced contrast focused on human thermal signature"""
        try:
            # Step 1: Thermal calibration - identify human temperature range
            # Typical human temperatures in thermal imagery fall in a specific range
            # We'll enhance this range specifically to make humans stand out
            
            # First, get overall image statistics
            img_min = np.min(image)
            img_max = np.max(image)
            img_range = img_max - img_min
            
            # Analyze histogram to identify potential human regions
            hist, bins = np.histogram(image.flatten(), 256, [img_min, img_max])
            
            # Human body typically appears as warmer regions (higher values)
            # in thermal imaging, often in the upper 30-40% of the range
            # Let's identify this region and enhance it
            
            # Find the temperature threshold that might represent human body
            # This is typically in the upper temperature range but not at the very top
            human_temp_lower = img_min + img_range * 0.6  # Approximate lower bound for human temps
            human_temp_upper = img_min + img_range * 0.9  # Approximate upper bound for human temps
            
            # Create a custom grayscale mapping function that:
            # 1. Preserves overall scene context with basic contrast
            # 2. Gives the human temperature range maximum contrast
            
            # Create output image with basic contrast stretching first
            output = np.zeros_like(image, dtype=np.uint8)
            
            # Basic contrast stretch to use full 0-255 range
            # but reserve the 100-220 range for human temperatures to create distinction
            mask_below_human = image < human_temp_lower
            mask_human = (image >= human_temp_lower) & (image <= human_temp_upper)
            mask_above_human = image > human_temp_upper
            
            # Map temperatures below human range to 0-100
            if np.any(mask_below_human):
                below_min = np.min(image[mask_below_human])
                below_max = human_temp_lower
                if below_max > below_min:
                    output[mask_below_human] = ((image[mask_below_human] - below_min) * 100.0 / 
                                               (below_max - below_min)).astype(np.uint8)
            
            # Map human temperature range to 100-220 (distinct middle gray)
            # This creates maximum contrast with background
            if np.any(mask_human):
                human_min = human_temp_lower
                human_max = human_temp_upper
                if human_max > human_min:
                    output[mask_human] = 100 + ((image[mask_human] - human_min) * 120.0 / 
                                               (human_max - human_min)).astype(np.uint8)
            
            # Map temperatures above human range to 220-255
            if np.any(mask_above_human):
                above_min = human_temp_upper
                above_max = np.max(image[mask_above_human])
                if above_max > above_min:
                    output[mask_above_human] = 220 + ((image[mask_above_human] - above_min) * 35.0 / 
                                                     (above_max - above_min)).astype(np.uint8)
            
            # Apply gentle edge enhancement
            kernel_sharpen = np.array([[-0.5, -0.5, -0.5],
                                       [-0.5,  5.0, -0.5],
                                       [-0.5, -0.5, -0.5]])
            
            sharpened = cv2.filter2D(output, -1, kernel_sharpen)
            
            # Apply a conservative noise reduction
            # Bilateral filter preserves edges while reducing noise
            final_img = cv2.bilateralFilter(sharpened, 5, 25, 5)
            
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

