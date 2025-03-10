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
        """Process mono16 image with enhanced contrast while preserving details"""
        try:
            # Step 1: Calculate histogram for dynamic range adaptation
            hist, bins = np.histogram(image.flatten(), 256, [0, 65535])
            cdf = hist.cumsum()
            
            # Find meaningful percentiles for dynamic range adaptation
            if cdf[-1] > 0:  # Avoid division by zero
                cdf_normalized = cdf * 65535 / cdf[-1]
                
                # Adaptive percentile selection based on image characteristics
                # Analyze thermal distribution to select optimal percentiles
                img_std = np.std(image)
                
                # For higher contrast scenes, use tighter bounds to avoid saturation
                min_percentile = 0.02  # Start with 2% percentile
                max_percentile = 0.98  # Start with 98% percentile
                
                # Adjust percentiles based on image statistics
                if img_std > 5000:  # High variance/contrast scene
                    min_percentile = 0.05
                    max_percentile = 0.95
                elif img_std < 1000:  # Low variance/contrast scene
                    min_percentile = 0.01
                    max_percentile = 0.99
                
                # Get intensity values at these percentiles
                lower_val = np.searchsorted(cdf_normalized, min_percentile * 65535)
                upper_val = np.searchsorted(cdf_normalized, max_percentile * 65535)
                
                # Ensure we have a reasonable range
                if upper_val <= lower_val:
                    upper_val = lower_val + 1
            else:
                # Fallback to simple min/max
                lower_val = np.min(image)
                upper_val = np.max(image)
                if upper_val <= lower_val:
                    upper_val = lower_val + 1
            
            # Step 2: Apply dynamic range adaptation
            adapted_img = np.clip((image - lower_val) * 65535 / (upper_val - lower_val), 0, 65535).astype(np.uint16)
            
            # Step 3: Convert to 8-bit
            img_8bit = (adapted_img / 256).astype(np.uint8)
            
            # Step 4: Apply CLAHE for local contrast enhancement
            clahe_img = self.clahe.apply(img_8bit)
            
            # Step 5: Apply a combination of filters to preserve edges while reducing noise
            # First, apply a small Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(clahe_img, (3, 3), 0)
            
            # Then apply a mild bilateral filter with carefully tuned parameters
            # This will preserve edges better than the previous implementation
            # Lower d (diameter) and appropriate sigma values prevent excessive blurring
            bilateral = cv2.bilateralFilter(
                blurred, 
                d=5,  # Smaller neighborhood
                sigmaColor=25,  # Reduced from 75
                sigmaSpace=5    # Reduced from 75
            )
            
            # Step 6: Apply unsharp masking to enhance edges
            # Create a Gaussian blurred version for the mask
            gaussian = cv2.GaussianBlur(bilateral, (5, 5), 1.0)
            # Apply unsharp mask (enhance edges by subtracting blurred image)
            unsharp_amount = 0.6  # Controls the enhancement intensity (0.5-1.5 is typical)
            final_img = cv2.addWeighted(bilateral, 1 + unsharp_amount, gaussian, -unsharp_amount, 0)
            
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

