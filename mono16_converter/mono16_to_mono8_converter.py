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
        
        # Try importing PyWavelets, install if not available
        self._setup_pywavelets()
        
        # Initialize parameters
        self.declare_parameter('clahe_clip_limit', 2.0)
        self.declare_parameter('clahe_grid_size_x', 8)
        self.declare_parameter('clahe_grid_size_y', 8)
        self.declare_parameter('bilateral_d', 9)
        self.declare_parameter('bilateral_sigma_color', 75)
        self.declare_parameter('bilateral_sigma_space', 75)
        self.declare_parameter('input_topic', 'image_raw')
        self.declare_parameter('output_topic', 'image_raw/mono8')
        self.declare_parameter('target_fps', 10.0)
        self.declare_parameter('target_image_encoding', 'INFERNO')  # Keeping for future use
        
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
    
    def _setup_pywavelets(self):
        """Setup PyWavelets package for wavelet-based processing"""
        try:
            import pywt
            self.pywt = pywt
            self.pywt_imported = True
            self.get_logger().info("PyWavelets successfully imported")
        except ImportError:
            self.get_logger().info("PyWavelets not found, attempting to install...")
            try:
                import subprocess
                import sys
                subprocess.check_call([sys.executable, "-m", "pip", "install", "pywavelets"])
                import pywt
                self.pywt = pywt
                self.pywt_imported = True
                self.get_logger().info("PyWavelets successfully installed and imported")
            except Exception as e:
                self.get_logger().warn(f"Could not install PyWavelets: {str(e)}. Proceeding without wavelet denoising.")
                self.pywt_imported = False

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
        """Enhanced thermal processing with extreme sharpening for important features"""
        try:
            # Step 1: Normalized contrast stretching
            img_float = image.astype(np.float32)
            p_low = np.percentile(img_float, 2)
            p_high = np.percentile(img_float, 98)
            
            if p_high <= p_low:
                p_high = p_low + 1
                
            normalized = np.clip((img_float - p_low) / (p_high - p_low), 0, 1)
            img_8bit = (normalized * 255).astype(np.uint8)
            
            # Step 2: Apply wavelet denoising if available
            working_img = self._apply_wavelet_denoising(img_8bit)
            
            # Step 3: Create feature mask for important details
            feature_mask = self._create_feature_mask(working_img)
            
            # Step 4: Apply zone-based enhancement
            enhanced_img = self._apply_zone_enhancement(working_img)
            
            # Step 5: Apply multi-scale detail enhancement
            recombined = self._enhance_details(enhanced_img, feature_mask)
            
            # Step 6: Final sharpening with feature preservation
            final_img = self._apply_adaptive_sharpening(recombined, feature_mask)
            
            return final_img.astype(np.uint8)
            
        except Exception as e:
            self.get_logger().error(f"Error in image processing: {str(e)}")
            # Fallback to basic normalization
            return self._normalize_to_uint8(image)
    
    def _normalize_to_uint8(self, image):
        """Efficiently convert to 8-bit with full dynamic range"""
        img_min = image.min()
        img_max = image.max()
        if img_max == img_min:
            return np.zeros(image.shape, dtype=np.uint8)
        return ((image - img_min) * 255.0 / (img_max - img_min)).astype(np.uint8)
    
    def _apply_wavelet_denoising(self, img_8bit):
        """Apply wavelet denoising if PyWavelets is available"""
        if not hasattr(self, 'pywt_imported') or not self.pywt_imported:
            return img_8bit
            
        try:
            img_float = img_8bit.astype(np.float32)
            coeffs = self.pywt.wavedec2(img_float, 'db4', level=1)
            new_coeffs = [coeffs[0]]
            
            detail_coeffs = coeffs[1]
            sigma = np.median(np.abs(detail_coeffs[0])) / 0.6745
            threshold = sigma * 1.0  # Extremely gentle thresholding to preserve all details
            
            for i in range(1, len(coeffs)):
                coeff_details = []
                for j in range(len(coeffs[i])):
                    processed = self.pywt.threshold(coeffs[i][j], threshold, mode='soft')
                    coeff_details.append(processed)
                
                new_coeffs.append(tuple(coeff_details))
            
            denoised = self.pywt.waverec2(new_coeffs, 'db4')
            
            if denoised.shape != img_float.shape:
                denoised = cv2.resize(denoised, (img_float.shape[1], img_float.shape[0]))
            
            return np.clip(denoised, 0, 255).astype(np.uint8)
            
        except Exception as e:
            self.get_logger().error(f"Error in wavelet denoising: {str(e)}")
            return img_8bit
    
    def _create_feature_mask(self, image):
        """Create a mask highlighting important thermal features"""
        # Apply Sobel filters for gradient calculation
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Normalize gradient magnitude
        magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Identify strong gradients (important features/edges)
        _, strong_edges = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to connect nearby edges
        kernel = np.ones((3, 3), np.uint8)
        connected_edges = cv2.morphologyEx(strong_edges, cv2.MORPH_CLOSE, kernel)
        
        # Dilate to include areas around important features
        return cv2.dilate(connected_edges, kernel, iterations=1) > 0
    
    def _apply_zone_enhancement(self, image):
        """Apply zone-based enhancement to simulate thermal colormap effects"""
        # Create multiple intensity zones (inspired by color-map bands)
        zones = {
            'very_cold': image < 50,
            'cold': (image >= 50) & (image < 100),
            'neutral': (image >= 100) & (image < 150),
            'warm': (image >= 150) & (image < 200),
            'hot': image >= 200
        }
        
        # Create zone-optimized versions
        enhanced_img = np.zeros_like(image)
        
        # Create CLAHE objects with different parameters
        clahe_settings = {
            'strong': cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)),
            'medium': cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6)),
            'gentle': cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        }
        
        # Map zones to CLAHE strength
        zone_clahe_map = {
            'very_cold': 'strong',
            'cold': 'strong',
            'neutral': 'medium',
            'warm': 'gentle',
            'hot': 'gentle'
        }
        
        # Apply zone-specific processing
        for zone_name, zone_mask in zones.items():
            if np.any(zone_mask):
                zone_img = np.zeros_like(image)
                zone_img[zone_mask] = image[zone_mask]
                
                # Get appropriate CLAHE strength for this zone
                clahe_type = zone_clahe_map[zone_name]
                processed = clahe_settings[clahe_type].apply(zone_img)
                
                # Only copy non-zero values
                mask = processed > 0
                enhanced_img[mask] = processed[mask]
        
        return enhanced_img
    
    def _enhance_details(self, image, feature_mask):
        """Apply multi-scale decomposition for detail enhancement"""
        # Create multi-scale decomposition
        gaussian = cv2.GaussianBlur(image, (0, 0), 3)
        detail = cv2.subtract(image, gaussian)
        
        # Apply extreme enhancement to important features
        detail_enhanced = np.copy(detail)
        
        # Super-strong enhancement for important features
        detail_enhanced[feature_mask] = cv2.multiply(detail[feature_mask], 3.5)
        
        # Regular enhancement for other areas
        detail_enhanced[~feature_mask] = cv2.multiply(detail[~feature_mask], 1.5)
        
        # Recombine
        return cv2.add(gaussian, detail_enhanced.astype(np.uint8))
    
    def _apply_adaptive_sharpening(self, image, feature_mask):
        """Apply adaptive sharpening based on feature importance"""
        blur = cv2.GaussianBlur(image, (0, 0), 2)
        
        # Create extreme sharpening mask - combine feature mask with edge detection
        edges = cv2.Canny(image, 50, 150)
        kernel = np.ones((3, 3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        extreme_mask = np.logical_or(feature_mask, edges_dilated > 0)
        
        # Create final image
        final_img = np.copy(image)
        
        # Apply extreme unsharp masking to important features
        final_img[extreme_mask] = cv2.addWeighted(
            image[extreme_mask], 1 + 2.5,  # Extreme sharpening amount
            blur[extreme_mask], -2.5, 0
        )
        
        # Apply moderate unsharp masking to other areas
        final_img[~extreme_mask] = cv2.addWeighted(
            image[~extreme_mask], 1 + 0.8,
            blur[~extreme_mask], -0.8, 0
        )
        
        # Final adaptive denoising
        weight_map = np.ones_like(final_img, dtype=np.float32)
        weight_map[extreme_mask] = 0.2  # Apply minimal filtering to important features
        weight_map[~extreme_mask] = 0.8  # Apply more filtering to other areas
        
        # Bilateral filter with very small diameter
        filtered = cv2.bilateralFilter(final_img, d=3, sigmaColor=15, sigmaSpace=5)
        
        # Weighted combination of filtered and unfiltered
        return cv2.addWeighted(
            final_img, 1.0 - weight_map,
            filtered, weight_map, 0
        )

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

