import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

class XRayStitcher:
    def __init__(self, feature_method='sift'):
        """
        Initialize the X-ray stitcher.
        
        Args:
            feature_method: Feature detection method ('sift', 'orb', or 'akaze')
        """
        self.feature_method = feature_method
        
        self.detector = cv2.SIFT.create(nfeatures=5000, contrastThreshold=0.03, edgeThreshold=15, sigma=1.6)
        # self.detector = cv2.ORB.create(nfeatures=2000)

        # Initialize feature detector
        # if feature_method == 'sift':
        #     self.detector = cv2.SIFT.create()
        # elif feature_method == 'orb':
        #     self.detector = cv2.ORB.create(nfeatures=2000)
        # elif feature_method == 'akaze':
        #     self.detector = cv2.AKAZE.create()
        # else:
        #     raise ValueError(f"Unsupported feature method: {feature_method}")
            
        # Initialize feature matcher
        self.matcher = cv2.BFMatcher()
    
    def load_images(self, img1_path, img2_path):
        """Load and preprocess X-ray images."""
        def load_and_compress(img_path):
            """Load image, converting to JPEG format if not already."""
            # Define the JPEG version path
            jpeg_path = img_path.rsplit('.', 1)[0] + '.jpg'
            
            # If JPEG version already exists, use it
            if os.path.exists(jpeg_path) and not img_path.endswith('.jpg'):
                print(f"Using existing JPEG version: {jpeg_path}")
                return cv2.imread(jpeg_path, cv2.IMREAD_UNCHANGED)
            
            # Read original image
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            
            # Handle TIFF files
            if img is None and img_path.lower().endswith(('.tiff', '.tif')):
                try:
                    import tifffile
                    img = tifffile.imread(img_path)
                except ImportError:
                    pass
            
            if img is not None:
                # Resize if image is large
                if img.shape[0] > 2048 or img.shape[1] > 2048:
                    # Calculate new dimensions
                    max_dimension = 2048
                    height, width = img.shape[:2]
                    scale = min(max_dimension / width, max_dimension / height)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
                
                # Save as JPEG if not already a JPEG
                if not img_path.endswith('.jpg'):
                    cv2.imwrite(jpeg_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    print(f"Saved JPEG version to: {jpeg_path}")
                    return cv2.imread(jpeg_path, cv2.IMREAD_UNCHANGED)
            
            return img
        
        # Load both images
        img1 = load_and_compress(img1_path)
        img2 = load_and_compress(img2_path)
        
        if img1 is None or img2 is None:
            raise ValueError("Failed to load images")
        
        # Convert to 8-bit if needed
        if img1.dtype != np.uint8:
            img1 = ((img1 - img1.min()) * (255.0 / (img1.max() - img1.min()))).astype(np.uint8)
        if img2.dtype != np.uint8:
            img2 = ((img2 - img2.min()) * (255.0 / (img2.max() - img2.min()))).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img1.shape) > 2:
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        if len(img2.shape) > 2:
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Store original images for visualization
        self.img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
        self.img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
        
        # Apply CLAHE to enhance contrast (often helpful for X-rays)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.img1 = clahe.apply(img1)
        self.img2 = clahe.apply(img2)
        
        return self.img1, self.img2
    
    def detect_and_match_features(self):
        """Enhanced feature detection with multi-scale approach for anatomical structures."""
        # Create multiple scaled versions of the images
        scales = [0.5, 0.75, 1.0, 1.5, 2.0]
        all_kp1, all_des1 = [], []
        all_kp2, all_des2 = [], []
        
        for scale in scales:
            # Resize images for multi-scale detection
            if scale != 1.0:
                h1, w1 = self.img1.shape
                h2, w2 = self.img2.shape
                resized1 = cv2.resize(self.img1, (int(w1 * scale), int(h1 * scale)))
                resized2 = cv2.resize(self.img2, (int(w2 * scale), int(h2 * scale)))
            else:
                resized1, resized2 = self.img1, self.img2
            
            # Detect keypoints at this scale
            kp1 = self.detector.detect(resized1, None)
            kp2 = self.detector.detect(resized2, None)
            
            # Compute descriptors
            kp1, des1 = self.detector.compute(resized1, kp1)
            kp2, des2 = self.detector.compute(resized2, kp2)
            
            # Adjust keypoint coordinates back to original scale
            if scale != 1.0:
                for kp in kp1:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
                for kp in kp2:
                    kp.pt = (kp.pt[0] / scale, kp.pt[1] / scale)
            
            # Add to collection
            if des1 is not None and des2 is not None:
                all_kp1.extend(kp1)
                all_des1.append(des1)
                all_kp2.extend(kp2)
                all_des2.append(des2)
        
        # Combine descriptors
        if all_des1 and all_des2:
            des1 = np.vstack(all_des1)
            des2 = np.vstack(all_des2)
            
            print(f"Detected {len(all_kp1)} keypoints in first image across scales")
            print(f"Detected {len(all_kp2)} keypoints in second image across scales")
            
            self.kp1, self.kp2 = all_kp1, all_kp2
            
            # Match with more relaxed parameters
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply relaxed ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.95 * n.distance:  # Very relaxed ratio
                    good_matches.append(m)
            
            print(f"Found {len(good_matches)} good matches")
            self.matches = good_matches
            
            # Extract matched keypoints
            if len(good_matches) >= 4:
                src_pts = np.array([[all_kp1[m.queryIdx].pt[0], all_kp1[m.queryIdx].pt[1]] 
                                   for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([[all_kp2[m.trainIdx].pt[0], all_kp2[m.trainIdx].pt[1]] 
                                   for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
            else:
                print("Warning: Very few matches found.")
                src_pts = np.zeros((0, 1, 2), dtype=np.float32)
                dst_pts = np.zeros((0, 1, 2), dtype=np.float32)
            
            return src_pts, dst_pts, good_matches
        else:
            print("No features detected across scales")
            return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((0, 1, 2), dtype=np.float32), []
    
    def compute_homography(self, src_pts, dst_pts):
        """Compute the homography matrix with more tolerant parameters."""
        if len(src_pts) < 4:
            print("Not enough matches to compute homography. Using identity matrix.")
            self.H = np.eye(3, dtype=np.float32)
            self.inliers_mask = np.ones((len(src_pts), 1), dtype=np.uint8)
            return self.H, 0
        
        # Find homography with more tolerant parameters
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 8.0)  # Increased threshold
        
        if H is None:
            print("Could not compute homography. Using identity matrix.")
            H = np.eye(3, dtype=np.float32)
            mask = np.ones((len(src_pts), 1), dtype=np.uint8)
        
        self.H = np.float32(H)
        self.inliers_mask = mask
        
        # Count inliers
        inlier_count = np.sum(mask)
        total_matches = len(mask)
        inlier_ratio = inlier_count / total_matches if total_matches > 0 else 0
        
        print(f"Inlier ratio: {inlier_ratio:.2f} ({inlier_count}/{total_matches})")
        
        return self.H, inlier_ratio
    
    def warp_and_stitch(self):
        """Warp and stitch the images based on the computed homography."""
        # Get dimensions
        h1, w1 = self.img1.shape
        h2, w2 = self.img2.shape
        
        # Create the panorama canvas
        corners1 = np.array([[0, 0], [0, h1], [w1, h1], [w1, 0]], dtype=np.float32).reshape(-1, 1, 2)
        corners2 = np.array([[0, 0], [0, h2], [w2, h2], [w2, 0]], dtype=np.float32).reshape(-1, 1, 2)
        
        # Transform corners of first image
        warped_corners1 = cv2.perspectiveTransform(corners1, np.array(self.H))
        all_corners = np.concatenate([warped_corners1, corners2], axis=0)
        
        # Find min and max coordinates
        min_corner = all_corners.min(axis=0).ravel() - 0.5
        max_corner = all_corners.max(axis=0).ravel() + 0.5
        x_min, y_min = int(min_corner[0]), int(min_corner[1])
        x_max, y_max = int(max_corner[0]), int(max_corner[1])
        
        # Translation matrix to shift to positive coordinates
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        H_adjusted = np.matmul(translation, self.H)
        
        # Warp first image
        output_size = (x_max - x_min, y_max - y_min)
        warped_img1 = cv2.warpPerspective(self.img1_color, H_adjusted, output_size)
        
        # Create a mask for the first warped image
        mask1 = np.ones((h1, w1), dtype=np.uint8) * 255
        warped_mask1 = cv2.warpPerspective(mask1, H_adjusted, output_size)
        
        # Create a translation matrix for the second image
        T = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]], dtype=np.float32)
        
        # Warp the second image
        warped_img2 = cv2.warpPerspective(self.img2_color, T, output_size)
        
        # Create a mask for the second image
        mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
        warped_mask2 = cv2.warpPerspective(mask2, T, output_size)
        
        # Create the final result
        result = np.zeros_like(warped_img1)
        
        # Blend the images in the overlapping region
        for y in range(output_size[1]):
            for x in range(output_size[0]):
                if warped_mask1[y, x] > 0 and warped_mask2[y, x] > 0:
                    # Overlapping region - take average
                    result[y, x] = (warped_img1[y, x].astype(float) + 
                                  warped_img2[y, x].astype(float)) / 2
                elif warped_mask1[y, x] > 0:
                    # Only first image
                    result[y, x] = warped_img1[y, x]
                elif warped_mask2[y, x] > 0:
                    # Only second image
                    result[y, x] = warped_img2[y, x]
        
        self.result = result
        return result
    
    def warp_and_stitch_non_rigid(self):
        """Warp and stitch the images using non-rigid transformation."""
        # Get dimensions
        h1, w1 = self.img1.shape
        h2, w2 = self.img2.shape
        
        # Create a larger canvas to accommodate the warped images
        max_width = max(w1, w2) * 2
        max_height = max(h1, h2) * 2
        
        # Create masks for both images
        mask1 = np.ones((h1, w1), dtype=np.uint8) * 255
        mask2 = np.ones((h2, w2), dtype=np.uint8) * 255
        
        # Apply TPS transformation to the first image
        warped_img1 = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        warped_mask1 = np.zeros((max_height, max_width), dtype=np.uint8)
        
        # Apply the transformation
        # Note: This is a simplified version - actual implementation would use the TPS transformer
        # For demonstration, we'll use a placeholder that would be replaced with actual TPS warping
        warped_img1[50:50+h1, 50:50+w1] = self.img1_color
        warped_mask1[50:50+h1, 50:50+w1] = mask1
        
        # Place the second image on the canvas
        warped_img2 = np.zeros((max_height, max_width, 3), dtype=np.uint8)
        warped_mask2 = np.zeros((max_height, max_width), dtype=np.uint8)
        
        # Position the second image with some overlap
        offset_x = int(w1 * 0.7)
        offset_y = int(h1 * 0.1)  # Adjust for vertical alignment
        
        warped_img2[offset_y:offset_y+h2, offset_x:offset_x+w2] = self.img2_color
        warped_mask2[offset_y:offset_y+h2, offset_x:offset_x+w2] = mask2
        
        # Apply exposure compensation
        warped_img1, warped_img2 = self.exposure_compensate(warped_img1, warped_img2, warped_mask1, warped_mask2)
        
        # Create the final result
        result = np.zeros_like(warped_img1)
        
        # Blend the images with weighted average in the overlapping region
        for y in range(max_height):
            for x in range(max_width):
                if warped_mask1[y, x] > 0 and warped_mask2[y, x] > 0:
                    # Overlapping region - weighted average
                    # Create a gradual transition
                    alpha = (x - offset_x) / w2 if x > offset_x else 0
                    alpha = max(0, min(1, alpha))  # Clamp between 0 and 1
                    
                    result[y, x] = (1 - alpha) * warped_img1[y, x] + alpha * warped_img2[y, x]
                elif warped_mask1[y, x] > 0:
                    # Only first image
                    result[y, x] = warped_img1[y, x]
                elif warped_mask2[y, x] > 0:
                    # Only second image
                    result[y, x] = warped_img2[y, x]
        
        # Crop the result to remove unnecessary black borders
        gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find bounding box of all non-zero pixels
            x, y, w, h = cv2.boundingRect(contours[0])
            for cnt in contours[1:]:
                x2, y2, w2, h2 = cv2.boundingRect(cnt)
                x = min(x, x2)
                y = min(y, y2)
                w = max(x + w, x2 + w2) - x
                h = max(y + h, y2 + h2) - y
            
            # Crop the result
            result = result[y:y+h, x:x+w]
        
        self.result = result
        return result
    
    def exposure_compensate(self, img1, img2, mask1, mask2):
        """Compensate exposure differences between overlapping regions."""
        # Find overlapping region
        overlap = cv2.bitwise_and(mask1, mask2)
        
        if np.sum(overlap) > 0:
            # Calculate mean intensities in overlapping region
            mean1 = np.mean(img1[overlap > 0])
            mean2 = np.mean(img2[overlap > 0])
            
            # Calculate scaling factor
            scale = mean1 / mean2 if mean2 > 0 else 1.0
            
            # Apply compensation to second image
            img2_compensated = np.clip(img2 * scale, 0, 255).astype(np.uint8)
            return img1, img2_compensated
        
        return img1, img2
    
    def visualize_matches(self, save_path=None):
        """Visualize the feature matches between images."""
        # Filter matches using the inliers mask
        inlier_matches = [self.matches[i] for i in range(len(self.matches)) 
                          if self.inliers_mask[i] == 1]
        
        # Draw matches
        match_img = cv2.drawMatches(self.img1, self.kp1, self.img2, self.kp2, 
                                    inlier_matches, outImg=np.array([]), 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matches ({len(inlier_matches)} inliers)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    def visualize_result(self, save_path=None):
        """Visualize the stitching result."""
        plt.figure(figsize=(15, 10))
        plt.imshow(cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB))
        plt.title("Stitched X-ray Images")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
            # Also save as image file
            cv2.imwrite(save_path, self.result)
        
        plt.show()
    
    def preprocess_xray(self, img, xray_type='general'):
        """Enhanced preprocessing with anatomical region emphasis."""
        # Convert to 8-bit if needed
        if img.dtype != np.uint8:
            img = ((img - img.min()) * (255.0 / (img.max() - img.min()))).astype(np.uint8)
        
        # Convert to grayscale if needed
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply multi-scale enhancement
        processed = img.copy()
        
        # Apply CLAHE with different parameters
        clahe1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        clahe2 = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        
        enhanced1 = clahe1.apply(processed)
        enhanced2 = clahe2.apply(processed)
        
        # Combine enhancements
        processed = cv2.addWeighted(enhanced1, 0.5, enhanced2, 0.5, 0)
        
        # Apply anatomical region enhancement based on type
        if xray_type == 'spine':
            # Enhance vertical structures (spine)
            kernel_v = np.ones((5, 1), np.float32) / 5
            vertical = cv2.filter2D(processed, -1, kernel_v)
            processed = cv2.addWeighted(processed, 0.7, vertical, 0.3, 0)
        elif xray_type == 'head':
            # Enhance circular structures (skull)
            kernel_size = 9
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            processed = cv2.morphologyEx(processed, cv2.MORPH_TOPHAT, kernel)
        elif xray_type == 'neck':
            # Enhance both vertical and horizontal structures
            kernel_v = np.ones((7, 1), np.float32) / 7
            kernel_h = np.ones((1, 7), np.float32) / 7
            vertical = cv2.filter2D(processed, -1, kernel_v)
            horizontal = cv2.filter2D(processed, -1, kernel_h)
            processed = cv2.addWeighted(vertical, 0.5, horizontal, 0.5, 0)
        
        # Apply edge enhancement for all types
        edges = cv2.Canny(processed, 50, 150)
        # Convert both arrays to float32 before blending
        processed = cv2.addWeighted(processed.astype(np.float32), 0.8, edges.astype(np.float32), 0.2, 0).astype(np.uint8)
        
        return processed
    
    def detect_anatomical_regions(self, xray_type):
        """Detect anatomical regions specific to the X-ray type."""
        # Create specialized detectors based on anatomical knowledge
        if xray_type == 'neck':
            # For neck, focus on cervical vertebrae and soft tissue
            # Apply Hough transform to detect lines (vertebrae)
            edges1 = cv2.Canny(self.img1, 50, 150, apertureSize=3)
            edges2 = cv2.Canny(self.img2, 50, 150, apertureSize=3)
            
            lines1 = cv2.HoughLinesP(edges1, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            lines2 = cv2.HoughLinesP(edges2, 1, np.pi/180, 50, minLineLength=50, maxLineGap=10)
            
            # Convert lines to keypoints
            kp1, kp2 = [], []
            if lines1 is not None:
                for line in lines1:
                    x1, y1, x2, y2 = map(float, line[0])  # Convert to float
                    kp1.append(cv2.KeyPoint(x1, y1, 10))
                    kp1.append(cv2.KeyPoint(x2, y2, 10))
            
            if lines2 is not None:
                for line in lines2:
                    x1, y1, x2, y2 = map(float, line[0])  # Convert to float
                    kp2.append(cv2.KeyPoint(x1, y1, 10))
                    kp2.append(cv2.KeyPoint(x2, y2, 10))
        
        elif xray_type == 'spine':
            # For spine, detect vertebral bodies
            # Use blob detection to find vertebrae
            params = cv2.SimpleBlobDetector.Params()
            params.filterByArea = True
            params.minArea = 100
            params.maxArea = 5000
            params.filterByCircularity = True
            params.minCircularity = 0.1
            params.filterByConvexity = True
            params.minConvexity = 0.5
            
            detector = cv2.SimpleBlobDetector.create(params)
            kp1 = detector.detect(self.img1)
            kp2 = detector.detect(self.img2)
        
        else:
            # Fall back to SIFT for other types
            kp1 = self.detector.detect(self.img1, None)
            kp2 = self.detector.detect(self.img2, None)
        
        # Compute descriptors using SIFT
        kp1, des1 = self.detector.compute(self.img1, kp1)
        kp2, des2 = self.detector.compute(self.img2, kp2)
        
        self.kp1, self.kp2 = kp1, kp2
        
        # Match with very relaxed parameters for anatomical structures
        if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
            matches = self.matcher.knnMatch(des1, des2, k=2)
            
            # Apply very relaxed ratio test for anatomical features
            good_matches = []
            for m, n in matches:
                if m.distance < 0.98 * n.distance:  # Extremely relaxed ratio
                    good_matches.append(m)
            
            print(f"Found {len(good_matches)} anatomical region matches")
            self.matches = good_matches
            
            # Extract matched keypoints
            if len(good_matches) >= 4:
                src_pts = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] 
                                   for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                dst_pts = np.array([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] 
                                   for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
            else:
                src_pts = np.zeros((0, 1, 2), dtype=np.float32)
                dst_pts = np.zeros((0, 1, 2), dtype=np.float32)
            
            return src_pts, dst_pts, good_matches
        else:
            return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((0, 1, 2), dtype=np.float32), []
    
    def check_spatial_consistency(self, src_pts, dst_pts, good_matches):
        """
        Check spatial consistency of matches by analyzing local neighborhoods.
        Discards matches that don't have consistent neighbors.
        """
        if len(good_matches) < 10:
            return src_pts, dst_pts, good_matches
        
        # Convert points to more manageable format
        src_points = src_pts.reshape(-1, 2)
        dst_points = dst_pts.reshape(-1, 2)
        
        # Parameters
        neighborhood_radius = 50  # Pixels
        min_consistent_neighbors = 3
        consistent_matches = []
        consistent_src_pts = []
        consistent_dst_pts = []
        
        # For each match, check if it has consistent neighbors
        for i, match in enumerate(good_matches):
            # Current match points
            src_pt = src_points[i]
            dst_pt = dst_points[i]
            
            # Find neighbors in source image
            neighbor_indices = []
            for j, other_src in enumerate(src_points):
                if i != j and np.linalg.norm(src_pt - other_src) < neighborhood_radius:
                    neighbor_indices.append(j)
            
            # Check if these neighbors are also consistent in destination image
            consistent_neighbors = 0
            for j in neighbor_indices:
                # Vector between points in source image
                src_vector = src_points[j] - src_pt
                # Expected vector in destination image
                dst_vector = dst_points[j] - dst_pt
                
                # Check if vectors are similar (direction and magnitude)
                vector_similarity = np.dot(src_vector, dst_vector) / (np.linalg.norm(src_vector) * np.linalg.norm(dst_vector) + 1e-6)
                ratio_of_magnitudes = np.linalg.norm(dst_vector) / (np.linalg.norm(src_vector) + 1e-6)
                
                if vector_similarity > 0.7 and 0.7 < ratio_of_magnitudes < 1.3:
                    consistent_neighbors += 1
            
            # Keep match if it has enough consistent neighbors
            if consistent_neighbors >= min_consistent_neighbors:
                consistent_matches.append(good_matches[i])
                consistent_src_pts.append(src_pt)
                consistent_dst_pts.append(dst_pt)
        
        print(f"Spatial consistency check: {len(consistent_matches)}/{len(good_matches)} matches retained")
        
        # Convert back to original format
        if len(consistent_matches) >= 4:
            consistent_src_pts = np.array(consistent_src_pts, dtype=np.float32).reshape(-1, 1, 2)
            consistent_dst_pts = np.array(consistent_dst_pts, dtype=np.float32).reshape(-1, 1, 2)
            return consistent_src_pts, consistent_dst_pts, consistent_matches
        else:
            # Not enough consistent matches, return original
            return src_pts, dst_pts, good_matches
    
    def visualize_match_context(self, save_path=None, max_windows=5):
        """Visualize the context around matches to help understand spatial relationships."""
        if not hasattr(self, 'matches') or len(self.matches) == 0:
            print("No matches to visualize")
            return
        
        # Create a copy of the images for visualization
        img1_vis = self.img1_color.copy()
        img2_vis = self.img2_color.copy()
        
        # Draw context regions around keypoints
        context_radius = 30
        
        # Only show a limited number of matches
        matches_to_show = self.matches[:min(max_windows, len(self.matches))]
        
        # Create a single figure for all patches
        if matches_to_show:
            fig, axes = plt.subplots(len(matches_to_show), 2, figsize=(8, 2*len(matches_to_show)))
            if len(matches_to_show) == 1:
                axes = [axes]  # Make it iterable for single match case
            
            for i, m in enumerate(matches_to_show):
                # Get keypoint coordinates
                x1, y1 = map(int, self.kp1[m.queryIdx].pt)
                x2, y2 = map(int, self.kp2[m.trainIdx].pt)
                
                # Draw circles around keypoints
                cv2.circle(img1_vis, (x1, y1), context_radius, (0, 255, 0), 2)
                cv2.circle(img2_vis, (x2, y2), context_radius, (0, 255, 0), 2)
                
                # Extract context patches
                patch1 = self.img1[max(0, y1-context_radius):min(self.img1.shape[0], y1+context_radius), 
                                  max(0, x1-context_radius):min(self.img1.shape[1], x1+context_radius)]
                patch2 = self.img2[max(0, y2-context_radius):min(self.img2.shape[0], y2+context_radius), 
                                  max(0, x2-context_radius):min(self.img2.shape[1], x2+context_radius)]
                
                # Resize patches to be the same size for visualization
                if patch1.size > 0 and patch2.size > 0:
                    patch1 = cv2.resize(patch1, (context_radius*2, context_radius*2))
                    patch2 = cv2.resize(patch2, (context_radius*2, context_radius*2))
                    
                    # Display in matplotlib instead of cv2.imshow
                    axes[i][0].imshow(cv2.cvtColor(patch1, cv2.COLOR_GRAY2RGB))
                    axes[i][0].set_title(f"Image 1 - Point {m.queryIdx}")
                    axes[i][0].axis('off')
                    
                    axes[i][1].imshow(cv2.cvtColor(patch2, cv2.COLOR_GRAY2RGB))
                    axes[i][1].set_title(f"Image 2 - Point {m.trainIdx}")
                    axes[i][1].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        # Draw all matches
        match_img = cv2.drawMatches(img1_vis, self.kp1, img2_vis, self.kp2, 
                                    self.matches, outImg=np.array([]), 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        plt.figure(figsize=(15, 8))
        plt.imshow(cv2.cvtColor(match_img, cv2.COLOR_BGR2RGB))
        plt.title(f"Feature Matches with Context ({len(self.matches)} matches)")
        plt.axis('off')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
    
    def detect_shape_features(self, img1, img2):
        """Detect broader shape patterns in images using contour analysis."""
        # Convert to binary images for contour detection
        _, thresh1 = cv2.threshold(img1, 127, 255, cv2.THRESH_BINARY)
        _, thresh2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours1, _ = cv2.findContours(thresh1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(thresh2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 100
        max_area = 10000
        contours1 = [c for c in contours1 if min_area < cv2.contourArea(c) < max_area]
        contours2 = [c for c in contours2 if min_area < cv2.contourArea(c) < max_area]
        
        # Create keypoints from contour centers
        kp1 = []
        for contour in contours1:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                # Use contour area to determine keypoint size
                area = cv2.contourArea(contour)
                size = np.sqrt(area) * 0.5
                kp1.append(cv2.KeyPoint(cx, cy, size))
        
        kp2 = []
        for contour in contours2:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                size = np.sqrt(area) * 0.5
                kp2.append(cv2.KeyPoint(cx, cy, size))
        
        print(f"Detected {len(kp1)} shape features in first image")
        print(f"Detected {len(kp2)} shape features in second image")
        
        return kp1, kp2
    
    def compute_shape_descriptors(self, img, keypoints, contours):
        """Compute descriptors for shapes based on contour properties."""
        if not keypoints:
            return None
        
        # Create a descriptor for each keypoint based on shape properties
        descriptors = []
        for kp, contour in zip(keypoints, contours):
            # Extract shape features
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h if h > 0 else 1
            extent = float(area) / (w * h) if w * h > 0 else 0
            
            # Create a simple descriptor vector
            descriptor = np.array([
                area, 
                perimeter, 
                aspect_ratio,
                extent,
                kp.pt[0] / img.shape[1],  # Normalized x position
                kp.pt[1] / img.shape[0],  # Normalized y position
            ], dtype=np.float32)
            
            descriptors.append(descriptor)
        
        return np.array(descriptors)
    
    def detect_and_match_patterns(self):
        """Detect and match broader patterns between images."""
        # Apply edge detection to highlight structures
        edges1 = cv2.Canny(self.img1, 50, 150)
        edges2 = cv2.Canny(self.img2, 50, 150)
        
        # Apply morphological operations to connect edges into shapes
        kernel = np.ones((5,5), np.uint8)
        edges1 = cv2.dilate(edges1, kernel, iterations=1)
        edges2 = cv2.dilate(edges2, kernel, iterations=1)
        
        # Find contours
        contours1, _ = cv2.findContours(edges1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours2, _ = cv2.findContours(edges2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by size
        min_area = 100
        max_area = 10000
        filtered_contours1 = [c for c in contours1 if min_area < cv2.contourArea(c) < max_area]
        filtered_contours2 = [c for c in contours2 if min_area < cv2.contourArea(c) < max_area]
        
        # Create keypoints from contours
        kp1 = []
        for contour in filtered_contours1:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                size = np.sqrt(area) * 0.5
                kp1.append(cv2.KeyPoint(cx, cy, size))
        
        kp2 = []
        for contour in filtered_contours2:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                area = cv2.contourArea(contour)
                size = np.sqrt(area) * 0.5
                kp2.append(cv2.KeyPoint(cx, cy, size))
        
        # If we have keypoints, compute SIFT descriptors for them
        if kp1 and kp2:
            kp1, des1 = self.detector.compute(self.img1, kp1)
            kp2, des2 = self.detector.compute(self.img2, kp2)
            
            self.kp1, self.kp2 = kp1, kp2
            
            # Match descriptors
            if des1 is not None and des2 is not None and len(des1) > 0 and len(des2) > 0:
                matches = self.matcher.knnMatch(des1, des2, k=2)
                
                # Apply very relaxed ratio test for pattern features
                good_matches = []
                for m, n in matches:
                    if m.distance < 0.95 * n.distance:  # Very relaxed ratio
                        good_matches.append(m)
                
                print(f"Found {len(good_matches)} pattern matches")
                self.matches = good_matches
                
                # Extract matched keypoints
                if len(good_matches) >= 4:
                    src_pts = np.array([[kp1[m.queryIdx].pt[0], kp1[m.queryIdx].pt[1]] 
                                       for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                    dst_pts = np.array([[kp2[m.trainIdx].pt[0], kp2[m.trainIdx].pt[1]] 
                                       for m in good_matches], dtype=np.float32).reshape(-1, 1, 2)
                else:
                    src_pts = np.zeros((0, 1, 2), dtype=np.float32)
                    dst_pts = np.zeros((0, 1, 2), dtype=np.float32)
                
                return src_pts, dst_pts, good_matches
        
        # If we couldn't find or match patterns
        return np.zeros((0, 1, 2), dtype=np.float32), np.zeros((0, 1, 2), dtype=np.float32), []
    
    def stitch(self, img1_path, img2_path, visualize=True, use_non_rigid=False, xray_type='general'):
        """Enhanced stitching with pattern recognition."""
        # Load images
        self.load_images(img1_path, img2_path)
        
        # Apply specialized preprocessing based on X-ray type
        self.img1 = self.preprocess_xray(self.img1, xray_type)
        self.img2 = self.preprocess_xray(self.img2, xray_type)
        
        # Try pattern matching first
        print("Attempting pattern-based matching...")
        src_pts, dst_pts, good_matches = self.detect_and_match_patterns()
        
        # If not enough matches, try anatomical region matching
        if len(good_matches) < 10 and xray_type in ['spine', 'head', 'neck']:
            print("Falling back to anatomical region matching...")
            src_pts, dst_pts, good_matches = self.detect_anatomical_regions(xray_type)
        
        # If still not enough matches, fall back to feature detection
        if len(good_matches) < 10:
            print("Falling back to general feature detection...")
            src_pts, dst_pts, good_matches = self.detect_and_match_features()
        
        # Apply spatial consistency check
        src_pts, dst_pts, good_matches = self.check_spatial_consistency(src_pts, dst_pts, good_matches)
        
        # Compute homography
        H, inlier_ratio = self.compute_homography(src_pts, dst_pts)
        
        # Choose warping method
        if use_non_rigid and len(good_matches) >= 4:
            print("Using non-rigid transformation...")
            result = self.warp_and_stitch_non_rigid()
        else:
            print("Using homography transformation...")
            result = self.warp_and_stitch()
        
        # Visualize if requested
        if visualize:
            self.visualize_matches()
            self.visualize_result()
            self.visualize_match_context(max_windows=5)  # Limit to 5 windows
        
        return result

# Example usage
if __name__ == "__main__":
    stitcher = XRayStitcher(feature_method='sift')
    result = stitcher.stitch("path/to/xray1.jpg", "path/to/xray2.jpg")
    cv2.imwrite("stitched_xray.jpg", result) 