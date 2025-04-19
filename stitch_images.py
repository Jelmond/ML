import os
from xray_stitcher import XRayStitcher
import argparse

def stitch_images_in_directory(use_non_rigid=False, xray_type='general'):
    # Create directories if they don't exist
    input_dir = "assets/imagesToStich"
    output_dir = "stitched"
    
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of images in input directory
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.tif'))]
    
    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {input_dir}. Found {len(image_files)} images.")
        return
    
    # Sort files to ensure consistent ordering
    image_files.sort()
    
    # Get full paths
    img1_path = os.path.join(input_dir, image_files[0])
    img2_path = os.path.join(input_dir, image_files[1])
    
    # Create output filename based on transformation type
    transform_type = "non_rigid" if use_non_rigid else "rigid"
    output_path = os.path.join(output_dir, f"stitched_{transform_type}_{xray_type}.jpg")
    
    print(f"Stitching images:")
    print(f"Image 1: {image_files[0]}")
    print(f"Image 2: {image_files[1]}")
    print(f"Transformation: {transform_type}")
    print(f"X-ray type: {xray_type}")
    
    # Initialize stitcher and process images
    try:
        stitcher = XRayStitcher(feature_method='sift')
        result = stitcher.stitch(
            img1_path, 
            img2_path, 
            visualize=True,
            use_non_rigid=use_non_rigid,
            xray_type=xray_type
        )
        
        # Save result
        stitcher.visualize_result(save_path=output_path)
        print(f"\nStitched image saved to: {output_path}")
        
    except Exception as e:
        print(f"Error during stitching: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Stitch X-ray images together')
    parser.add_argument('--non-rigid', action='store_true', help='Use non-rigid transformation')
    parser.add_argument('--type', choices=['general', 'spine', 'head', 'chest', 'neck'], 
                        default='general', help='Type of X-ray images')
    
    args = parser.parse_args()
    
    stitch_images_in_directory(use_non_rigid=args.non_rigid, xray_type=args.type) 