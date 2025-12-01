#!/usr/bin/env python3
"""Image preprocessing script for Assembly Swarm Environment.

This script processes target shape images (PNG files) and creates a pickle
file containing the grid coordinates for use in training.

The script:
1. Loads PNG images from a specified folder
2. Converts to binary using Otsu's thresholding
3. Extracts grid cell centers from the shape
4. Scales coordinates to physical units
5. Saves results to a pickle file

Usage:
    python preprocess_shapes.py --image_dir /path/to/images --output results.pkl
    
    # Or use defaults (looks for fig/ directory in project root)
    python preprocess_shapes.py

The output pickle contains:
    - 'l_cell': List of grid cell sizes for each shape
    - 'grid_coords': List of (n_grid, 2) arrays with grid centers
    - 'binary_image': List of binary image arrays (for visualization)
    - 'shape_bound_points': List of bounding boxes [x_min, x_max, y_min, y_max]
"""

from typing import Dict, List, Any, Optional
from pathlib import Path
import numpy as np
import pickle
import os
import glob
import argparse


def process_image(
    image_path: str,
    grid_size: int = 36,
    target_height: float = 2.2,
    visualize: bool = False,
) -> Dict[str, Any]:
    """Process a single image to extract grid coordinates.
    
    Args:
        image_path: Path to the input PNG image
        grid_size: Pixel size for grid discretization
        target_height: Target height in physical units
        visualize: Whether to show visualization plot
        
    Returns:
        Dictionary with:
            - 'l_cell': Scaled grid cell size
            - 'grid_coords': Grid center coordinates (n_grid, 2)
            - 'binary_image': Processed binary image
            - 'shape_bound_points': Bounding box [x_min, x_max, y_min, y_max]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV (cv2) is required. Install with: pip install opencv-python")
    
    # Load and binarize the input image
    gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if gray_image is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    _, binary_image = cv2.threshold(
        gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Crop image to shape boundaries
    black_pixels = np.argwhere(binary_image == 0)
    if len(black_pixels) == 0:
        raise ValueError(f"No black pixels found in image: {image_path}")
    
    min_y, min_x = black_pixels.min(axis=0)
    max_y, max_x = black_pixels.max(axis=0)
    binary_image = binary_image[min_y:max_y + 1, min_x:max_x + 1]

    # Flip vertically (origin at bottom-left for consistency with environment)
    height, width = binary_image.shape
    binary_image = np.flipud(binary_image)

    # Extract grid centers from black regions
    black_grid_coords = []
    for i in range(grid_size, height - grid_size, grid_size):
        for j in range(grid_size, width - grid_size, grid_size):
            # Extract current grid section
            grid_section = binary_image[i:i + grid_size, j:j + grid_size]
            
            # Calculate grid center coordinates
            center_x = j + grid_size / 2
            center_y = i + grid_size / 2

            # Check if grid is entirely within black region
            black_pixel_count = np.sum(grid_section == 0)
            total_pixel_count = grid_size * grid_size
            black_pixel_ratio = black_pixel_count / total_pixel_count

            # Save grid center if fully within target shape
            if black_pixel_ratio >= 1.0:
                black_grid_coords.append([center_x, center_y])

    if len(black_grid_coords) == 0:
        raise ValueError(f"No valid grid cells found in image: {image_path}")
    
    # Convert to numpy array
    black_grid_coords = np.array(black_grid_coords, dtype=np.float64)
    print(f"  {os.path.basename(image_path)}: {len(black_grid_coords)} grid cells")

    # Center coordinates at origin
    x_mean = np.mean(black_grid_coords[:, 0])
    y_mean = np.mean(black_grid_coords[:, 1])
    black_grid_coords[:, 0] -= x_mean
    black_grid_coords[:, 1] -= y_mean

    # Calculate shape boundaries
    x_min = np.min(black_grid_coords[:, 0])
    x_max = np.max(black_grid_coords[:, 0])
    y_min = np.min(black_grid_coords[:, 1])
    y_max = np.max(black_grid_coords[:, 1])

    # Scale coordinates to target physical size
    real_height = y_max - y_min
    if real_height == 0:
        real_height = x_max - x_min  # Use width if height is zero
    h_scale = target_height / real_height
    grid_coords = h_scale * black_grid_coords
    l_cell = grid_size * h_scale

    # Calculate scaled boundary points
    shape_bound_points = np.array([
        (x_min - x_mean) * h_scale,
        (x_max - x_mean) * h_scale,
        (y_min - y_mean) * h_scale,
        (y_max - y_mean) * h_scale,
    ])

    # Optional visualization
    if visualize:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Original binary image
            axes[0].imshow(binary_image, cmap='gray', origin='lower')
            axes[0].set_title('Binary Image')
            
            # Grid centers
            axes[1].scatter(grid_coords[:, 0], grid_coords[:, 1], 
                          c='green', s=20, alpha=0.8)
            axes[1].set_aspect('equal')
            axes[1].set_title(f'Grid Centers ({len(grid_coords)} cells)')
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("  matplotlib not available for visualization")

    return {
        'l_cell': l_cell,
        'grid_coords': grid_coords,
        'binary_image': binary_image,
        'shape_bound_points': shape_bound_points,
    }


def process_image_folder(
    image_dir: str,
    output_file: str,
    grid_size: int = 36,
    target_height: float = 2.2,
    visualize: bool = False,
) -> str:
    """Process all PNG images in a folder and save to pickle.
    
    Args:
        image_dir: Directory containing PNG images
        output_file: Path for output pickle file
        grid_size: Pixel size for grid discretization
        target_height: Target height in physical units
        visualize: Whether to show visualization for each image
        
    Returns:
        Path to the created pickle file
    """
    # Find all PNG images
    image_paths = glob.glob(os.path.join(image_dir, '*.png'))
    
    # Sort by numeric filename if possible (e.g., 1.png, 2.png, ...)
    def sort_key(path):
        name = os.path.basename(path).split('.')[0]
        try:
            return int(name)
        except ValueError:
            return name
    
    image_paths = sorted(image_paths, key=sort_key)
    
    if len(image_paths) == 0:
        raise FileNotFoundError(f"No PNG images found in: {image_dir}")
    
    print(f"Processing {len(image_paths)} images from: {image_dir}")
    
    # Process each image
    results = {
        "l_cell": [],
        "grid_coords": [],
        "binary_image": [],
        "shape_bound_points": [],
    }
    
    for image_path in image_paths:
        try:
            data = process_image(
                image_path, 
                grid_size=grid_size,
                target_height=target_height,
                visualize=visualize,
            )
            results["l_cell"].append(data['l_cell'])
            results["grid_coords"].append(data['grid_coords'])
            results["binary_image"].append(data['binary_image'])
            results["shape_bound_points"].append(data['shape_bound_points'])
        except Exception as e:
            print(f"  Warning: Failed to process {image_path}: {e}")
    
    if len(results["grid_coords"]) == 0:
        raise ValueError("No images were successfully processed")
    
    # Save to pickle
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"\nSaved {len(results['grid_coords'])} shapes to: {output_path}")
    print(f"  Grid cells per shape: {[len(g) for g in results['grid_coords']]}")
    print(f"  Cell sizes: {[f'{l:.3f}' for l in results['l_cell']]}")
    
    return str(output_path)


def get_default_paths():
    """Get default image directory and output file paths.
    
    Default location: jaxMARL/fig/
    """
    # Navigate from this file: cfg/ -> jax_marl/ -> jaxMARL/
    current_dir = Path(__file__).resolve().parent
    jaxmarl_root = current_dir.parent.parent  # jaxMARL/
    
    # Default location: jaxMARL/fig/
    image_dir = jaxmarl_root / "fig"
    output_file = image_dir / "results.pkl"
    
    return str(image_dir), str(output_file)


def main():
    """Main entry point for command-line usage."""
    default_image_dir, default_output = get_default_paths()
    
    parser = argparse.ArgumentParser(
        description="Preprocess shape images for Assembly Swarm Environment"
    )
    parser.add_argument(
        "--image_dir", 
        type=str, 
        default=default_image_dir,
        help=f"Directory containing PNG images (default: {default_image_dir})"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default=default_output,
        help=f"Output pickle file path (default: {default_output})"
    )
    parser.add_argument(
        "--grid_size", 
        type=int, 
        default=36,
        help="Grid size in pixels for discretization (default: 36)"
    )
    parser.add_argument(
        "--target_height", 
        type=float, 
        default=2.2,
        help="Target height in physical units (default: 2.2)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Show visualization for each processed image"
    )
    
    args = parser.parse_args()
    
    try:
        process_image_folder(
            image_dir=args.image_dir,
            output_file=args.output,
            grid_size=args.grid_size,
            target_height=args.target_height,
            visualize=args.visualize,
        )
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"\nMake sure PNG images exist in: {args.image_dir}")
        print("Or specify a different directory with --image_dir")
        exit(1)


if __name__ == "__main__":
    main()
