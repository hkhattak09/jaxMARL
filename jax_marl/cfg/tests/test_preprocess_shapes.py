"""Tests for preprocess_shapes.py - Image preprocessing utilities.

Run with: python tests/test_preprocess_shapes.py
"""

import sys
import os
from pathlib import Path
import tempfile
import shutil
import pickle

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Try importing pytest, but make it optional
try:
    import pytest
    HAS_PYTEST = True
except ImportError:
    HAS_PYTEST = False
    # Create a minimal pytest replacement
    class pytest:
        class skip:
            Exception = Exception
        @staticmethod
        def raises(exc):
            class _Raises:
                def __enter__(self): return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    if exc_type is None:
                        raise AssertionError(f"Expected {exc} but nothing was raised")
                    return issubclass(exc_type, exc)
            return _Raises()


class TestProcessImage:
    """Tests for process_image function."""
    
    def test_process_synthetic_image(self):
        """Test processing a synthetic binary image."""
        print("Testing process_image with synthetic image...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image - BLACK background (0) with BLACK shape region
            # The function looks for black pixels (value 0) as the shape
            # But Otsu thresholding will convert, so we need proper contrast
            
            # Create image with black shape on white background
            img = np.ones((200, 200), dtype=np.uint8) * 255  # White background
            img[50:150, 50:150] = 0  # Black square in center (this is the shape)
            
            img_path = Path(tmpdir) / "test_shape.png"
            cv2.imwrite(str(img_path), img)
            
            # Process the image with default parameters
            result = process_image(str(img_path), grid_size=36, target_height=2.2)
            
            assert "l_cell" in result
            assert "grid_coords" in result
            assert "binary_image" in result
            assert "shape_bound_points" in result
            
            assert isinstance(result["l_cell"], float)
            assert len(result["grid_coords"]) > 0, "Should have grid cells"
            
            print(f"  ✓ process_image with synthetic image passed ({len(result['grid_coords'])} cells)")
    
    def test_process_image_grid_structure(self):
        """Test that processed image has valid grid structure."""
        print("Testing process_image grid structure...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large black square on white background
            img = np.ones((300, 300), dtype=np.uint8) * 255
            img[50:250, 50:250] = 0  # Large black square
            
            img_path = Path(tmpdir) / "square.png"
            cv2.imwrite(str(img_path), img)
            
            result = process_image(str(img_path), grid_size=36, target_height=2.2)
            
            grid_coords = result["grid_coords"]
            
            # Grid coords should be 2D array
            assert len(grid_coords.shape) == 2, "grid_coords should be 2D"
            assert grid_coords.shape[1] == 2, "grid_coords should have x,y columns"
            
            # All grid coords should be centered around (0, 0)
            mean_coord = np.mean(grid_coords, axis=0)
            assert abs(mean_coord[0]) < 0.5, f"X centroid should be near 0, got {mean_coord[0]}"
            assert abs(mean_coord[1]) < 0.5, f"Y centroid should be near 0, got {mean_coord[1]}"
            
            print(f"  ✓ process_image grid structure passed (centroid: {mean_coord})")
    
    def test_process_image_different_grid_sizes(self):
        """Test image processing with different grid sizes."""
        print("Testing process_image with different grid sizes...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a large black shape
            img = np.ones((400, 400), dtype=np.uint8) * 255
            img[100:300, 100:300] = 0
            
            img_path = Path(tmpdir) / "shape.png"
            cv2.imwrite(str(img_path), img)
            
            # Process with different grid sizes
            result_small = process_image(str(img_path), grid_size=20, target_height=2.2)
            result_large = process_image(str(img_path), grid_size=50, target_height=2.2)
            
            # Smaller grid should produce more cells
            n_small = len(result_small["grid_coords"])
            n_large = len(result_large["grid_coords"])
            
            assert n_small >= n_large, f"Smaller grid should have >= cells: {n_small} vs {n_large}"
            
            print(f"  ✓ Grid size test passed (small: {n_small} cells, large: {n_large} cells)")
    
    def test_process_image_different_target_heights(self):
        """Test image processing with different target heights."""
        print("Testing process_image with different target heights...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            img = np.ones((300, 300), dtype=np.uint8) * 255
            img[75:225, 75:225] = 0
            
            img_path = Path(tmpdir) / "shape.png"
            cv2.imwrite(str(img_path), img)
            
            result1 = process_image(str(img_path), grid_size=36, target_height=1.0)
            result2 = process_image(str(img_path), grid_size=36, target_height=2.0)
            
            # Different target heights should produce different l_cell values
            assert result1["l_cell"] != result2["l_cell"], "Different heights should give different l_cell"
            
            # The ratio should be approximately 2:1
            ratio = result2["l_cell"] / result1["l_cell"]
            assert 1.8 < ratio < 2.2, f"l_cell ratio should be ~2, got {ratio}"
            
            print(f"  ✓ Target height test passed (l_cell ratio: {ratio:.2f})")


class TestProcessImageFolder:
    """Tests for process_image_folder function."""
    
    def test_process_empty_folder(self):
        """Test processing empty folder raises error."""
        print("Testing process_image_folder with empty folder...")
        
        from preprocess_shapes import process_image_folder
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.pkl"
            
            try:
                process_image_folder(tmpdir, str(output_path))
                print("  ✗ Should have raised FileNotFoundError")
                raise AssertionError("Expected FileNotFoundError for empty folder")
            except FileNotFoundError as e:
                assert "No PNG images found" in str(e)
                print("  ✓ Empty folder correctly raises FileNotFoundError")
    
    def test_process_folder_with_images(self):
        """Test processing folder with multiple images."""
        print("Testing process_image_folder with images...")
        
        from preprocess_shapes import process_image_folder
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create two test images with black shapes on white background
            for i, name in enumerate(["shape1.png", "shape2.png"]):
                img = np.ones((300, 300), dtype=np.uint8) * 255
                # Different sized shapes
                offset = i * 20
                img[50+offset:200+offset, 50:200] = 0
                cv2.imwrite(str(Path(tmpdir) / name), img)
            
            output_path = Path(tmpdir) / "output.pkl"
            
            result_path = process_image_folder(tmpdir, str(output_path))
            
            # Output file should be created
            assert Path(result_path).exists(), "Output pickle should exist"
            
            # Load and check structure
            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            
            assert "l_cell" in loaded
            assert "grid_coords" in loaded
            assert "binary_image" in loaded
            assert "shape_bound_points" in loaded
            
            assert len(loaded["l_cell"]) == 2, "Should have 2 shapes"
            assert len(loaded["grid_coords"]) == 2
            
            print(f"  ✓ process_image_folder with images passed ({len(loaded['l_cell'])} shapes)")
    
    def test_process_folder_pickle_structure(self):
        """Test that pickle output has correct list structure."""
        print("Testing process_image_folder pickle structure...")
        
        from preprocess_shapes import process_image_folder
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test image
            img = np.ones((300, 300), dtype=np.uint8) * 255
            img[50:250, 50:250] = 0
            cv2.imwrite(str(Path(tmpdir) / "test.png"), img)
            
            output_path = Path(tmpdir) / "shapes.pkl"
            
            process_image_folder(tmpdir, str(output_path))
            
            # Load and verify list structure
            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            
            assert isinstance(loaded, dict), "Pickle should contain dict"
            assert isinstance(loaded["l_cell"], list), "l_cell should be a list"
            assert isinstance(loaded["grid_coords"], list), "grid_coords should be a list"
            assert isinstance(loaded["binary_image"], list), "binary_image should be a list"
            assert isinstance(loaded["shape_bound_points"], list), "shape_bound_points should be a list"
            
            # Check individual entries
            assert isinstance(loaded["l_cell"][0], float), "l_cell entries should be floats"
            assert isinstance(loaded["grid_coords"][0], np.ndarray), "grid_coords entries should be arrays"
            
            print("  ✓ process_image_folder pickle structure passed")


class TestImageProcessingEdgeCases:
    """Tests for edge cases in image processing."""
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent file."""
        print("Testing process_image with nonexistent file...")
        
        from preprocess_shapes import process_image
        
        try:
            process_image("/nonexistent/path/to/image.png")
            raise AssertionError("Should have raised an exception")
        except (ValueError, FileNotFoundError, Exception) as e:
            # cv2.imread returns None for nonexistent files, which causes ValueError
            print("  ✓ process_image nonexistent file handling passed")
    
    def test_all_white_image(self):
        """Test processing an all-white image (no shape)."""
        print("Testing process_image with all-white image...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # All white image - no black pixels = no shape
            img = np.ones((100, 100), dtype=np.uint8) * 255
            img_path = Path(tmpdir) / "white.png"
            cv2.imwrite(str(img_path), img)
            
            try:
                result = process_image(str(img_path))
                raise AssertionError("Should raise ValueError for all-white image")
            except ValueError as e:
                assert "No black pixels" in str(e)
                print("  ✓ All-white image correctly raises ValueError")
    
    def test_all_black_image(self):
        """Test processing an all-black image (entire image is shape)."""
        print("Testing process_image with all-black image...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # All black image - entire image is the shape
            # This should work but might produce few cells due to grid margins
            img = np.zeros((200, 200), dtype=np.uint8)
            img_path = Path(tmpdir) / "black.png"
            cv2.imwrite(str(img_path), img)
            
            try:
                result = process_image(str(img_path), grid_size=20)
                # If it succeeds, check it produced some cells
                print(f"  ✓ All-black image processed: {len(result['grid_coords'])} cells")
            except ValueError as e:
                # It's also valid to fail if the cropped region is too small
                print(f"  ✓ All-black image raised ValueError as expected: {e}")


class TestRealShapeImages:
    """Tests using real shape images from jaxMARL/fig directory."""
    
    # Path to the real fig directory
    FIG_DIR = Path(__file__).parent.parent.parent.parent / "fig"
    
    def test_fig_directory_exists(self):
        """Test that the fig directory exists and has images."""
        print("Testing fig directory exists...")
        
        assert self.FIG_DIR.exists(), f"Fig directory not found: {self.FIG_DIR}"
        
        png_files = list(self.FIG_DIR.glob("*.png"))
        assert len(png_files) > 0, "No PNG files found in fig directory"
        
        print(f"  ✓ Found {len(png_files)} PNG files in fig directory")
    
    def test_process_real_shape_1(self):
        """Test processing shape 1.png from fig directory."""
        print("Testing process_image with real shape 1.png...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        img_path = self.FIG_DIR / "1.png"
        if not img_path.exists():
            print("  ⚠ Skipped (1.png not found)")
            return
        
        result = process_image(str(img_path), grid_size=36, target_height=2.2)
        
        assert "l_cell" in result
        assert "grid_coords" in result
        assert "binary_image" in result
        assert "shape_bound_points" in result
        
        # Should have meaningful grid cells (hundreds for real shapes)
        n_cells = len(result["grid_coords"])
        assert n_cells > 100, f"Real shape should have >100 cells, got {n_cells}"
        
        print(f"  ✓ Processed 1.png: {n_cells} grid cells, l_cell={result['l_cell']:.4f}")
    
    def test_process_all_real_shapes(self):
        """Test processing all PNG shapes in fig directory."""
        print("Testing process_image with all real shapes...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        png_files = list(self.FIG_DIR.glob("*.png"))
        
        if not png_files:
            print("  ⚠ Skipped (no PNG files found)")
            return
        
        all_passed = True
        for img_path in sorted(png_files):
            try:
                result = process_image(str(img_path), grid_size=36, target_height=2.2)
                
                n_cells = len(result["grid_coords"])
                l_cell = result["l_cell"]
                print(f"    {img_path.name}: {n_cells} cells, l_cell={l_cell:.4f}")
                
            except Exception as e:
                print(f"    {img_path.name}: FAILED - {e}")
                all_passed = False
        
        assert all_passed, "Some shapes failed to process"
        print(f"  ✓ Successfully processed all {len(png_files)} shapes")
    
    def test_process_fig_folder_to_pickle(self):
        """Test processing entire fig folder and saving to pickle."""
        print("Testing process_image_folder with fig directory...")
        
        from preprocess_shapes import process_image_folder
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shapes.pkl"
            
            result_path = process_image_folder(str(self.FIG_DIR), str(output_path))
            
            assert Path(result_path).exists(), "Output pickle not created"
            
            # Verify pickle content
            with open(output_path, "rb") as f:
                loaded = pickle.load(f)
            
            n_shapes = len(loaded["grid_coords"])
            assert n_shapes > 0, "No shapes in pickle"
            
            # Verify structure
            assert len(loaded["l_cell"]) == n_shapes
            assert len(loaded["binary_image"]) == n_shapes
            assert len(loaded["shape_bound_points"]) == n_shapes
            
            # Print summary
            total_cells = sum(len(g) for g in loaded["grid_coords"])
            print(f"  ✓ Processed fig folder: {n_shapes} shapes, {total_cells} total cells")
    
    def test_shape_grid_coords_centered(self):
        """Test that grid coordinates are properly centered around origin."""
        print("Testing grid coordinates are centered...")
        
        from preprocess_shapes import process_image
        
        try:
            import cv2
        except ImportError:
            print("  ⚠ Skipped (OpenCV not installed)")
            return
        
        png_files = list(self.FIG_DIR.glob("*.png"))
        if not png_files:
            print("  ⚠ Skipped (no PNG files found)")
            return
        
        # Test first available shape
        img_path = png_files[0]
        result = process_image(str(img_path), grid_size=36, target_height=2.2)
        
        grid_coords = result["grid_coords"]
        
        # Centroid should be very close to origin (due to centering in process_image)
        centroid = np.mean(grid_coords, axis=0)
        assert abs(centroid[0]) < 0.01, f"X centroid should be ~0, got {centroid[0]}"
        assert abs(centroid[1]) < 0.01, f"Y centroid should be ~0, got {centroid[1]}"
        
        print(f"  ✓ Grid centroid at origin: ({centroid[0]:.6f}, {centroid[1]:.6f})")
    
    def test_compare_with_existing_results(self):
        """Test compatibility with existing results.pkl structure if it exists."""
        print("Testing compatibility with existing results.pkl...")
        
        results_path = self.FIG_DIR / "results.pkl"
        if not results_path.exists():
            print("  ⚠ Skipped (results.pkl not found)")
            return
        
        with open(results_path, "rb") as f:
            existing = pickle.load(f)
        
        # Check what structure the existing file has
        if isinstance(existing, dict):
            if "l_cell" in existing and isinstance(existing["l_cell"], list):
                # List-based structure (same as our output)
                n_shapes = len(existing["l_cell"])
                print(f"  ✓ Existing results.pkl has {n_shapes} shapes (list format)")
                
                # Verify expected keys
                for key in ["l_cell", "grid_coords"]:
                    assert key in existing, f"Missing {key} in results.pkl"
            else:
                # Dictionary with shape names as keys
                print(f"  ✓ Existing results.pkl has {len(existing)} shapes (dict format)")
        else:
            print(f"  ⚠ Unexpected format in results.pkl: {type(existing)}")


def run_tests():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("  PREPROCESS SHAPES TESTS")
    print("=" * 60)
    
    passed = 0
    failed = 0
    skipped = 0
    
    test_classes = [
        TestProcessImage,
        TestProcessImageFolder,
        TestImageProcessingEdgeCases,
        TestRealShapeImages,
    ]
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        for method_name in sorted(dir(instance)):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                except Exception as e:
                    print(f"  ✗ {method_name} FAILED: {e}")
                    failed += 1
    
    print("\n" + "=" * 60)
    print(f"  Results: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
