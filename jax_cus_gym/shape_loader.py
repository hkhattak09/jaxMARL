"""Shape loading utilities for Assembly Swarm Environment.

This module provides functionality to load target shapes from pickle files
and convert them to JAX arrays for use in the environment.

The pickle files should contain:
- 'l_cell': Grid cell sizes for each shape
- 'grid_coords': Grid center coordinates for each shape
- 'binary_image': Binary image representations for rendering
- 'shape_bound_points': Bounding boxes for visualization
"""

from typing import Tuple, List, Dict, Any, Optional, Union
import pickle
import os
import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class ShapeData:
    """Container for a single target shape.
    
    Attributes:
        grid_centers: Grid cell centers, shape (n_grid, 2)
        l_cell: Cell size
        n_grid: Number of grid cells
        binary_image: Binary image for rendering (optional, stored as array)
        bound_points: Bounding box [x_min, x_max, y_min, y_max]
    """
    grid_centers: jnp.ndarray  # (n_grid, 2)
    l_cell: float
    n_grid: int
    binary_image: Optional[jnp.ndarray] = None  # (H, W)
    bound_points: Optional[jnp.ndarray] = None  # (4,)


@struct.dataclass
class ShapeLibrary:
    """Container for multiple target shapes.
    
    All shapes are padded to the same size for JAX compatibility.
    
    Attributes:
        grid_centers: All grid centers, shape (n_shapes, max_n_grid, 2)
        l_cells: Cell sizes for each shape, shape (n_shapes,)
        n_grids: Number of grid cells per shape, shape (n_shapes,)
        n_shapes: Total number of shapes
        max_n_grid: Maximum grid cells across all shapes
        shape_masks: Valid grid cell masks, shape (n_shapes, max_n_grid)
    """
    grid_centers: jnp.ndarray  # (n_shapes, max_n_grid, 2)
    l_cells: jnp.ndarray  # (n_shapes,)
    n_grids: jnp.ndarray  # (n_shapes,)
    n_shapes: int
    max_n_grid: int
    shape_masks: jnp.ndarray  # (n_shapes, max_n_grid) - True for valid cells


def load_shapes_from_pickle(filepath: str) -> ShapeLibrary:
    """Load shapes from a pickle file.
    
    Expected pickle structure:
    {
        'l_cell': list of cell sizes,
        'grid_coords': list of (2, n_grid) arrays,
        'binary_image': list of 2D arrays (optional),
        'shape_bound_points': list of [x_min, x_max, y_min, y_max] (optional)
    }
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        ShapeLibrary containing all shapes
        
    Raises:
        FileNotFoundError: If file doesn't exist
        KeyError: If required keys are missing
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Shape file not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    # Extract required fields
    l_cells = data['l_cell']
    grid_coords_list = data['grid_coords']
    
    # Handle different formats - grid_coords might be (2, n) or (n, 2)
    processed_grids = []
    for grid in grid_coords_list:
        grid = jnp.array(grid)
        if grid.shape[0] == 2 and grid.ndim == 2:
            # Shape is (2, n_grid), transpose to (n_grid, 2)
            grid = grid.T
        processed_grids.append(grid)
    
    n_shapes = len(l_cells)
    n_grids = [g.shape[0] for g in processed_grids]
    max_n_grid = max(n_grids)
    
    # Pad all grids to max size
    padded_grids = []
    masks = []
    for i, grid in enumerate(processed_grids):
        n = grid.shape[0]
        if n < max_n_grid:
            # Pad with zeros
            padding = jnp.zeros((max_n_grid - n, 2))
            padded = jnp.concatenate([grid, padding], axis=0)
        else:
            padded = grid
        padded_grids.append(padded)
        
        # Create mask
        mask = jnp.arange(max_n_grid) < n
        masks.append(mask)
    
    # Stack into arrays
    grid_centers = jnp.stack(padded_grids, axis=0)  # (n_shapes, max_n_grid, 2)
    l_cells_arr = jnp.array(l_cells)
    n_grids_arr = jnp.array(n_grids)
    shape_masks = jnp.stack(masks, axis=0)
    
    return ShapeLibrary(
        grid_centers=grid_centers,
        l_cells=l_cells_arr,
        n_grids=n_grids_arr,
        n_shapes=n_shapes,
        max_n_grid=max_n_grid,
        shape_masks=shape_masks,
    )


def get_shape_from_library(
    library: ShapeLibrary,
    shape_idx: int,
) -> Tuple[jnp.ndarray, float, jnp.ndarray]:
    """Extract a single shape from the library.
    
    Args:
        library: ShapeLibrary containing all shapes
        shape_idx: Index of shape to extract
        
    Returns:
        grid_centers: Grid centers for this shape (max_n_grid, 2)
        l_cell: Cell size
        mask: Valid cell mask (max_n_grid,)
    """
    grid_centers = library.grid_centers[shape_idx]
    l_cell = library.l_cells[shape_idx]
    mask = library.shape_masks[shape_idx]
    
    return grid_centers, l_cell, mask


def apply_shape_transform(
    grid_centers: jnp.ndarray,
    mask: jnp.ndarray,
    rotation_angle: float = 0.0,
    scale: float = 1.0,
    offset: jnp.ndarray = None,
) -> jnp.ndarray:
    """Apply transformation to grid centers (domain randomization).
    
    Args:
        grid_centers: Grid centers, shape (n_grid, 2) or (max_n_grid, 2)
        mask: Valid cell mask, shape (n_grid,) or (max_n_grid,)
        rotation_angle: Rotation angle in radians
        scale: Scale factor
        offset: Translation offset, shape (2,)
        
    Returns:
        Transformed grid centers (same shape as input)
    """
    if offset is None:
        offset = jnp.zeros(2)
    
    # Build rotation matrix
    cos_a = jnp.cos(rotation_angle)
    sin_a = jnp.sin(rotation_angle)
    rotation_matrix = jnp.array([
        [cos_a, -sin_a],
        [sin_a, cos_a]
    ])
    
    # Apply transformations: scale -> rotate -> translate
    transformed = grid_centers * scale
    transformed = jnp.dot(transformed, rotation_matrix.T)
    transformed = transformed + offset
    
    # Zero out invalid cells (masked)
    transformed = jnp.where(mask[:, None], transformed, 0.0)
    
    return transformed


def create_procedural_shape(
    shape_type: str = "rectangle",
    n_cells_x: int = 4,
    n_cells_y: int = 4,
    l_cell: float = 0.3,
    center: jnp.ndarray = None,
) -> Tuple[jnp.ndarray, float]:
    """Create a procedural target shape.
    
    Args:
        shape_type: Type of shape ("rectangle", "cross", "ring", "line")
        n_cells_x: Number of cells in x direction
        n_cells_y: Number of cells in y direction
        l_cell: Cell size
        center: Center position, shape (2,)
        
    Returns:
        grid_centers: Grid cell centers, shape (n_grid, 2)
        l_cell: Cell size
    """
    if center is None:
        center = jnp.zeros(2)
    
    if shape_type == "rectangle":
        # Standard rectangular grid
        x = jnp.arange(n_cells_x) * l_cell - (n_cells_x - 1) * l_cell / 2
        y = jnp.arange(n_cells_y) * l_cell - (n_cells_y - 1) * l_cell / 2
        xx, yy = jnp.meshgrid(x, y)
        grid_centers = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)
        
    elif shape_type == "cross":
        # Cross/plus shape
        x = jnp.arange(n_cells_x) * l_cell - (n_cells_x - 1) * l_cell / 2
        y = jnp.arange(n_cells_y) * l_cell - (n_cells_y - 1) * l_cell / 2
        
        # Horizontal bar
        h_cells = jnp.stack([x, jnp.zeros_like(x)], axis=-1)
        # Vertical bar
        v_cells = jnp.stack([jnp.zeros_like(y), y], axis=-1)
        
        # Combine (remove center duplicate)
        grid_centers = jnp.concatenate([h_cells, v_cells[1:]], axis=0)
        
    elif shape_type == "ring":
        # Circular ring
        n_cells = n_cells_x * 4  # Approximate
        angles = jnp.linspace(0, 2 * jnp.pi, n_cells, endpoint=False)
        radius = (n_cells_x / 2) * l_cell
        x = radius * jnp.cos(angles)
        y = radius * jnp.sin(angles)
        grid_centers = jnp.stack([x, y], axis=-1)
        
    elif shape_type == "line":
        # Simple line
        x = jnp.arange(n_cells_x) * l_cell - (n_cells_x - 1) * l_cell / 2
        y = jnp.zeros(n_cells_x)
        grid_centers = jnp.stack([x, y], axis=-1)
        
    else:
        # Default to rectangle
        return create_procedural_shape("rectangle", n_cells_x, n_cells_y, l_cell, center)
    
    # Apply center offset
    grid_centers = grid_centers + center
    
    return grid_centers, l_cell


def create_shape_library_from_procedural(
    shape_types: List[str] = None,
    n_cells: int = 4,
    l_cell: float = 0.3,
) -> ShapeLibrary:
    """Create a shape library from procedural shapes.
    
    Useful for testing when no pickle file is available.
    
    Args:
        shape_types: List of shape types to include
        n_cells: Number of cells per dimension
        l_cell: Cell size
        
    Returns:
        ShapeLibrary containing all shapes
    """
    if shape_types is None:
        shape_types = ["rectangle", "cross", "ring", "line"]
    
    shapes = []
    l_cells = []
    n_grids = []
    
    for shape_type in shape_types:
        grid, lc = create_procedural_shape(shape_type, n_cells, n_cells, l_cell)
        shapes.append(grid)
        l_cells.append(lc)
        n_grids.append(grid.shape[0])
    
    max_n_grid = max(n_grids)
    
    # Pad to same size
    padded_grids = []
    masks = []
    for i, grid in enumerate(shapes):
        n = grid.shape[0]
        if n < max_n_grid:
            padding = jnp.zeros((max_n_grid - n, 2))
            padded = jnp.concatenate([grid, padding], axis=0)
        else:
            padded = grid
        padded_grids.append(padded)
        masks.append(jnp.arange(max_n_grid) < n)
    
    return ShapeLibrary(
        grid_centers=jnp.stack(padded_grids, axis=0),
        l_cells=jnp.array(l_cells),
        n_grids=jnp.array(n_grids),
        n_shapes=len(shape_types),
        max_n_grid=max_n_grid,
        shape_masks=jnp.stack(masks, axis=0),
    )


def save_shapes_to_pickle(
    library: ShapeLibrary,
    filepath: str,
    include_images: bool = False,
) -> None:
    """Save shape library to pickle file.
    
    Args:
        library: ShapeLibrary to save
        filepath: Output file path
        include_images: Whether to include binary images
    """
    # Convert to format expected by original code
    data = {
        'l_cell': [float(lc) for lc in library.l_cells],
        'grid_coords': [],
        'binary_image': [],
        'shape_bound_points': [],
    }
    
    for i in range(library.n_shapes):
        n_grid = int(library.n_grids[i])
        grid = library.grid_centers[i, :n_grid]
        
        # Transpose to (2, n_grid) format
        data['grid_coords'].append(grid.T)
        
        # Compute bounds
        x_min, y_min = jnp.min(grid, axis=0)
        x_max, y_max = jnp.max(grid, axis=0)
        data['shape_bound_points'].append([
            float(x_min), float(x_max), float(y_min), float(y_max)
        ])
        
        # Placeholder for binary image
        if include_images:
            # Create simple binary image
            resolution = 50
            x = jnp.linspace(float(x_min), float(x_max), resolution)
            y = jnp.linspace(float(y_min), float(y_max), resolution)
            xx, yy = jnp.meshgrid(x, y)
            points = jnp.stack([xx.flatten(), yy.flatten()], axis=-1)
            
            # Check which points are near grid cells
            dists = jnp.linalg.norm(
                points[:, None, :] - grid[None, :, :], axis=-1
            )
            min_dists = jnp.min(dists, axis=1)
            in_shape = min_dists < library.l_cells[i]
            image = in_shape.reshape(resolution, resolution)
            data['binary_image'].append(image)
        else:
            data['binary_image'].append(None)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    # Test shape loading utilities
    print("Testing shape loader...")
    
    # Create procedural library
    library = create_shape_library_from_procedural()
    print(f"Created library with {library.n_shapes} shapes")
    print(f"Max grid size: {library.max_n_grid}")
    print(f"Grid shapes: {library.n_grids}")
    
    # Test shape extraction
    grid, l_cell, mask = get_shape_from_library(library, 0)
    print(f"Shape 0: {jnp.sum(mask)} valid cells, l_cell={l_cell}")
    
    # Test transformation
    transformed = apply_shape_transform(
        grid, mask,
        rotation_angle=jnp.pi / 4,
        scale=1.5,
        offset=jnp.array([1.0, 0.5])
    )
    print(f"Transformed shape center: {jnp.mean(transformed[mask], axis=0)}")
    
    # Save and reload
    test_path = "/tmp/test_shapes.pkl"
    save_shapes_to_pickle(library, test_path)
    print(f"Saved to {test_path}")
    
    reloaded = load_shapes_from_pickle(test_path)
    print(f"Reloaded library with {reloaded.n_shapes} shapes")
    
    print("Shape loader test passed!")
