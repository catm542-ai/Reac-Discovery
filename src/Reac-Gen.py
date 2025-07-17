"""
Reac-Gen
Reac-Gen generates 3D reactor geometries using 20 mathematical surface equations, automates scaling and slicing, 
calculates key metrics (e.g., surface area, tortuosity), and exports STL files for 3D printing with corresponding Excel data.
@author: Cristopher Tinajero
"""

import numpy as np
import trimesh
import pandas as pd
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.path import Path
from skimage import measure


# ===========================================
# Function to generate continuous mesh from a scalar field
# ===========================================
def generate_continuous(equation_func, size, resolution, level, desired_height):
    """
    Generates a continuous 3D mesh from a scalar field defined by 'equation_func'.

    Parameters:
      - equation_func: function of three variables (X, Y, Z) returning scalar field values.
      - size: defines the X and Y domain limits: [-size, size].
      - resolution: number of sampling points in X and Y.
      - level: threshold level for the isosurface.
      - desired_height: total desired height in the Z direction (mesh extends from -size to desired_height - size).

    Returns:
      - mesh: generated mesh (trimesh.Trimesh) using marching cubes.
    """
    # X and Y domain: [-size, size]
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)

    # For Z: we want the total height to be:
    #   total_height = (z_max - (-size)) = (z_max + size) = desired_height  =>  z_max = desired_height - size
    z_max = desired_height - size

    # Extend the Z domain from -size to z_max
    # Adjust Z resolution proportionally to maintain aspect ratio with X and Y
    resolution_z = int(np.ceil(resolution * ((z_max + size) / (2 * size))))
    z = np.linspace(-size, z_max, resolution_z)

    # Create the 3D grid (use indexing='ij' to keep axis order [x, y, z])
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Evaluate the scalar field across the entire grid
    scalar_field = equation_func(X, Y, Z)

    # Create a binary volume using the threshold level
    volume = scalar_field > level

    # Compute voxel pitch for X and Y
    pitch_xy = (2 * size) / (resolution - 1)

    # For Z, based on the Z range
    pitch_z = (z_max + size) / (resolution_z - 1) if resolution_z > 1 else pitch_xy

    # Use the smaller pitch to maintain near-cubic voxels (or adjust as needed)
    pitch = min(pitch_xy, pitch_z)

    # Generate the mesh using marching cubes from the binary volume
    mesh = trimesh.voxel.ops.matrix_to_marching_cubes(volume, pitch=pitch)

    return mesh

# ===========================================
# Geometry generation functions using generate_continuous()
# ===========================================

def generate_gyroid_full(size, resolution, level, desired_height):
    def gyroid(X, Y, Z):
        return np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)
    return generate_continuous(gyroid, size, resolution, level, desired_height)

def generate_schwarz_p_full(size, resolution, level, desired_height):
    def schwarz_p(X, Y, Z):
        return np.cos(X) + np.cos(Y) + np.cos(Z)
    return generate_continuous(schwarz_p, size, resolution, level, desired_height)

def generate_schwarz_d_full(size, resolution, level, desired_height):
    def schwarz_d(X, Y, Z):
        return np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)
    return generate_continuous(schwarz_d, size, resolution, level, desired_height)

def generate_diamond_full(size, resolution, level, desired_height):
    def diamond(X, Y, Z):
        return np.cos(X)*np.cos(Y)*np.cos(Z) - np.sin(X)*np.sin(Y)*np.sin(Z)
    return generate_continuous(diamond, size, resolution, level, desired_height)

def generate_neovius_full(size, resolution, level, desired_height):
    def neovius(X, Y, Z):
        return (np.sin(X)*np.sin(Y)*np.sin(Z) +
                np.sin(X)*np.cos(Y)*np.cos(Z) +
                np.cos(X)*np.sin(Y)*np.cos(Z) +
                np.cos(X)*np.cos(Y)*np.sin(Z))
    return generate_continuous(neovius, size, resolution, level, desired_height)

def generate_iwp_full(size, resolution, level, desired_height):
    def iwp(X, Y, Z):
        return (np.cos(X - Y)*np.cos(Y - Z)*np.cos(Z - X) -
                np.cos(X + Y)*np.cos(Y + Z)*np.cos(Z + X))
    return generate_continuous(iwp, size, resolution, level, desired_height)

def generate_schoen_g_full(size, resolution, level, desired_height):
    def schoen_g(X, Y, Z):
        return (np.cos(X)*np.sin(Y)*np.cos(Z) +
                np.sin(X)*np.cos(Y)*np.sin(Z) +
                np.cos(X)*np.cos(Y)*np.cos(Z))
    return generate_continuous(schoen_g, size, resolution, level, desired_height)

def generate_double_diamond_full(size, resolution, level, desired_height):
    def double_diamond(X, Y, Z):
        return (np.cos(2*X)*np.cos(2*Y) +
                np.cos(2*Y)*np.cos(2*Z) +
                np.cos(2*Z)*np.cos(2*X))
    return generate_continuous(double_diamond, size, resolution, level, desired_height)

def generate_split_p_full(size, resolution, level, desired_height):
    def split_p(X, Y, Z):
        return (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z) -
                np.cos(X)*np.cos(Y)*np.cos(Z))
    return generate_continuous(split_p, size, resolution, level, desired_height)

def generate_split_d_full(size, resolution, level, desired_height):
    def split_d(X, Y, Z):
        return (np.sin(2*X)*np.cos(Y) +
                np.sin(2*Y)*np.cos(Z) +
                np.sin(2*Z)*np.cos(X))
    return generate_continuous(split_d, size, resolution, level, desired_height)

def generate_g_wp_full(size, resolution, level, desired_height):
    def g_wp(X, Y, Z):
        return (np.sin(X)*np.cos(Y) +
                np.sin(Y)*np.cos(Z) +
                np.sin(Z)*np.cos(X) +
                np.cos(X)*np.cos(Y)*np.cos(Z))
    return generate_continuous(g_wp, size, resolution, level, desired_height)

def generate_lidinoid_full(size, resolution, level, desired_height):
    def lidinoid(X, Y, Z):
        return (np.sin(X)*np.sin(2*Y) +
                np.sin(Y)*np.sin(2*Z) +
                np.sin(Z)*np.sin(2*X))
    return generate_continuous(lidinoid, size, resolution, level, desired_height)

def generate_schoen_c_full(size, resolution, level, desired_height):
    def schoen_c(X, Y, Z):
        return (np.cos(X) + np.cos(Y) + np.cos(Z) +
                np.cos(X+Y+Z))
    return generate_continuous(schoen_c, size, resolution, level, desired_height)

def generate_p_surface_full(size, resolution, level, desired_height):
    def p_surface(X, Y, Z):
        return (np.cos(X) + np.cos(Y) + np.cos(Z) -
                np.sin(X)*np.sin(Y)*np.sin(Z))
    return generate_continuous(p_surface, size, resolution, level, desired_height)

def generate_primitive_p_full(size, resolution, level, desired_height):
    def primitive_p(X, Y, Z):
        return (np.cos(X+Y+Z) -
                np.cos(X)*np.cos(Y)*np.cos(Z))
    return generate_continuous(primitive_p, size, resolution, level, desired_height)

def generate_elliptic_3_fold_full(size, resolution, level, desired_height):
    def elliptic_3_fold(X, Y, Z):
        return (np.cos(3*X) +
                np.cos(3*Y) +
                np.cos(3*Z))
    return generate_continuous(elliptic_3_fold, size, resolution, level, desired_height)

def generate_lamella_full(size, resolution, level, desired_height):
    def lamella(X, Y, Z):
        return (np.cos(2*X) +
                np.cos(2*Y) +
                np.cos(2*Z))
    return generate_continuous(lamella, size, resolution, level, desired_height)

def generate_diamond_primitive_full(size, resolution, level, desired_height):
    def diamond_primitive(X, Y, Z):
        return (np.cos(X)*np.cos(Y)*np.cos(Z) +
                np.sin(X)*np.sin(Y)*np.sin(Z))
    return generate_continuous(diamond_primitive, size, resolution, level, desired_height)

def generate_fractal_full(size, resolution, level, desired_height):
    def fractal(X, Y, Z):
        return (np.cos(X)*np.cos(Y)*np.cos(Z)*
                np.cos(X+Y+Z))
    return generate_continuous(fractal, size, resolution, level, desired_height)

def generate_cross_field_full(size, resolution, level, desired_height):
    def cross_field(X, Y, Z):
        return np.cos(2*X)*np.cos(2*Y)*np.cos(2*Z)
    return generate_continuous(cross_field, size, resolution, level, desired_height)

# ===========================================
# Utility functions
# ===========================================

def scale_mesh_to_dimensions(mesh, desired_dimensions_mm):
    """
    Scales the mesh so its bounding box fits within desired_dimensions_mm.
    """
    bb = mesh.bounds
    original_dims = bb[1] - bb[0]
    scale_factors = desired_dimensions_mm / original_dims
    min_scale = np.min(scale_factors)
    mesh.apply_scale(min_scale)
    return mesh


def cut_with_cylinder(mesh, desired_dimensions_mm):
    """
    Cuts the mesh to fit inside a cylindrical reactor defined by desired_dimensions_mm.
    """
    radius = desired_dimensions_mm[0] / 2.0
    height = desired_dimensions_mm[2]
    # Create a cylinder (extra height ensures full intersection)
    cylinder = trimesh.creation.cylinder(radius=radius, height=height * 1.5, sections=100)
    bb = mesh.bounds
    center = (bb[0] + bb[1]) / 2.0
    cylinder.apply_translation(center)
    
    cut_mesh = mesh.intersection(cylinder)
    return cut_mesh, cylinder


def calculate_tortuosity(mesh, num_samples=100, seed=42):
    """
    Calculates the tortuosity of the mesh using vertical ray tracing paths.
    """
    np.random.seed(seed)
    bb = mesh.bounds
    radius = (bb[1][0] - bb[0][0]) / 2.0
    height = bb[1][2] - bb[0][2]
    top_z = bb[1][2]
    bottom_z = bb[0][2]
    
    sample_points = []
    while len(sample_points) < num_samples:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        if x**2 + y**2 <= radius**2:
            sample_points.append([x, y, top_z])
    sample_points = np.array(sample_points)
    
    path_lengths = []
    for p in sample_points:
        start = p
        ray_origin = np.array([start])
        ray_direction = np.array([[0, 0, -1]])
        locations, _, _ = mesh.ray.intersects_location(ray_origin, ray_direction)
        if len(locations) > 0:
            path_length = 0
            prev = start
            for loc in locations:
                path_length += np.linalg.norm(loc - prev)
                prev = loc
            path_length += np.linalg.norm(prev - np.array([start[0], start[1], bottom_z]))
            path_lengths.append(path_length)
    if path_lengths:
        avg_path = np.mean(path_lengths)
        return avg_path / height
    else:
        return np.nan


def save_to_stl(mesh, geometry_name, size, resolution, level, dimensions_mm):
    """
    Saves the mesh to an STL file with descriptive filename.
    """
    filename = f'estructura_{geometry_name}_S_{size}_R_{resolution}_L_{level}_D_{dimensions_mm[0]}x{dimensions_mm[1]}x{dimensions_mm[2]}.stl'
    mesh.export(filename)
    print(f"Mesh saved to: {filename}")


def save_metrics_to_excel(metrics, filename='geometries_metrics.xlsx'):
    """
    Saves the list of metric dictionaries to an Excel file.
    """
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False)
    print(f"Metrics saved to: {filename}")

# ===========================================
# Main execution block
# ===========================================

if __name__ == "__main__":
    # Generation parameters (in arbitrary units for the original domain)
    size = 35            # Domain in X and Y: [-size, size]
    resolution = 300     # Resolution for X and Y
    level = 0            # Threshold for the isosurface
    height = 10

    # Desired reactor dimensions in mm: [diameter, diameter, height]
    desired_dimensions_mm = np.array([10, 10, height * size / 5])

    # List of geometry generation functions to use (continuous version)
    # Add or comment out functions based on which geometries you want to generate
    geometry_functions = [
        #generate_gyroid_full,
        #generate_schwarz_p_full,
        #generate_diamond_full,
        #generate_neovius_full,
        #generate_iwp_full,
        generate_schoen_g_full,
        #generate_double_diamond_full,
        #generate_split_p_full,
        #generate_split_d_full,
        #generate_g_wp_full,
        #generate_lidinoid_full,
        #generate_schoen_c_full,
        #generate_p_surface_full,
        #generate_primitive_p_full,
        #generate_elliptic_3_fold_full,
        #generate_lamella_full,
        #generate_diamond_primitive_full,
        #generate_fractal_full,
        #generate_cross_field_full,
    ]

    # List to store metrics for each generated geometry
    metrics = []

    # Generate each selected geometry
    for func in geometry_functions:
        geometry_name = func.__name__.replace("generate_", "").replace("_full", "")
        print(f"Generating geometry: {geometry_name}")

        # Generate the extended continuous mesh in Z (using desired height in mm)
        mesh = func(size, resolution, level, desired_dimensions_mm[2])

        # Scale mesh to fit the desired bounding box dimensions (X, Y, Z)
        scaled_mesh = scale_mesh_to_dimensions(mesh, desired_dimensions_mm)

        # Cut mesh to fit cylindrical reactor shape
        cylindrical_mesh, cylinder = cut_with_cylinder(scaled_mesh, desired_dimensions_mm)

        # Save mesh to STL file
        save_to_stl(cylindrical_mesh, geometry_name, size, resolution, level, desired_dimensions_mm)

        # Calculate metrics: surface area, volume, tortuosity
        surface_area = cylindrical_mesh.area
        structure_volume = cylindrical_mesh.volume
        radius_mm = desired_dimensions_mm[0] / 2.0
        height_mm = desired_dimensions_mm[2]
        cylinder_volume = np.pi * radius_mm**2 * height_mm
        volume_fraction = structure_volume / cylinder_volume * 100
        tortuosity = calculate_tortuosity(cylindrical_mesh)

        metrics.append({
            'Geometry': geometry_name,
            'Surface Area (sq mm)': surface_area,
            'Structure Volume (cu mm)': structure_volume,
            'Cylinder Volume (cu mm)': cylinder_volume,
            'Volume Fraction (%)': volume_fraction,
            'Tortuosity': tortuosity
        })

    # Save metrics to Excel file
    save_metrics_to_excel(metrics)
    print("All metrics saved.")