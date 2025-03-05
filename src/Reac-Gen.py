"""
Reac-Gen
Reac-Gen generates 3D reactor geometries using 20 mathematical surface equations, automates scaling and slicing, 
calculates key metrics (e.g., surface area, tortuosity), and exports STL files for 3D printing with corresponding Excel data.
@author: Cristopher Tinajero
"""

import numpy as np
import trimesh
import pandas as pd

# ===========================
# Geometry Generation Functions
# ===========================

# Function to generate a gyroid structure based on implicit surface equations
def generate_gyroid(size, resolution, level):
    # Create 3D grid of points using the specified size and resolution
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    # Define the gyroid equation
    gyroid = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)
    # Create binary volume matrix by applying the level threshold
    gyroid_volume = gyroid > level
    # Convert binary matrix to 3D mesh using marching cubes
    gyroid_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(gyroid_volume, pitch=2*size/resolution)
    
    return gyroid_mesh

def generate_schwarz_p(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    schwarz_p = np.cos(X) + np.cos(Y) + np.cos(Z)

    schwarz_p_volume = schwarz_p > level
    schwarz_p_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(schwarz_p_volume, pitch=2*size/resolution)
    
    return schwarz_p_mesh

def generate_schwarz_d(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    schwarz_d = np.sin(X) * np.cos(Y) + np.sin(Y) * np.cos(Z) + np.sin(Z) * np.cos(X)

    schwarz_d_volume = schwarz_d > level
    schwarz_d_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(schwarz_d_volume, pitch=2*size/resolution)
    
    return schwarz_d_mesh

def generate_diamond(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    diamond = np.cos(X) * np.cos(Y) * np.cos(Z) - np.sin(X) * np.sin(Y) * np.sin(Z)

    diamond_volume = diamond > level
    diamond_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(diamond_volume, pitch=2*size/resolution)
    
    return diamond_mesh

def generate_neovius(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    neovius = (np.sin(X) * np.sin(Y) * np.sin(Z) +
               np.sin(X) * np.cos(Y) * np.cos(Z) +
               np.cos(X) * np.sin(Y) * np.cos(Z) +
               np.cos(X) * np.cos(Y) * np.sin(Z))

    neovius_volume = neovius > level
    neovius_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(neovius_volume, pitch=2*size/resolution)
    
    return neovius_mesh

def generate_iwp(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    iwp = (np.cos(X - Y) * np.cos(Y - Z) * np.cos(Z - X) -
           np.cos(X + Y) * np.cos(Y + Z) * np.cos(Z + X))

    iwp_volume = iwp > level
    iwp_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(iwp_volume, pitch=2*size/resolution)
    
    return iwp_mesh

def generate_schoen_g(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    schoen_g = (np.cos(X) * np.sin(Y) * np.cos(Z) +
                np.sin(X) * np.cos(Y) * np.sin(Z) +
                np.cos(X) * np.cos(Y) * np.cos(Z))

    schoen_g_volume = schoen_g > level
    schoen_g_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(schoen_g_volume, pitch=2*size/resolution)
    
    return schoen_g_mesh

def generate_double_diamond(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    double_diamond = (np.cos(2*X) * np.cos(2*Y) +
                      np.cos(2*Y) * np.cos(2*Z) +
                      np.cos(2*Z) * np.cos(2*X))

    double_diamond_volume = double_diamond > level
    double_diamond_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(double_diamond_volume, pitch=2*size/resolution)
    
    return double_diamond_mesh

def generate_split_p(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    split_p = (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z) -
               np.cos(X) * np.cos(Y) * np.cos(Z))

    split_p_volume = split_p > level
    split_p_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(split_p_volume, pitch=2*size/resolution)
    
    return split_p_mesh

def generate_split_d(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    split_d = (np.sin(2*X) * np.cos(Y) +
               np.sin(2*Y) * np.cos(Z) +
               np.sin(2*Z) * np.cos(X))

    split_d_volume = split_d > level
    split_d_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(split_d_volume, pitch=2*size/resolution)
    
    return split_d_mesh

def generate_g_wp(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    g_wp = (np.sin(X) * np.cos(Y) +
            np.sin(Y) * np.cos(Z) +
            np.sin(Z) * np.cos(X) +
            np.cos(X) * np.cos(Y) * np.cos(Z))
   
    g_wp_volume = g_wp > level
    g_wp_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(g_wp_volume, pitch=2*size/resolution)
    
    return g_wp_mesh

def generate_lidinoid(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    lidinoid = (np.sin(X) * np.sin(2*Y) +
                np.sin(Y) * np.sin(2*Z) +
                np.sin(Z) * np.sin(2*X))

    lidinoid_volume = lidinoid > level
    lidinoid_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(lidinoid_volume, pitch=2*size/resolution)
    
    return lidinoid_mesh

def generate_schoen_c(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    schoen_c = (np.cos(X) + np.cos(Y) + np.cos(Z) +
                np.cos(X + Y + Z))

    schoen_c_volume = schoen_c > level
    schoen_c_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(schoen_c_volume, pitch=2*size/resolution)
    
    return schoen_c_mesh

def generate_p_surface(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    p_surface = (np.cos(X) + np.cos(Y) + np.cos(Z) -
                 np.sin(X) * np.sin(Y) * np.sin(Z))

    p_surface_volume = p_surface > level
    p_surface_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(p_surface_volume, pitch=2*size/resolution)
    
    return p_surface_mesh

def generate_primitive_p(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    primitive_p = (np.cos(X + Y + Z) -
                   np.cos(X) * np.cos(Y) * np.cos(Z))

    primitive_p_volume = primitive_p > level
    primitive_p_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(primitive_p_volume, pitch=2*size/resolution)
    
    return primitive_p_mesh

def generate_elliptic_3_fold(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    elliptic_3_fold = (np.cos(3*X) +
                       np.cos(3*Y) +
                       np.cos(3*Z))

    elliptic_3_fold_volume = elliptic_3_fold > level
    elliptic_3_fold_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(elliptic_3_fold_volume, pitch=2*size/resolution)
    
    return elliptic_3_fold_mesh

def generate_lamella(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    lamella = (np.cos(2*X) +
               np.cos(2*Y) +
               np.cos(2*Z))

    lamella_volume = lamella > level
    lamella_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(lamella_volume, pitch=2*size/resolution)
    
    return lamella_mesh

def generate_diamond_primitive(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    diamond_primitive = (np.cos(X) * np.cos(Y) * np.cos(Z) +
                         np.sin(X) * np.sin(Y) * np.sin(Z))

    diamond_primitive_volume = diamond_primitive > level
    diamond_primitive_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(diamond_primitive_volume, pitch=2*size/resolution)
    
    return diamond_primitive_mesh

def generate_fractal(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    fractal = (np.cos(X) * np.cos(Y) * np.cos(Z) *
               np.cos(X + Y + Z))

    fractal_volume = fractal > level
    fractal_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(fractal_volume, pitch=2*size/resolution)
    
    return fractal_mesh

def generate_cross_field(size, resolution, level):
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)
    z = np.linspace(-size, size, resolution)
    X, Y, Z = np.meshgrid(x, y, z)
    cross_field = (np.cos(2*X) * np.cos(2*Y) * np.cos(2*Z))

    cross_field_volume = cross_field > level
    cross_field_mesh = trimesh.voxel.ops.matrix_to_marching_cubes(cross_field_volume, pitch=2*size/resolution)
    
    return cross_field_mesh

# ===========================
# Utility Functions
# ===========================

# Function to scale a 3D mesh to fit specific dimensions
def scale_mesh_to_dimensions(mesh, desired_dimensions_mm):
    bounding_box = mesh.bounds
    original_dimensions = bounding_box[1] - bounding_box[0]  # [x, y, z]
    # Calculate scaling factors for each dimension
    scale_factors = desired_dimensions_mm / original_dimensions
    # Use the smallest scale factor to maintain proportions
    min_scale_factor = np.min(scale_factors)
    mesh.apply_scale(min_scale_factor)
    
    return mesh

# Function to cut a mesh to fit within a cylindrical boundary
def cut_with_cylinder(mesh, desired_dimensions_mm):
    radius = desired_dimensions_mm[0] / 2.0
    height = desired_dimensions_mm[2]
    # Create a cylindrical mesh for intersection
    cylinder = trimesh.creation.cylinder(radius=radius, height=height * 1.5, sections=100)
    # Center the cylinder on the mesh
    bounding_box = mesh.bounds
    center_x = (bounding_box[1][0] + bounding_box[0][0]) / 2.0
    center_y = (bounding_box[1][1] + bounding_box[0][1]) / 2.0
    center_z = (bounding_box[1][2] + bounding_box[0][2]) / 2.0
    cylinder.apply_translation([center_x, center_y, center_z])
    # Perform the intersection to clip the mesh
    cut_mesh = mesh.intersection(cylinder)
    
    return cut_mesh, cylinder

# Function to stack cylindrical meshes to achieve the desired height
def stack_cylinders(cylinder_mesh, desired_height_mm):
    current_height = cylinder_mesh.bounding_box.extents[2] / 10  # Convert to cm
    overlap_factor = 0.01
    num_repeats = int(np.ceil(desired_height_mm / (current_height * (1 - overlap_factor))))
    meshes = [cylinder_mesh.copy() for _ in range(num_repeats)]
    
    for i, m in enumerate(meshes):
        m.apply_translation([0, 0, i * current_height * (1 - overlap_factor) * 10])  # Back to mm
    
    combined_mesh = trimesh.util.concatenate(meshes)
    # Slice the combined mesh to ensure it fits the desired height
    z_min = combined_mesh.bounds[0][2]
    z_max = z_min + desired_height_mm  # Convert to mm
    
    plane_origin = [0, 0, z_max]
    plane_normal = [0, 0, -1]
    
    sliced_mesh = combined_mesh.slice_plane(plane_origin, plane_normal)
    
    return sliced_mesh
# Function to save a 3D mesh as an STL file
def save_to_stl(mesh, geometry_name, size, resolution, level, dimensions_mm):
    filename = f'esturctura_{geometry_name}_S_{size}_R_{resolution}_L_{level}_D_{dimensions_mm[0]}x{dimensions_mm[1]}x{dimensions_mm[2]}.stl'
    mesh.export(filename)

# Function to calculate the tortuosity of a structure
def calculate_tortuosity(mesh, num_samples=100, seed=42):
    np.random.seed(seed)  # Set random seed for reproducibility

    bounding_box = mesh.bounds
    radius = (bounding_box[1][0] - bounding_box[0][0]) / 2.0
    height = bounding_box[1][2] - bounding_box[0][2]
    top_surface_z = bounding_box[1][2]
    bottom_surface_z = bounding_box[0][2]
    # Generate random sample points on the top surface within the cylindrical radius
    sample_points = []
    while len(sample_points) < num_samples:
        x = np.random.uniform(-radius, radius)
        y = np.random.uniform(-radius, radius)
        if x**2 + y**2 <= radius**2:
            sample_points.append([x, y, top_surface_z])
    sample_points = np.array(sample_points)
    # Calculate path lengths for each sample point
    path_lengths = []
    for point in sample_points:
        start_point = point
        ray_origins = np.array([start_point])
        ray_directions = np.array([[0, 0, -1]])
        locations, index_ray, index_tri = mesh.ray.intersects_location(ray_origins, ray_directions)
        if len(locations) > 0:
            path_length = 0
            previous_point = start_point
            for loc in locations:
                path_length += np.linalg.norm(loc - previous_point)
                previous_point = loc
            path_length += np.linalg.norm(previous_point - [start_point[0], start_point[1], bottom_surface_z])
            path_lengths.append(path_length)
    # Compute average path length and calculate tortuosity
    if path_lengths:
        average_path_length = np.mean(path_lengths)
        tortuosity = average_path_length / height
    else:
        tortuosity = np.nan
    return tortuosity

# Function to save calculated metrics to an Excel file
def save_metrics_to_excel(metrics, filename='geometries_metrics.xlsx'):
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False)
    print(f"Metrics saved to {filename}")

# ===========================
# Main Execution
# ===========================

size = 10  # Defines the spatial boundary of the scalar field
resolution = 300  # Specifies the number of sample points along each axis
level = 0  # Determines the isosurface cutoff for the structure

# Example dimensions in mm
desired_dimensions_mm = np.array([10, 10, 10])  # Width, depth, and height of the reactor


desired_height_mm = 50 # Total height of the stacked cylinders

# List of geometry generation functions
geometry_functions = [
    generate_gyroid
    # Include additional functions as needed
]

# Lista para almacenar las métricas de todas las geometrías
metrics = []

# Loop through each geometry generation function
for func in geometry_functions:
    geometry_name = func.__name__.replace("generate_", "")  # Extract the geometry name
    print(f"Generating {geometry_name} geometry...")
    mesh = func(size, resolution, level)  # Generate the geometry
    scaled_mesh = scale_mesh_to_dimensions(mesh, desired_dimensions_mm)  # Scale to desired dimensions
    cylindrical_mesh, cylinder = cut_with_cylinder(scaled_mesh, desired_dimensions_mm)  # Cut to fit cylindrical reactor
    stacked_mesh = stack_cylinders(cylindrical_mesh, desired_height_mm)  # Stack cylinders to reach desired height
    save_to_stl(stacked_mesh, geometry_name, size, resolution, level, desired_dimensions_mm)  # Save as STL file
    surface_area = stacked_mesh.area  # Calculate surface area
    structure_volume = stacked_mesh.volume  # Calculate structure volume
    radius_mm = desired_dimensions_mm[0] / 2.0
    height_mm = desired_height_mm
    cylinder_volume = np.pi * (radius_mm**2) * height_mm  # Calculate cylinder volume
    fraction_volume = structure_volume / cylinder_volume  # Calculate volume fraction
    tortuosity = calculate_tortuosity(stacked_mesh)  # Calculate tortuosity
    metrics.append({
        'Geometry': geometry_name,
        'Surface Area (sq mm)': surface_area,
        'Structure Volume (cu mm)': structure_volume,
        'Cylinder Volume (cu mm)': cylinder_volume,
        'Volume Fraction (%)': fraction_volume * 100,
        'Tortuosity': tortuosity
    })

# Save all calculated metrics to an Excel file
save_metrics_to_excel(metrics)
print("All geometries have been generated, saved as STL files, and their metrics have been calculated and saved to an Excel file.")
