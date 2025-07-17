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
# Functions for slicing based on scaled function domain
# ===========================================

def generate_function_slices_2D(equation_func, resolution_xy, desired_dimensions_mm,
                                dz_um=10, level=0, output_folder='slices',
                                cmap='Reds', make_images=0, flow_ml_per_min=0.3,
                                density_g_cc=0.0528, viscosity_Pa_s=4.09e-5):

    # Create output folder
    os.makedirs(output_folder, exist_ok=True)

    # Geometric parameters
    diameter_mm = desired_dimensions_mm[0]
    height_mm   = desired_dimensions_mm[2]
    radius      = diameter_mm / 2.0

    # XY sampling grid and circular mask
    x = np.linspace(0, diameter_mm, resolution_xy)
    y = np.linspace(0, diameter_mm, resolution_xy)
    X_plot, Y_plot = np.meshgrid(x, y)
    X = X_plot - radius
    Y = Y_plot - radius
    mask = (X**2 + Y**2) <= radius**2

    # Spatial resolution and pixel area
    dx = diameter_mm / resolution_xy
    dy = dx
    pixel_area = dx * dy

    # Z slicing
    num_slices = int(np.ceil(height * 1000 / dz_um_I))
    z_values   = np.linspace(0, height, num_slices)

    # Z coordinates in microns and mm (for labeling)
    Z_um       = np.arange(num_slices) * dz_um_I
    Z_mm_plot  = Z_um * 1e-3

    # Flow conversion: ml/min → mm³/s
    Q_mm3_s = flow_ml_per_min * 1e3 / 60.0

    # Fluid properties
    rho = density_g_cc * 1000.0  # g/cc to kg/m³
    mu  = viscosity_Pa_s         # Pa·s

    # Metrics lists
    void_areas = []     # Ae(z)
    per_diff   = []     # P(z) = interior - exterior

    # Process each slice
    for i, z in enumerate(tqdm(z_values, desc="Slices")):
        # Scale domain to original function (uses global 'size')
        X_scaled = (X / diameter_mm) * 2 * size
        Y_scaled = (Y / diameter_mm) * 2 * size
        Z_scaled = ((z / height_mm) * 2 * size) - size
        scalar_field = equation_func(
            X_scaled,
            Y_scaled,
            np.full_like(X_scaled, Z_scaled)
        )

        # Apply circular mask
        field_masked = np.where(mask, scalar_field, np.nan)

        # Transversal void area Ae(z)
        void_mask = (scalar_field <= level) & mask
        ae = np.sum(void_mask) * pixel_area
        void_areas.append(ae)

        # Internal perimeter (solid contours sum)
        solid = (~void_mask) & mask
        contours = measure.find_contours(solid.astype(float), 0.5)
        per_int = 0.0
        for c in contours:
            coords = np.column_stack([c[:,1]*dx, c[:,0]*dy])
            diffs  = np.diff(coords, axis=0)
            per_int += np.hypot(diffs[:,0], diffs[:,1]).sum()
            per_int += np.linalg.norm(coords[0] - coords[-1])

        # External perimeter (circle segments in contact with solid)
        circle_contour = measure.find_contours(mask.astype(float), 0.5)[0]
        cc = np.column_stack([circle_contour[:,1]*dx, circle_contour[:,0]*dy])
        center = np.array([radius, radius])
        per_ext = 0.0
        for j in range(len(cc)):
            p1, p2 = cc[j], cc[(j+1) % len(cc)]
            mid    = 0.5 * (p1 + p2)
            v      = mid - center
            if np.linalg.norm(v) == 0:
                continue
            test_pt = mid - (v / np.linalg.norm(v)) * (0.5 * dx)
            ix = int(round(test_pt[1] / dy))
            jx = int(round(test_pt[0] / dx))
            if (0 <= ix < resolution_xy) and (0 <= jx < resolution_xy):
                if solid[ix, jx]:
                    per_ext += np.linalg.norm(p2 - p1)

        # Difference P(z)
        per_diff.append(per_int - per_ext)

        # Image generation (if enabled)
        if make_images:
            z_um_mm = (i * dz_um_I) / 1000.0
            fig, ax = plt.subplots(figsize=(5, 5))
            ax.set_title(
                f"Z = {z_um_mm:.2f} mm — Ae: {ae:.2f} mm² — P(z): {per_int-per_ext:.2f} mm"
            )
            ax.set_xlim(0, diameter_mm)
            ax.set_ylim(0, diameter_mm)
            ax.set_aspect('equal')

            ax.contourf(
                X_plot, Y_plot, field_masked,
                levels=[level, np.nanmax(field_masked)], cmap=cmap
            )
            ax.contour(
                X_plot, Y_plot, field_masked,
                levels=[level], colors='red', linewidths=0.5
            )
            ax.contourf(
                X_plot, Y_plot, void_mask,
                levels=[0.5, 1], colors='#ADD8E6', alpha=0.3
            )

            for c in contours:
                coords = np.column_stack([c[:,1]*dx, c[:,0]*dy])
                ax.plot(coords[:,0], coords[:,1], color='yellow', linewidth=2)

            for j in range(len(cc)):
                p1, p2 = cc[j], cc[(j+1) % len(cc)]
                mid    = 0.5 * (p1 + p2)
                v      = mid - center
                if np.linalg.norm(v) == 0:
                    continue
                test_pt = mid - (v / np.linalg.norm(v)) * (0.5 * dx)
                ix = int(round(test_pt[1] / dy))
                jx = int(round(test_pt[0] / dx))
                if (0 <= ix < resolution_xy) and (0 <= jx < resolution_xy):
                    if solid[ix, jx]:
                        ax.plot(
                            [p1[0], p2[0]], [p1[1], p2[1]],
                            color='orange', linewidth=2
                        )

            circle_patch = plt.Circle((radius, radius), radius,
                                      color='black', fill=False, linewidth=1)
            ax.add_patch(circle_patch)

            plt.savefig(os.path.join(output_folder, f"slice_{i:04d}.png"), dpi=200)
            plt.close()

    # Calculate total area At and height L
    At = np.pi * (diameter_mm ** 2) / 4.0
    L  = height

    Z_um = np.arange(num_slices) * dz_um_I

    Ae = np.array(void_areas)
    Pz = np.array(per_diff)
    eps_G = Ae / At
    asp   = Pz / At
    dh    = 4 * Ae / Pz
    u_z   = Q_mm3_s / Ae
    Re_z = rho * (u_z*1e-3) * (dh*1e-3) / mu

    df = pd.DataFrame({
        'Z_um':           Z_um,
        'Ae(z)_mm2':      Ae,
        'P(z)_mm':        Pz,
        'εG(z) (Ae/At)':  eps_G,
        'a_sp(z)_mm-1':   asp,
        'd_h(z)_mm':      dh,
        'u(z)_mm/s':      u_z,
        'Re(z)':          Re_z,
        'At_mm2':         At,
        'L_mm':           L
    })

    df.to_excel(os.path.join(output_folder, "slice_metrics.xlsx"), index=False)
    print(f"[OK] Data saved to {output_folder} (images={'yes' if make_images else 'no'})")
    
# ===========================================
# Function to generate continuous mesh from a scalar field
# ===========================================
def generate_continuous(equation_func, size, resolution, level, desired_height):
    """
    Generates a continuous 3D mesh using a scalar field defined by 'equation_func'.

    Parameters:
        equation_func: A function f(X, Y, Z) returning the scalar field.
        size: Domain boundary along X and Y, from -size to +size.
        resolution: Number of points sampled in X and Y.
        level: Isosurface threshold level.
        desired_height: Final height of the geometry in the Z-direction.

    Returns:
        A 3D mesh (trimesh.Trimesh) constructed using marching cubes.
    """
    # Define X and Y domain: [-size, size]
    x = np.linspace(-size, size, resolution)
    y = np.linspace(-size, size, resolution)

    # Determine Z domain so that mesh spans from -size to (desired_height - size)
    z_max = desired_height - size
    resolution_z = int(np.ceil(resolution * ((z_max + size) / (2 * size))))
    z = np.linspace(-size, z_max, resolution_z)

    # Create 3D grid of points
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Evaluate scalar field
    scalar_field = equation_func(X, Y, Z)

    # Threshold the volume
    volume = scalar_field > level

    # Compute voxel pitch for accurate scaling
    pitch_xy = (2 * size) / (resolution - 1)
    pitch_z = (z_max + size) / (resolution_z - 1) if resolution_z > 1 else pitch_xy
    pitch = min(pitch_xy, pitch_z)

    # Generate mesh using marching cubes
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
    Scales the mesh proportionally so that its bounding box fits within the specified dimensions.
    """
    bb = mesh.bounds
    original_dims = bb[1] - bb[0]
    scale_factors = desired_dimensions_mm / original_dims
    min_scale = np.min(scale_factors)
    mesh.apply_scale(min_scale)
    return mesh

def cut_with_cylinder(mesh, desired_dimensions_mm):
    """
    Cuts the mesh to fit within a cylindrical boundary defined by the given dimensions.
    """
    radius = desired_dimensions_mm[0] / 2.0
    height = desired_dimensions_mm[2]
    cylinder = trimesh.creation.cylinder(radius=radius, height=height * 1.5, sections=100)
    bb = mesh.bounds
    center = (bb[0] + bb[1]) / 2.0
    cylinder.apply_translation(center)
    cut_mesh = mesh.intersection(cylinder)
    return cut_mesh, cylinder

def calculate_tortuosity(mesh, num_samples=100, seed=42):
    """
    Estimates the tortuosity of the mesh by simulating vertical ray paths from top to bottom.
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
        ray_origin = np.array([p])
        ray_direction = np.array([[0, 0, -1]])
        locations, _, _ = mesh.ray.intersects_location(ray_origin, ray_direction)
        if len(locations) > 0:
            path_length = 0
            prev = p
            for loc in locations:
                path_length += np.linalg.norm(loc - prev)
                prev = loc
            path_length += np.linalg.norm(prev - np.array([p[0], p[1], bottom_z]))
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
    filename = f"estructura_{geometry_name}_S_{size}_R_{resolution}_L_{level}_D_{dimensions_mm[0]}x{dimensions_mm[1]}x{dimensions_mm[2]}.stl"
    mesh.export(filename)
    print(f"Mesh saved to: {filename}")

def save_metrics_to_excel(metrics, filename='geometries_metrics.xlsx'):
    """
    Saves a list of metric dictionaries to an Excel file.
    """
    df = pd.DataFrame(metrics)
    df.to_excel(filename, index=False)
    print(f"Metrics saved to: {filename}")


# ===========================================
# Main execution block
# ===========================================

if __name__ == "__main__":
    size = 10
    resolution = 300
    level = 0
    height = 110
    dz_um_I = 10000

    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "outputs")
    os.makedirs(output_dir, exist_ok=True)

    desired_dimensions_mm = np.array([10, 10, height * size / 5])

    geometry_functions = [
        generate_gyroid_full,  # You can enable other geometries as needed
    ]

    function_map = {
        "gyroid": lambda X, Y, Z: (np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)),
        "schwarz_p": lambda X, Y, Z: (np.cos(X) + np.cos(Y) + np.cos(Z)),
        "schwarz_d": lambda X, Y, Z: (np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X)),
        "diamond": lambda X, Y, Z: (np.cos(X)*np.cos(Y)*np.cos(Z) - np.sin(X)*np.sin(Y)*np.sin(Z)),
        "neovius": lambda X, Y, Z: (
            np.sin(X)*np.sin(Y)*np.sin(Z)
            + np.sin(X)*np.cos(Y)*np.cos(Z)
            + np.cos(X)*np.sin(Y)*np.cos(Z)
            + np.cos(X)*np.cos(Y)*np.sin(Z)),
        "iwp": lambda X, Y, Z: (np.cos(X - Y)*np.cos(Y - Z)*np.cos(Z - X) - np.cos(X + Y)*np.cos(Y + Z)*np.cos(Z + X)),
        "schoen_g": lambda X, Y, Z: (np.cos(X)*np.sin(Y)*np.cos(Z) + np.sin(X)*np.cos(Y)*np.sin(Z) + np.cos(X)*np.cos(Y)*np.cos(Z)),
        "double_diamond": lambda X, Y, Z: (np.cos(2*X)*np.cos(2*Y) + np.cos(2*Y)*np.cos(2*Z) + np.cos(2*Z)*np.cos(2*X)),
        "split_p": lambda X, Y, Z: (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z) - np.cos(X)*np.cos(Y)*np.cos(Z)),
        "split_d": lambda X, Y, Z: (np.sin(2*X)*np.cos(Y) + np.sin(2*Y)*np.cos(Z) + np.sin(2*Z)*np.cos(X)),
        "g_wp": lambda X, Y, Z: (np.sin(X)*np.cos(Y) + np.sin(Y)*np.cos(Z) + np.sin(Z)*np.cos(X) + np.cos(X)*np.cos(Y)*np.cos(Z)),
        "lidinoid": lambda X, Y, Z: (np.sin(X)*np.sin(2*Y) + np.sin(Y)*np.sin(2*Z) + np.sin(Z)*np.sin(2*X)),
        "schoen_c": lambda X, Y, Z: (np.cos(X) + np.cos(Y) + np.cos(Z) + np.cos(X+Y+Z)),
        "p_surface": lambda X, Y, Z: (np.cos(X) + np.cos(Y) + np.cos(Z) - np.sin(X)*np.sin(Y)*np.sin(Z)),
        "primitive_p": lambda X, Y, Z: (np.cos(X+Y+Z) - np.cos(X)*np.cos(Y)*np.cos(Z)),
        "elliptic_3_fold": lambda X, Y, Z: (np.cos(3*X) + np.cos(3*Y) + np.cos(3*Z)),
        "lamella": lambda X, Y, Z: (np.cos(2*X) + np.cos(2*Y) + np.cos(2*Z)),
        "diamond_primitive": lambda X, Y, Z: (np.cos(X)*np.cos(Y)*np.cos(Z) + np.sin(X)*np.sin(Y)*np.sin(Z)),
        "fractal": lambda X, Y, Z: (np.cos(X)*np.cos(Y)*np.cos(Z)*np.cos(X+Y+Z)),
        "cross_field": lambda X, Y, Z: (np.cos(2*X)*np.cos(2*Y)*np.cos(2*Z))
    }

    metrics = []

    for func in geometry_functions:
        geometry_name = func.__name__.replace("generate_", "").replace("_full", "")
        print(f"\n[INFO] Generating geometry: {geometry_name}")

        geom_folder = os.path.join(output_dir, f"{geometry_name}_S{size}_L{level}")
        os.makedirs(geom_folder, exist_ok=True)

        mesh = func(size, resolution, level, desired_dimensions_mm[2])
        scaled_mesh = scale_mesh_to_dimensions(mesh, desired_dimensions_mm)
        cylindrical_mesh, _ = cut_with_cylinder(scaled_mesh, desired_dimensions_mm)
        stl_path = os.path.join(geom_folder, "estructura.stl")
        cylindrical_mesh.export(stl_path)

        surface_area = cylindrical_mesh.area
        structure_volume = cylindrical_mesh.volume
        radius_mm = desired_dimensions_mm[0] / 2.0
        height_mm = desired_dimensions_mm[2]
        cylinder_volume = np.pi * radius_mm**2 * height_mm
        volume_fraction = structure_volume / cylinder_volume * 100
        tortuosity = calculate_tortuosity(cylindrical_mesh)

        slices_folder = os.path.join(geom_folder, "slices")
        generate_function_slices_2D(
            equation_func=function_map[geometry_name],
            resolution_xy=400,
            desired_dimensions_mm=desired_dimensions_mm,
            dz_um=dz_um_I * 2,
            level=level,
            output_folder=slices_folder,
            make_images=0
        )

        slice_data_path = os.path.join(slices_folder, "slice_metrics.xlsx")
        dh_array = None
        if os.path.exists(slice_data_path):
            df_slice = pd.read_excel(slice_data_path)
            if 'd_h(z)_mm' in df_slice.columns:
                dh_array = df_slice['d_h(z)_mm'].to_numpy()

        geom_metrics = {
            'Geometry': geometry_name,
            'Surface Area (sq mm)': surface_area,
            'Structure Volume (cu mm)': structure_volume,
            'Cylinder Volume (cu mm)': cylinder_volume,
            'Volume Fraction (%)': volume_fraction,
            'Tortuosity': tortuosity
        }
        metrics.append(geom_metrics)

    summary_path = os.path.join(output_dir, "geometries_metrics.xlsx")
    pd.DataFrame(metrics).to_excel(summary_path, index=False)
    print(f"\n[FINISHED] All metrics saved to: {summary_path}")


