import open3d as o3d
import numpy as np
import os
import cv2
import json
import matplotlib.pyplot as plt
import math
from scipy.spatial.transform import Rotation as R

def generate_color_palette(num_colors):
    """Generates a list of distinct colors."""
    return plt.cm.get_cmap('hsv', num_colors)

def is_valid_rotation_matrix(R):
    if R.shape[0] != R.shape[1]:
        return False
    if not np.isclose(np.linalg.det(R), 1):
        return False
    identity_approx = np.dot(R, R.T)
    return np.allclose(identity_approx, np.eye(R.shape[0]))

def check_extrinsic_matrix(extrinsic_matrix):
    if extrinsic_matrix.shape != (4, 4):
        return False
    rotation_matrix = extrinsic_matrix[:3, :3]
    if not is_valid_rotation_matrix(rotation_matrix):
        return False
    return True


def depth_map_to_point_cloud(depth_map, intrinsic_matrix, extrinsic_matrix, additive_factor):

    if not check_extrinsic_matrix(extrinsic_matrix):

        R = extrinsic_matrix[:3, :3]
        U, _, Vt = np.linalg.svd(R)
        R_corrected = np.dot(U, Vt)

        extrinsic_matrix[:3, :3] = R_corrected

    
    v, u = np.meshgrid(np.arange(depth_map.shape[0]), np.arange(depth_map.shape[1]), indexing='ij')

    # Flatten the arrays
    u_flatten = u.flatten()
    v_flatten = v.flatten()
    depth_flatten = depth_map.flatten()

    # Filter out pixels with a depth of zero
    valid_depth_indices = depth_flatten > 0
    u_filtered = u_flatten[valid_depth_indices]
    v_filtered = v_flatten[valid_depth_indices]
    depth_filtered = depth_flatten[valid_depth_indices]

    depth_filtered = depth_filtered / 1.089

    # Compute the 3D points in camera coordinates
    x = (u_filtered - intrinsic_matrix[0, 2]) * depth_filtered / intrinsic_matrix[0, 0]
    y = (v_filtered - intrinsic_matrix[1, 2]) * depth_filtered / intrinsic_matrix[1, 1]

    z = depth_filtered + additive_factor

    # Stack to get homogeneous coordinates
    points_camera = np.vstack((x, y, z, np.ones_like(x)))

    # flip the z axis of the extrinsic matrix by flipping the sign of all elements in the third column
    extrinsic_matrix[:3, 2] = -extrinsic_matrix[:3, 2]
    extrinsic_matrix[:3, 1] = -extrinsic_matrix[:3, 1]
            

    # Transform to world coordinates
    points_world = np.dot(extrinsic_matrix, points_camera)
    points_world = points_world[:, ::100]

    return points_world[:3].T


def process_blender_images(config):
    # Load transforms
    with open(config['transforms_path'], 'r') as file:
        transforms = json.load(file)

    # Generate color palette
    color_palette = generate_color_palette(len(transforms['frames']))

    # Calculate intrinsic matrix based on config
    fx = config['focal_length_mm'] * config['image_size'][0] / config['sensor_size_mm']
    fy = config['focal_length_mm'] * config['image_size'][1] / config['sensor_size_mm']
    cx, cy = config['image_size'][0] / 2, config['image_size'][1] / 2
    default_intrinsics = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    # Process each frame
    for additive_factor in config['additive_factors']:
        combined_cloud = o3d.geometry.PointCloud()
        for i, element in enumerate(transforms['frames']):
            depth_map_path = config['depths_dir'] + 'Image' + element['file_path'].split('/')[1].split('.')[0] + '.exr'
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)
            depth_map[depth_map > config['max_depth']] = 0.0

            extrinsics = np.array(element['transform_matrix'])

            point_cloud_data = depth_map_to_point_cloud(depth_map, default_intrinsics, extrinsics, additive_factor)
            
            color = np.array(color_palette(i))[:3]
            colors = np.tile(color, (point_cloud_data.shape[0], 1))

            point_cloud = o3d.geometry.PointCloud()
            point_cloud.points = o3d.utility.Vector3dVector(point_cloud_data)
            point_cloud.colors = o3d.utility.Vector3dVector(colors)

            combined_cloud += point_cloud

            
        filtered_points = np.asarray(combined_cloud.points)
      
        # Update combined_cloud with filtered points
        combined_cloud.points = o3d.utility.Vector3dVector(filtered_points)
       
        # Save the combined cloud
        output_file = config['output_dir'] + f"combined_cloud_{additive_factor}.npy"
        np.save(output_file, np.asarray(combined_cloud.points))

    return

def calculate_average_points(point_cloud, config):

    # if points are numpy.ndarray
    if isinstance(point_cloud, np.ndarray):
        points = point_cloud
    else:
        # Extract points from the point cloud
        points = np.asarray(point_cloud.points)

    if('avg_min_depth' in config):
        z1 = config['avg_min_depth']
        z2 = config['avg_max_depth']
    else:
        z1 = np.min(points[:, 2])
        z2 = np.max(points[:, 2])
    m = (z2 - z1) / config['num_slices']

    average_points = []
    
    # Iterate through each slice
    z = z1
    while z < z2:
        # Filter points in the current slice
        slice_points = points[(points[:, 2] >= z) & (points[:, 2] < z + m)]
        
        # Calculate the average point if the slice is not empty
        if len(slice_points) > 0:
            avg_point = np.mean(slice_points, axis=0)
            average_points.append(avg_point)

        z += m

    if('output_dir' in config):
        # Save the average points to a file
        average_points_file = config['output_dir'] + 'average_points.npy'
        np.save(average_points_file, np.array(average_points))

    return average_points

def process_and_save_exr_files(input_directory, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for filename in os.listdir(input_directory):
        if filename.endswith(".exr"):
            # Load the .exr file
            depth_map_path = os.path.join(input_directory, filename)
            depth_map = cv2.imread(depth_map_path, cv2.IMREAD_ANYDEPTH)

            # Mask out areas that are too far away
            depth_map[depth_map > 10] = 0.0

            # Save as .npy file
            npy_path = os.path.join(output_directory, filename.replace('.exr', '.npy'))
            np.save(npy_path, depth_map)

            # Normalize and apply colormap for .png file
            normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
            depth_map_8bit = np.uint8(normalized_depth_map)
            colored_depth_map = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_JET)

            # Save as .png file
            png_path = os.path.join(output_directory, filename.replace('.exr', '.png'))
            cv2.imwrite(png_path, colored_depth_map)

            print(f"Processed and saved: {filename}")
