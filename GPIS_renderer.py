import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt
import random
import string
import json
import cv2
import imageio
import os
import time
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"


class GPIS_renderer():
    def __init__(self, model, params):
      
        self.device = model.get_device()
        self.model = model

        self.transforms_path = params.get('transforms_path', None)
        self.ray_march_params = params.get('ray_march_params', self.default_march_params())
        self.depth_start = params.get('depth_start', 0.2)
        
        self.object_center = params.get('object_center', np.array((0, 0, 0)))
        self.object_radius = params.get('object_radius', 1.0)

        self.cam_params, self.transforms = self.load_transforms(self.transforms_path)

        if('cam_params' in params):
            self.cam_params = params['cam_params']

        self.use_variable_step = self.ray_march_params.get('use_variable_step', False)

    def default_march_params(self):

        return {
            'step_size': 0.001,
            'max_steps': 120,
            'threshold': 0.05
        }

    def default_cam_params(self, cam_type):

        if(cam_type == 'Pinhole'):
            return {
                'fovx': 90,
                'fovy': 90,
                'near': 0.1,
                'far': 1000,
                'width': 1800,
                'height': 1800
            }
        else:
            raise NotImplementedError
    
    def load_transforms(self, transforms_path):


        def rotx(degrees):
            radians = np.deg2rad(degrees)

            rotation_matrix = np.array([
                [1, 0, 0],
                [0, np.cos(radians), -np.sin(radians)],
                [0, np.sin(radians), np.cos(radians)]
            ])

            return rotation_matrix

        if(transforms_path is None):
            transforms = []
            for i in range(1):
                position = np.array([.35,-.025,.60])
                orientation = np.array([[1,0,0],[0,1,0],[0,0,1]])
                transforms.append((position, orientation))

            cam_params = self.default_cam_params('Pinhole')
        else:
            # load data from transforms.json
            with open(transforms_path) as f:
                transforms = json.load(f)

            total_frames = len(transforms['frames'])

            transforms_out = []
            for i, element in enumerate(transforms['frames']):

                transform_matrix = np.array(element['transform_matrix'])

                # Extract the rotation matrix
                rotation_matrix = transform_matrix[:3, :3]

                position_vector = transform_matrix[:3, 3]

                transforms_out.append((position_vector, rotation_matrix))

            # if cam_params exists in the json file, use those values
            cam_params = self.default_cam_params('Pinhole')

            return cam_params, transforms_out



        return cam_params, transforms

    def find_center_of_object(self, range, num_points, threshold=.1):

            x = np.linspace(range[0], range[1], num_points)
            y = np.linspace(range[2], range[3], num_points)
            z = np.linspace(range[4], range[5], num_points)

            X, Y, Z = np.meshgrid(x, y, z)
            points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

            y_pred, _ = self.model.infer(points)

            y_pred = y_pred.cpu().numpy()
            y_pred = y_pred.reshape(X.shape)

            zero_level_set_points = np.argwhere(np.abs(y_pred) < threshold)

            if zero_level_set_points.size == 0:
                print("No points found in the 0-level set. Cannot determine the center.")
                return None

            x_s = x[zero_level_set_points[:, 0]]
            y_s = y[zero_level_set_points[:, 1]]
            z_s = z[zero_level_set_points[:, 2]]

            center = np.array([np.mean(x_s), np.mean(y_s), np.mean(z_s)])
            return center

   
    def generate_rays(self, transform, object_center, radius):

        position, orientation = transform
        cam_params = self.cam_params

        cx = self.cam_params['cx']
        cy = self.cam_params['cy']

        focal_length_x = self.fov_to_focal_length(cam_params['fovx'], cam_params['width'])
        focal_length_y = self.fov_to_focal_length(cam_params['fovy'], cam_params['height'])
    
        x_pixel, y_pixel = np.meshgrid(np.arange(cam_params['width']) - cx,
                                    np.arange(cam_params['height']) - cy)
        
        direction = np.stack((x_pixel / focal_length_x, y_pixel / focal_length_y, -np.ones_like(x_pixel)), axis=-1)

        direction = np.einsum('ij,klj->kli', orientation, direction)
        direction /= np.linalg.norm(direction, axis=2, keepdims=True)

        rays = np.stack((np.repeat(position[np.newaxis, :], cam_params['width'] * cam_params['height'], axis=0), direction.reshape(-1, 3)), axis=1)

        start_positions = np.array([self.calculate_ray_sphere_intersection(ray_origin, ray_dir, object_center, radius)
                                    for ray_origin, ray_dir in rays])

        return rays, start_positions



    @staticmethod
    def calculate_ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    
        oc = sphere_center - ray_origin
        
        a = np.dot(ray_direction, ray_direction)
        b = -2.0 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - sphere_radius ** 2

        discriminant = b ** 2 - 4 * a * c

        if discriminant > 0:
            # Two intersections, take the closest positive one
            t1 = (-b - np.sqrt(discriminant)) / (2 * a)
            t2 = (-b + np.sqrt(discriminant)) / (2 * a)
            t = min(t1, t2)
            if t > 0:
                return ray_origin + t * ray_direction

        return ray_origin

    @staticmethod
    def fov_to_focal_length(fov, image_size):

        return image_size / (2 * np.tan(np.radians(fov) / 2))

    def ray_march(self, transform):


        object_center = self.object_center
        radius = self.object_radius
        total_calls = 0

        periodic_calls = 0
        start_time = time.time()
        last_update_time = start_time 

        # Generate rays and starting positions
        rays, start_positions = self.generate_rays(transform, object_center, radius)

        max_steps = self.ray_march_params['max_steps']
        threshold = self.ray_march_params['threshold']

        variable_threshold = self.ray_march_params['variable_threshold']
        variable_slope = self.ray_march_params['variable_slope']
        variable_min_step = self.ray_march_params['variable_min_step']
        
 
        max_points_per_batch = 160 ** 2 


        num_rays = len(rays)
        depth_map = np.full((num_rays, 1), np.nan)  # Initialize depth map with NaN
        var_map = np.full((num_rays, 1), np.nan)  # Initialize variance map with NaN

        depths = np.zeros(num_rays)  # Initialize array to hold current depths for all rays

        depths = np.linalg.norm(start_positions - rays[:, 0], axis=1) # init the depth of each ray as the distance from the origin

        points = start_positions
        directions = np.array([ray[1] for ray in rays])  # All directions
        active_mask = np.ones(num_rays, dtype=bool)  # Initialize active mask for all rays

        for i in range(max_steps):
            # Compute distances only for active rays
            active_points = points[active_mask]
            active_directions = directions[active_mask]

            if len(active_points) == 0:
                break  # Exit if no active rays left

            # Filter points based on distance to object center
            distances_to_center = np.linalg.norm(active_points - object_center, axis=1)
            within_radius_mask = distances_to_center < radius + threshold 

            # Infer only for points within the radius
            points_within_radius = active_points[within_radius_mask]
            directions_within_radius = active_directions[within_radius_mask]

            if len(points_within_radius) > 0:

                if len(points_within_radius) > max_points_per_batch:
                    all_distances = []
                    all_variances = []
                    # Split points into batches
                    for batch_start in range(0, len(points_within_radius), max_points_per_batch):
                        batch_end = batch_start + max_points_per_batch
                        batch_points = points_within_radius[batch_start:batch_end]
                        distances_batch, var_batch = self.model.infer(batch_points)
                        all_distances.append(distances_batch)
                        all_variances.append(var_batch)

                    distances = torch.cat(all_distances)
                    var = torch.cat(all_variances)
                else:
                    distances, var = self.model.infer(points_within_radius)

                total_calls += len(points_within_radius)
                periodic_calls += len(points_within_radius)
                distances = distances.cpu().numpy()
                var = var.cpu().numpy()

                # Calculate variable step sizes
                if(self.use_variable_step):
                    step_sizes = variable_slope * np.abs(distances) 
                    step_sizes = np.maximum(step_sizes, variable_min_step)
                else:
                    step_sizes = np.full(len(distances), self.ray_march_params['step_size'])

                # Update points and depths for active rays within radius
                active_indices_within_radius = np.where(active_mask)[0][within_radius_mask]
                points[active_indices_within_radius] += step_sizes[:, None] * directions_within_radius
                depths[active_indices_within_radius] += step_sizes

                if(self.use_variable_step):
                    hit_mask = np.abs(distances) < variable_threshold
                else:
                    hit_mask = distances < threshold

                
                depth_map[active_indices_within_radius[hit_mask]] = depths[active_indices_within_radius[hit_mask]].reshape(-1, 1)
                var_map[active_indices_within_radius[hit_mask]] = var[hit_mask].reshape(-1, 1)

                # Update active mask
                active_mask[active_indices_within_radius[hit_mask]] = False

            # Early exit if all rays have hit or gone beyond the threshold
            if not active_mask.any():
                break

            # Periodic performance update
            current_time = time.time()
            if current_time - last_update_time >= 1:  # Check if 10 seconds have elapsed
                elapsed_time = current_time - last_update_time
                if elapsed_time > 0:
                    avg_inferences_per_second = periodic_calls / elapsed_time
                    print(f'\rAverage inferences per second (last 1s): {avg_inferences_per_second:.2f}', end='')
                    last_update_time = current_time  # Reset last update time
                    periodic_calls = 0  # Reset total calls for the next period


        # Reshape depth map and variance map to the image dimensions
        depth_map = depth_map.reshape(self.cam_params['height'], self.cam_params['width'])
        var_map = var_map.reshape(self.cam_params['height'], self.cam_params['width'])

        return depth_map, var_map

    def render(self, json_data):

        depth_maps = []
        var_maps = []
        for i, transform in enumerate(self.transforms):

            start = time.time()

            depth_map, var_map = self.ray_march(transform)

            depth_map = np.flip(depth_map, axis=0)
            var_map = np.flip(var_map, axis=0)

            file_number = json_data['frames'][i]['file_path'].split('/')[1].split('.')[0]

            depth_map_path = 'output/depth/Image' + file_number + '.npy'
            self.save_depth_EXR(depth_map, path = depth_map_path)

            depth_map_path = 'output/depth/Image' + file_number + '.png'
            self.save_depth_with_colormap(depth_map, path = depth_map_path)

            depth_map_path = 'output/var/Image' + file_number + '.npy'
            self.save_depth_EXR(var_map, path = depth_map_path)

            depth_map_path = 'output/var/Image' + file_number + '.png'
            self.save_depth_with_colormap(var_map, path = depth_map_path)

            end = time.time()
            print(f'time elapsed: {end - start}')


        return depth_maps, var_maps
    
    def save_depth_fig(self, depth_map, path = ''):

        plt.figure(figsize=(25, 25))
        plt.imshow(depth_map)
        plt.colorbar()

        plt.xlabel('X')
        plt.ylabel('Y')


        cam_params_str = f'fovxy{self.cam_params["fovx"]}_{self.cam_params["fovy"]}_w{self.cam_params["width"]}_h{self.cam_params["height"]}'
        ray_march_params_str = f'ss{self.ray_march_params["step_size"]}_maxs{self.ray_march_params["max_steps"]}_thres{self.ray_march_params["threshold"]}'
        gpis_params_str = f'ls{self.model.lengthscale_range}_nu{self.model.nu}_os{self.model.outputscale_range}_mean{self.model.constant_mean_range}_opt{self.model.optimizer_steps}_presub{self.model.pre_sub}'
        plt.title(f'C:{cam_params_str}\nRay{ray_march_params_str}\n GPIS {gpis_params_str}')

        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        plt.savefig(path + name + '.png')

        with open(path + 'depth_map_names.txt', 'a') as f:
            f.write(f'{name} : {path + cam_params_str + ray_march_params_str + gpis_params_str}.png \n')


    def save_depth_EXR(self, depth_map, path = ''):

        np.save(path, depth_map)

    def save_depth(self, depth_map, path = ''):

        cv2.imwrite(path, depth_map)

    def save_depth_with_colormap(self, depth_map, path=''):

        # Normalize the depth map to be in the range 0-255
        normalized_depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)

        # Convert to 8-bit image
        depth_map_8bit = np.uint8(normalized_depth_map)

        # Apply the 'jet' colormap
        colored_depth_map = cv2.applyColorMap(depth_map_8bit, cv2.COLORMAP_JET)

        # Save the colored depth map
        cv2.imwrite(path, colored_depth_map)
