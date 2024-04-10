import os
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import time
from matplotlib.ticker import ScalarFormatter
from GPIS_utils import plot_pc_data, plot_0_level_set, plot_SDF_slices, plot_rays, focal_length_to_fov
from GPIS_renderer import GPIS_renderer
from GPIS import GPISModel
import random
import json
from blender_utils import process_blender_images, calculate_average_points
import argparse


parser = argparse.ArgumentParser(description='Specify the location of the params file.')
parser.add_argument('params_file', type=str, help='The location of the params file.')
args = parser.parse_args()

with open(args.params_file) as f:
    params = json.load(f)

proc_params = params['proc_params']
data_loader_params = params['data_loader_params']
gpis_params = params['gpis_params']
ray_march_params = params['ray_march_params']
camera_params = params['camera_params']

pc_directory = data_loader_params['pc_directory']
points = np.load(pc_directory + 'combined_cloud.npy')
normals = np.load(pc_directory + 'combined_cloud_normals.npy')
normals_reverse = np.load(pc_directory + 'combined_cloud_normals_reverse.npy')

downsample = proc_params['downsample']
points = points[::downsample]
normals = normals[::1]
normals_reverse = normals_reverse[::1]

x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

X = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
Y = np.zeros((np.shape(X)[0], 1)) - 1

X_negative = np.hstack((normals[:, 0].reshape(-1,1), normals[:, 1].reshape(-1,1), normals[:, 2].reshape(-1,1)))
Y_negative = np.zeros((np.shape(X_negative)[0], 1)) + proc_params['additive_factors_applied'][1] - 1

X_positive = np.hstack((normals_reverse[:, 0].reshape(-1,1), normals_reverse[:, 1].reshape(-1,1), normals_reverse[:, 2].reshape(-1,1)))
Y_positive = np.zeros((np.shape(X_positive)[0], 1)) + proc_params['additive_factors_applied'][0]  - 1


config = {'num_slices': proc_params['slices'],'avg_min_depth': proc_params['min_depth'],'avg_max_depth': proc_params['max_depth']}
centers = calculate_average_points(points, config)

if(proc_params['use_average']):
    centers = np.array(centers)

    X_center = np.hstack((centers[:, 0].reshape(-1,1), centers[:, 1].reshape(-1,1), centers[:, 2].reshape(-1,1)))
    Y_center = np.zeros((np.shape(X_center)[0], 1)) + proc_params['average_factor_applied'] - 1

    X = np.vstack((X,X_negative, X_positive, X_center))
    Y = np.vstack((Y, Y_negative, Y_positive, Y_center))
else:
    X = np.vstack((X,X_negative, X_positive))
    Y = np.vstack((Y, Y_negative, Y_positive))


model = GPISModel(X, Y, gpis_params)
model.train()


x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])
z_min = np.min(X[:, 2])
z_max = np.max(X[:, 2])

# plot_0_level_set([x_min, x_max, y_min, y_max, z_min, z_max], 50, model, threshold = .005, show=False, save= False, path = 'output/0-level/')
# plot_SDF_slices([.42], [x_min, x_max, y_min, y_max], 50, model, show=False, save=False, path = 'output/contour/')

# plt.show()

fov_x, fov_y = focal_length_to_fov(camera_params['focal_length_x'], camera_params['focal_length_y'], camera_params['width'], camera_params['height'])

center_y = camera_params['height'] - camera_params['cy']
center_x = camera_params['width'] - camera_params['cx']

object_center = np.mean(points, axis = 0)
object_radius = np.max(np.linalg.norm(points - object_center, axis = 1))

render_params = {'transforms_path': data_loader_params['camera_pose_directory']+'transforms.json', 
        'cam_params': {'fovx': fov_x, 'fovy': fov_y, 'width': camera_params['width'], 'height': camera_params['height'], 'cx': center_x, 'cy': center_y}, 
        'ray_march_params': ray_march_params, 
        'object_center': object_center, 'object_radius': object_radius}

renderer = GPIS_renderer(model, render_params)


rays, start_positions = renderer.generate_rays(renderer.transforms[0], object_center, object_radius)


with open(render_params['transforms_path']) as f:
    data = json.load(f)

depth_maps, var_maps = renderer.render(data)