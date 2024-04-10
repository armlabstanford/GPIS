import os
import math
import torch
import gpytorch
from matplotlib import pyplot as plt
import numpy as np
import time
from matplotlib.ticker import ScalarFormatter
from GPIS_utils import plot_pc_data, plot_0_level_set, plot_SDF_slices, plot_rays
from GPIS_renderer import GPIS_renderer
from GPIS import GPISModel
import random
import json
from blender_utils import process_blender_images, calculate_average_points
import argparse

parser = argparse.ArgumentParser(description='Specify the location of the params file.')
parser.add_argument('params_file', type=str, help='The location of the params file.')
args = parser.parse_args()


# load the params
with open(args.params_file) as f:
    params = json.load(f)

proc_config = params['proc_params']
gpis_params = params['gpis_params']
ray_march_params = params['ray_march_params']
camera_params = params['camera_params']
dataloader_params = params['dataloader_params']

process_blender_images(proc_config)

dir = proc_config['output_dir']
points = np.load(dir + 'combined_cloud_'+str(proc_config['additive_factors'][0])+'.npy')
normals = np.load(dir + 'combined_cloud_'+str(proc_config['additive_factors'][1])+'.npy')
normals_reverse = np.load(dir + 'combined_cloud_'+str(proc_config['additive_factors'][2])+'.npy')
centers = np.load(dir + 'average_points.npy')


# downsample the data
downsample = proc_config['downsample']
grad_ratio = proc_config['grad_ratio']
points = points[::downsample]
normals = normals[::downsample*grad_ratio]
normals_reverse = normals_reverse[::downsample*grad_ratio]


# prepare data to condition GPIS
x = points[:, 0]
y = points[:, 1]
z = points[:, 2]

X = np.hstack((x.reshape(-1,1), y.reshape(-1,1), z.reshape(-1,1)))
Y = np.zeros((np.shape(X)[0], 1)) - 1


X_negative = np.hstack((normals[:, 0].reshape(-1,1), normals[:, 1].reshape(-1,1), normals[:, 2].reshape(-1,1)))
Y_negative = np.zeros((np.shape(X_negative)[0], 1)) + proc_config['additive_factors_applied'][1] - 1

X_positive = np.hstack((normals_reverse[:, 0].reshape(-1,1), normals_reverse[:, 1].reshape(-1,1), normals_reverse[:, 2].reshape(-1,1)))
Y_positive = np.zeros((np.shape(X_positive)[0], 1)) + proc_config['additive_factors_applied'][0]  - 1

X_center = np.hstack((centers[:, 0].reshape(-1,1), centers[:, 1].reshape(-1,1), centers[:, 2].reshape(-1,1)))
Y_center = np.zeros((np.shape(X_center)[0], 1)) + proc_config['average_factor_applied'] - 1

X = np.vstack((X, X_positive, X_center))
Y = np.vstack((Y, Y_positive, Y_center))


model = GPISModel(X, Y, gpis_params)


# train the model
model.train()

x_min = np.min(X[:, 0])
x_max = np.max(X[:, 0])
y_min = np.min(X[:, 1])
y_max = np.max(X[:, 1])
z_min = np.min(X[:, 2])
z_max = np.max(X[:, 2])


plot_0_level_set([x_min, x_max, y_min, y_max, z_min, z_max], 50, model, threshold = .005, show=False, save= False, path = 'output/0-level/')
plot_SDF_slices([.78], [x_min, x_max, y_min, y_max], 50, model, show=False, save=False, path = 'output/contour/')
plt.show()

object_center = np.mean(points, axis = 0)
object_radius = np.max(np.linalg.norm(points - object_center, axis = 1))

# now render a depth map

params = {'transforms_path': dataloader_params['camera_pose_directory']+'transforms_train.json', 
        'cam_params': camera_params, 
        'ray_march_params': ray_march_params, 
        'object_center': object_center, 'object_radius': object_radius}

renderer = GPIS_renderer(model, params)

rays, start_positions = renderer.generate_rays(renderer.transforms[0], object_center, object_radius)

with open(params['transforms_path']) as f:
    data = json.load(f)

depth_maps, var_maps = renderer.render(data)