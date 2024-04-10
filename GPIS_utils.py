import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import time
import random
import string


def plot_pc_data(points, normals = None, neg_normals = None, show = True):
    fig = plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 22})
    ax = fig.add_subplot(111, projection='3d')
    color = '#782028'
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c = color)

    if(normals is not None):
        ax.scatter(normals[:, 0],normals[:, 1], normals[:, 2], c='red')
        if(neg_normals is not None):
            ax.scatter(neg_normals[:, 0],neg_normals[:, 1], neg_normals[:, 2], c='blue')

    # add labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Point Cloud of DT data')

    if(show):
        plt.show()

def focal_length_to_fov(focal_length_x, focal_length_y, image_width, image_height):
   
    fov_x_rad = 2 * np.arctan(image_width / (2 * focal_length_x))
    fov_y_rad = 2 * np.arctan(image_height / (2 * focal_length_y))
    
    fov_x = np.degrees(fov_x_rad)
    fov_y = np.degrees(fov_y_rad)
    
    return fov_x, fov_y


def plot_rays(rays, start_positions, length, pc_data=None, number=500, show=True):
   
    fig = plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 22})
    ax = fig.add_subplot(111, projection='3d')

    indices = np.random.choice(range(np.shape(rays)[0]), number)
    for i in indices:
        origin, direction = rays[i]
        start_position = start_positions[i]

        direction = direction / np.linalg.norm(direction) * length

        end_point = start_position + direction

        ax.plot([start_position[0], end_point[0]],
                [start_position[1], end_point[1]],
                [start_position[2], end_point[2]], 'b-')

    if pc_data is not None:
        ax.scatter(pc_data[:, 0], pc_data[:, 1], pc_data[:, 2], c=pc_data[:, 2], cmap='jet')

    if pc_data is not None:
        max_range = np.array([pc_data[:, 0].max() - pc_data[:, 0].min(), 
                              pc_data[:, 1].max() - pc_data[:, 1].min(), 
                              pc_data[:, 2].max() - pc_data[:, 2].min()]).max() / 2.0
        mid_x = (pc_data[:, 0].max() + pc_data[:, 0].min()) * 0.5
        mid_y = (pc_data[:, 1].max() + pc_data[:, 1].min()) * 0.5
        mid_z = (pc_data[:, 2].max() + pc_data[:, 2].min()) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

    ax.set_box_aspect([1, 1, 1]) 
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('Rays')

    if show:
        plt.show()


def plot_0_level_set(bounds, num_points, model, threshold=.1, show=True, save=False, path='', params_str=None):
    x = np.linspace(bounds[0], bounds[1], num_points)
    y = np.linspace(bounds[2], bounds[3], num_points)
    z = np.linspace(bounds[4], bounds[5], num_points)

    batch_size = 30
    batched_points = []

    # Split the grid into smaller batches
    for i in range(0, num_points, batch_size):
        for j in range(0, num_points, batch_size):
            for k in range(0, num_points, batch_size):
                X_batch, Y_batch, Z_batch = np.meshgrid(x[i:i + batch_size], y[j:j + batch_size], z[k:k + batch_size])
                batched_points.append(np.column_stack([X_batch.ravel(), Y_batch.ravel(), Z_batch.ravel()]))

    # Initialize arrays for predictions and uncertainties
    y_pred_full = np.empty((num_points, num_points, num_points))
    sigma_full = np.empty((num_points, num_points, num_points))

    # Process each batch
    for batch in batched_points:
        start_time = time.time()
        y_pred, sigma = model.infer(batch)

        y_pred = y_pred.cpu().numpy().reshape(-1)
        sigma = sigma.cpu().numpy().reshape(-1)

        # Reconstruct the full grid
        indices = np.round((batch - [bounds[0], bounds[2], bounds[4]]) / [(bounds[1]-bounds[0])/(num_points-1), (bounds[3]-bounds[2])/(num_points-1), (bounds[5]-bounds[4])/(num_points-1)]).astype(int)
        for idx, (yp, sig) in enumerate(zip(y_pred, sigma)):
            y_pred_full[indices[idx][0], indices[idx][1], indices[idx][2]] = yp
            sigma_full[indices[idx][0], indices[idx][1], indices[idx][2]] = sig

    points = np.argwhere(np.abs(y_pred_full) < threshold)

    # Extract the coordinates
    x_s, y_s, z_s = x[points[:, 0]], y[points[:, 1]], z[points[:, 2]]

    sigma = sigma_full[points[:, 0], points[:, 1], points[:, 2]]

    # Plotting
    fig = plt.figure(figsize=(15, 15))
    plt.rcParams.update({'font.size': 15})
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(x_s, y_s, z_s, c=sigma, cmap='hot')
    cbar = plt.colorbar(scatter)
    cbar.set_label('uncertainty')

    ax.set_box_aspect([1, 1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title('0 level set of GPIS model')


    if show:
        plt.show()

    if save:
        gpis_params_str = f'ls{model.lengthscale_range}_nu{model.nu}_os{model.outputscale_range}_mean{model.constant_mean_range}_opt{model.optimizer_steps}_presub{model.pre_sub}'
        plt.title(f'GPIS {gpis_params_str}')

        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        plt.savefig(path + name + '.png')

        with open(path + 'names.txt', 'a') as f:
            f.write(f'{name} : {path + gpis_params_str}.png \n')

        if params_str is not None:
            with open(path + '0-level-params.txt', 'a') as f:
                f.write(f'{name} : {params_str} \n')

        # Define the PLY file path
        ply_file_path = path + 'points_for_blender.ply'

        # center the points
        x_s = x_s - np.mean(x_s)
        y_s = y_s - np.mean(y_s)
        z_s = z_s - np.mean(z_s)
        # Save the points as a PLY file
        save_points_as_ply(x_s, y_s, z_s, sigma, ply_file_path)
        print(f'Points saved as PLY to {ply_file_path}')

def save_points_as_ply(x_s, y_s, z_s, sigma, path):

    sigma_normalized = (sigma - sigma.min()) / (sigma.max() - sigma.min())
    colors = plt.cm.hot(sigma_normalized)  # Use matplotlib's colormap

    with open(path, 'w') as ply_file:
        ply_file.write("ply\n")
        ply_file.write("format ascii 1.0\n")
        ply_file.write(f"element vertex {len(x_s)}\n")
        ply_file.write("property float x\n")
        ply_file.write("property float y\n")
        ply_file.write("property float z\n")
        ply_file.write("property uchar red\n")
        ply_file.write("property uchar green\n")
        ply_file.write("property uchar blue\n")
        ply_file.write("end_header\n")
        
        # Write vertex data
        for x, y, z, color in zip(x_s, y_s, z_s, colors):
            r, g, b = (color[:3] * 255).astype(int)
            ply_file.write(f"{x} {y} {z} {r} {g} {b}\n")

        print(f'Points written to {path}')

def plot_SDF_slices(z_values, range, num_points, model, show=True, save = False, path = '', params_str = None):

    x_min, x_max, y_min, y_max = range
    x = np.linspace(x_min, x_max, num_points)
    y = np.linspace(y_min, y_max, num_points)

    # Set up the figure and subplots
    num_subplots = len(z_values)
    fig, axes = plt.subplots(1, num_subplots, figsize=(15 * num_subplots, 15))
    plt.rcParams.update({'font.size': 30})

    for i, z in enumerate(z_values):
        # Create a meshgrid for the current z slice
        X, Y = np.meshgrid(x, y)
        Z = np.full(X.shape, z)
        points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

        # Perform inference
        y_pred, sigma = model.infer(points)

        # move the predictions to the cpu
        y_pred = y_pred.cpu().numpy()
        sigma = sigma.cpu().numpy()

        # Reshape the predictions and sigma to match the grid
        y_pred = y_pred.reshape(X.shape)
        sigma = sigma.reshape(X.shape)

        # Plot the current slice
        ax = axes[i] if num_subplots > 1 else axes

        contour = ax.contourf(X, Y, y_pred, levels=50, cmap='RdGy', vmin = -np.max(y_pred), vmax = np.max(y_pred))

        cbar = plt.colorbar(contour, ax=ax)
        cbar.set_label('Signed Distance')
        ax.set_title(f'Distance')
        ax.set_xlabel('X', fontsize = 30)
        ax.set_ylabel('Y', fontsize = 30)
        ax.tick_params(axis='both', which='major', labelsize=30)

        # set axis equal
        ax.set_box_aspect(1.0)

    plt.tight_layout()
    if show:
        plt.show()

    if(save):
        gpis_params_str = f'ls{model.lengthscale_range}_nu{model.nu}_os{model.outputscale_range}_mean{model.constant_mean_range}_opt{model.optimizer_steps}_presub{model.pre_sub}'

        name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=8))
        plt.savefig(path + name + '.png')

        if params_str is not None:
            with open(path + '0-level-params.txt', 'a') as f:
                f.write(f'{name} : {params_str} \n')

        