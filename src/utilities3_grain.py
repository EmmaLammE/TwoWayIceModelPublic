import torch
import numpy as np
import scipy.io
import h5py
import torch.nn as nn
import torch.nn.functional as F
import math
import operator
from functools import reduce

import matplotlib.image as mpimg
from scipy.ndimage import label
from scipy import ndimage
from scipy.spatial import distance
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from skimage.morphology import skeletonize
from skimage.morphology import disk
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as ticker

def log_normalize(data, max_ref, min_ref, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        log_data = torch.log10(data)  # Use torch.log10 for logarithmic scaling
        max_ref, min_ref = np.log10(max_ref), np.log10(min_ref)
        print(f'Using max_ref={max_ref} min_ref={min_ref} to normalize data')
        # min_val, max_val = torch.min(log_data), torch.max(log_data)  # Torch min/max functions
        scaled = (log_data - min_ref) / (max_ref - min_ref)  # Normalize between 0 and 1
        # Scale to the desired feature range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def log_normalize_np(data, max_ref, min_ref, feature_range=(-1, 1)):
    """
    data: NumPy array
    Apply log10 scaling and then min/max normalization to feature_range.
    """
    data = np.log10(data)  # log10 scale
    min_log = np.log10(min_ref)
    max_log = np.log10(max_ref)
    # scale to [0..1]
    data = (data - min_log) / (max_log - min_log)
    # now scale to (feature_range)
    data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return data

def maxmin_normalize(data, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        min_val, max_val = torch.min(data), torch.max(data)  # Use torch min/max functions
        scaled = (data - min_val) / (max_val - min_val)  # Normalize between 0 and 1
        # Scale to the desired feature range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def reference_normalize(data, max_ref, min_ref, feature_range=(-1, 1)):
    if isinstance(data, torch.Tensor):
        print(f'Using max_ref={max_ref} min_ref={min_ref} to normalize data')
        scaled = (data - min_ref) / (max_ref - min_ref)  # Normalize between 0 and 1
        # Scale to the desired feature range
        return scaled * (feature_range[1] - feature_range[0]) + feature_range[0]
    else:
        raise TypeError("Input data must be a torch.Tensor")

def reference_normalize_np(data, min_ref, max_ref, feature_range=(-1, 1)):
    """
    data: NumPy array
    Apply min/max normalization to feature_range.
    """
    data = (data - min_ref) / (max_ref - min_ref)  # [0..1]
    data = data * (feature_range[1] - feature_range[0]) + feature_range[0]
    return data

def smoothness_loss(predictions):
    # prediction should be in shapes [num batch, nx]
    # Compute temporal derivatives using finite differences
    temporal_diff = predictions[:, 1:] - predictions[:, :-1]
    # Penalize large changes between consecutive time steps
    smoothness_penalty = torch.mean(temporal_diff**2)
    return smoothness_penalty

def smoothness_loss_2d(predictions):
    # prediction should be in shapes [num batch, nx]
    # Compute temporal derivatives using finite differences
    temporal_diff = predictions[:, 1:,:] - predictions[:, :-1,:] 
    # Penalize large changes between consecutive time steps
    smoothness_penalty = torch.mean(temporal_diff**2)
    return smoothness_penalty

def generate_unique_color(existing_colors):
    while True:
        color = tuple(np.random.choice(range(256), size=3))
        existing_colors.add(color)
        return color

def reinitialize(grain_pred):
    # initialize to binary images
    # 1 - inside, 0 - boundary
    grain_pred_plt = torch.clone(grain_pred)
    grain_mean = 0.4*torch.mean(grain_pred_plt)
    # print(grain_mean)
    grain_pred_plt[grain_pred>grain_mean] = 1 # grains
    grain_pred_plt[grain_pred<=grain_mean] = -1 # boundaries
    # grain_pred_plt[0:1,:], grain_pred_plt[-2:-1,:], grain_pred_plt[:,0:1], grain_pred_plt[:,-2:-1] = -1,-1,-1,-1
    
    # set isolated island grains to 1
    grain_pred_plt = reset_small_islands(grain_pred_plt, max_island_size=12)
    grain_pred_plt = uniform_thickness_skeleton(grain_pred_plt, target_thickness=2)
    grain_pred_plt = trim_branches(grain_pred_plt)
    grain_pred_plt = reset_small_islands(grain_pred_plt, max_island_size=3)
    grain_pred_plt = connect_loose_ends(grain_pred_plt,distance_threshold=15)
    grain_pred_plt = draw_line_to_negative_one_group(grain_pred_plt,search_radius=5)
    grain_temp = grain_pred_plt.clone()
    grain_temp[grain_temp != 1] = -1
    grain_pred_plt = uniform_thickness_skeleton(grain_temp, target_thickness=2)# grain_pred_plt = draw_line_to_negative_one_group(grain_pred_plt,search_radius=7)
    # grain_pred_plt = trim_branches(grain_pred_plt)
    # grain_pred_plt = reset_small_islands(grain_pred_plt, max_island_size=3)
    # grain_pred_plt = uniform_thickness_skeleton(grain_pred_plt, target_thickness=2)
    # make sure to set boundaries to -1
    grain_pred_plt[grain_pred_plt != 1] = -1
    # reset_islands(grain_pred_plt, size_limit=20) # not working
    return grain_pred_plt

def reinitialize_test(grain_pred):
    # initialize to binary images
    # 1 - inside, 0 - boundary
    grain_pred_plt = torch.clone(grain_pred)
    grain_mean = 1*torch.mean(grain_pred_plt)
    # print(grain_mean)
    grain_pred_plt[grain_pred>grain_mean] = 1 # grains
    grain_pred_plt[grain_pred<=grain_mean] = -1 # boundaries
    # grain_pred_plt[0:1,:], grain_pred_plt[-2:-1,:], grain_pred_plt[:,0:1], grain_pred_plt[:,-2:-1] = -1,-1,-1,-1
    
    # set isolated island grains to 1
    grain_pred_plt = reset_small_islands(grain_pred_plt, max_island_size=15)
    grain_pred_plt = connect_broken_boundaries_optimized(grain_pred_plt, max_gap=5)
    return grain_pred_plt

def dfs(matrix, i, j, visited):
    if (i < 0 or i >= matrix.shape[0] or
        j < 0 or j >= matrix.shape[1] or
        visited[i][j] or matrix[i][j] != -1):
        return 0
    visited[i][j] = True
    size = 1
    # Check all 8 neighbors
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            if di == 0 and dj == 0:
                continue
            size += dfs(matrix, i + di, j + dj, visited)
    return size

def find_island_size(matrix, start_i, start_j, visited):
    stack = [(start_i, start_j)]
    size = 0
    while stack:
        i, j = stack.pop()
        if (i < 0 or i >= matrix.shape[0] or
            j < 0 or j >= matrix.shape[1] or
            visited[i, j] or matrix[i, j] != -1):
            continue
        visited[i, j] = True
        size += 1
        for di in [-1, 0, 1]:
            for dj in [-1, 0, 1]:
                stack.append((i + di, j + dj))
    return size

def reset_small_islands(matrix, max_island_size=3):
    # Ensure the input is a PyTorch tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix)
    visited_bound = torch.zeros_like(matrix, dtype=torch.bool)
    visited_grain = torch.zeros_like(matrix, dtype=torch.bool)
    result = matrix.clone()
    
    def dfs_bound(x, y):
        stack_bound = [(x, y)]
        island_coords_bound = []
        island_size_bound = 0
        while stack_bound:
            cx, cy = stack_bound.pop()
            if (0 <= cx < matrix.shape[0] and 0 <= cy < matrix.shape[1] and
                (matrix[cx, cy] == -1 or matrix[cx, cy] == 4) and not visited_bound[cx, cy]):
                visited_bound[cx, cy] = True
                island_coords_bound.append((cx, cy))
                island_size_bound += 1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            stack_bound.append((cx + dx, cy + dy))
        return island_coords_bound, island_size_bound
    def dfs_grain(x, y):
        stack_grain = [(x, y)]
        island_coords_grain = []
        island_size_grain = 0
        while stack_grain:
            cx, cy = stack_grain.pop()
            if (0 <= cx < matrix.shape[0] and 0 <= cy < matrix.shape[1] and
                (matrix[cx, cy] == 1 or matrix[cx, cy] == 4) and not visited_grain[cx, cy]):
                visited_grain[cx, cy] = True
                island_coords_grain.append((cx, cy))
                island_size_grain += 1
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx != 0 or dy != 0:
                            stack_grain.append((cx + dx, cy + dy))
        return island_coords_grain, island_size_grain
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == -1 and not visited_bound[i, j]:
                island_coords_bound, island_size_bound = dfs_bound(i, j)
                if island_size_bound <= max_island_size:
                    for x, y in island_coords_bound:
                        result[x, y] = 1
                # Additional check for single-pixel connections
                for x, y in island_coords_bound:
                    sum_neighbor = 0
                    for dx in [-1, 0, 1]:
                        for dy in [-1, 0, 1]:
                            if dx != 0 or dy != 0:
                                nx, ny = x + dx, y + dy
                                # print(f'x {x} y {y}, nx {nx} ny {ny}, matrix[nx, ny] {matrix[nx, ny]}')
                                if (0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and
                                    matrix[nx, ny] == -1):
                                    sum_neighbor += 1
                    if(sum_neighbor>=1 and sum_neighbor<=1):
                        result[x, y] = 1
    
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 1 and not visited_grain[i, j]:
                island_coords_grain, island_size_grain = dfs_grain(i, j)
                if island_size_grain <= max_island_size//2:
                    for x, y in island_coords_grain:
                        result[x, y] = -1
    #             # Additional check for single-pixel connections
    #             for x, y in island_coords_grain:
    #                 sum_neighbor = 0
    #                 for dx in [-1, 0, 1]:
    #                     for dy in [-1, 0, 1]:
    #                         if dx != 0 or dy != 0:
    #                             nx, ny = x + dx, y + dy
    #                             # print(f'x {x} y {y}, nx {nx} ny {ny}, matrix[nx, ny] {matrix[nx, ny]}')
    #                             if (0 <= nx < matrix.shape[0] and 0 <= ny < matrix.shape[1] and
    #                                 matrix[nx, ny] == 1):
    #                                 sum_neighbor += 1
    #                 if(sum_neighbor>=1 and sum_neighbor<=1):
    #                     result[x, y] = -1
    return result

def connect_loose_ends(matrix, distance_threshold=20):
    result = matrix.clone()
    # Find loose ends
    loose_ends,result = find_loose_ends(result)
    # remove loose ends that are very close to each other, i.e. at the same end
    loose_ends,result = post_process_loose_ends(loose_ends,result, distance_threshold=2)
    loose_ends_ray = loose_ends.copy()

    # ####### debug #######
    # result = matrix.copy()
    # print("num of loose ends: ",len(loose_ends))
    # for x, y in loose_ends:
    #     result[x,y] = 3
    # print(np.unique(result))
    # ####### debug #######
    
    # set it to higher number to avoid connecting for actually good image
    if len(loose_ends) < 30: 
        return matrix  # No connections to make
    print(f'connecting {len(loose_ends)} loose ends ')
    
    # Loop through all loose ends
    # for i, loose_end in enumerate(loose_ends):
    #     # Calculate distances between this loose end and all other loose ends
    #     distances = [distance.euclidean(loose_end, other) for other in loose_ends]
    #     # Set the distance to itself to infinity to avoid connecting to itself
    #     distances[i] = float('inf')
    #     # If there are no other loose ends to compare, continue to next loose end
    #     if not distances:
    #         continue
    #     closest_index = torch.argmin(torch.tensor(distances)) 
    #     # Check if the closest loose end is within the threshold distance
    #     if distances[closest_index] <= distance_threshold:
    #         # Connect the loose ends by filling in -1 values between them
    #         closest_end = loose_ends[closest_index]
    #         rr, cc = torch.linspace(loose_end[0], closest_end[0], steps=100).to(torch.int32), \
    #                  torch.linspace(loose_end[1], closest_end[1], steps=100).to(torch.int32)
    #         draw_wide_line(result, rr, cc, width=1)

    # print("loose_ends: ",loose_ends)
    # Loop through all loose ends
    # Initialize an index to track loose ends
    i = 0
    # Loop through all loose ends
    while i < len(loose_ends):
        loose_end = loose_ends[i]
        # Calculate distances between this loose end and all other loose ends
        distances = [distance.euclidean(loose_end, other) for other in loose_ends]
        # Set the distance to itself to infinity to avoid connecting to itself
        distances[i] = float('inf')
        # If there are no other loose ends to compare, continue to next loose end
        if not distances:
            i += 1
            continue
        closest_index = torch.argmin(torch.tensor(distances)) 
        # Check if the closest loose end is within the threshold distance
        if distances[closest_index] <= distance_threshold:
            # Connect the loose ends by filling in -1 values between them
            closest_end = loose_ends[closest_index]
            rr, cc = torch.linspace(loose_end[0], closest_end[0], steps=100).to(torch.int32), \
                    torch.linspace(loose_end[1], closest_end[1], steps=100).to(torch.int32)
            draw_wide_line(result, rr, cc, width=1)
            # Remove both loose ends (current loose_end and the closest loose_end)
            if closest_index > i:
                # First remove the closest end, then remove the current loose end
                loose_ends.pop(closest_index)
                loose_ends.pop(i)  # After removing closest_end, i is still valid
            else:
                # First remove the current loose end, then remove the closest end
                loose_ends.pop(i)
                loose_ends.pop(closest_index)  # closest_index is now valid after pop(i)
            
            # No need to increment `i` here because we just popped `i` from the list.
        else:
            # Move to the next loose end if no connection is made
            i += 1
    # print("loose_ends: ",loose_ends)
    return result

def find_loose_ends(matrix):
    # Convert matrix to binary where -1 is True (line) and 1 is False (background)
    binary = (matrix == -1)
    
    # Pad the matrix with False to handle edge cases
    # padded = np.pad(binary, pad_width=1, mode='constant', constant_values=True)
    # pad is in the form of (padding_left, padding_right, padding_top, padding_bottom)
    padded = F.pad(binary, pad=(1, 1, 1, 1), mode='constant', value=True)
    # print(matrix.shape,padded.shape)
    # print(f'padded shape: {padded.shape}')
    loose_ends = []
    pix_thresh,pix_thresh_low = 2,2
    # up down loose ends
    for i in range(1, padded.shape[0] - 3):
        for j in range(1, padded.shape[1] - 3):
            if padded[i, j]:
                neighbors_up = [
                     padded[i-2, j-2],padded[i-2, j-1],padded[i-2, j],padded[i-2, j+1],padded[i-2, j+2],
                     padded[i-1, j-2],padded[i-1, j-1],padded[i-1, j],padded[i-1, j+1],padded[i-1, j+2],
                                      # padded[i,   j-1],               padded[i,   j+1],
                    
                ]
                neighbors_down = [
                                     # padded[i,   j-1],               padded[i,   j+1],
                    padded[i+1, j-2],padded[i+1, j-1],padded[i+1, j],padded[i+1, j+1],padded[i+1, j+2],
                    padded[i+2, j-2],padded[i+2, j-1],padded[i+2, j],padded[i+2, j+1],padded[i+2, j+2],
                ]
                # if ((sum(neighbors_up)>=6) & (sum(neighbors_down)<6) | (sum(neighbors_up)<6) & (sum(neighbors_down)>=6)):
                #     # Subtract 1 from coordinates to account for padding
                #     loose_ends.append((i-1, j-1))
    # left right loose end
                neighbors_left = [
                     padded[i-2, j-2], padded[i-2, j-1],
                     padded[i-1, j-2], padded[i-1, j-1], 
                     padded[i,   j-2], padded[i,   j-1],  
                     padded[i+1, j-2], padded[i+1, j-1], 
                     padded[i+2, j-2], padded[i+2, j-1],
                ]
                neighbors_right = [
                                     padded[i-2, j+1],padded[i-2, j+2],
                                     padded[i-1, j+1],padded[i-1, j+2], 
                                     padded[i,   j+1],padded[i,   j+2],
                                     padded[i+1, j+1],padded[i+1, j+2], 
                                     padded[i+2, j+1],padded[i+2, j+2]
                ]
                # if ((sum(neighbors_left)>=6) & (sum(neighbors_right)<6) | (sum(neighbors_left)<6) & (sum(neighbors_right)>=6)):
                #     # Subtract 1 from coordinates to account for padding
                #     loose_ends.append((i-1, j-1))
    # diagonal
                neighbors_upleft = [
                     padded[i-2, j-2],padded[i-2, j-1],padded[i-2, j],padded[i-2, j+1],
                     padded[i-1, j-2],padded[i-1, j-1],padded[i-1, j],
                     padded[i,   j-2],padded[i,   j-1],  
                     padded[i+1, j-2],
                ]
                neighbors_downright = [
                                                                        padded[i-1,j+2],
                                                      padded[i,   j+1], padded[i,  j+2],
                                       padded[i+1,j], padded[i+1, j+1], padded[i+1,j+2],
                     padded[i+2, j-1], padded[i+2,j], padded[i+2, j+1], padded[i+2,j+2]
                ]
                neighbors_upright = [
                     padded[i-2, j-1],padded[i-2, j],padded[i-2, j+1],padded[i-2, j+2],
                                      padded[i-1, j],padded[i-1, j+1],padded[i-1, j+2], 
                                                     padded[i,   j+1],padded[i,   j+2],  
                                                                      padded[i+1, j+2], 
                ]
                neighbors_downleft = [
                     padded[i-1,j-2], 
                     padded[i,  j-2], padded[i,  j-1],
                     padded[i+1,j-2], padded[i+1,j-1],padded[i+1,j], 
                     padded[i+2,j-2], padded[i+2,j-1],padded[i+2,j], padded[i+2, j+1], 
                ]
                if ( (((sum(neighbors_upleft)>=pix_thresh) & (sum(neighbors_downright)<=pix_thresh_low) | (sum(neighbors_upleft)<=pix_thresh_low) & (sum(neighbors_downright)>=pix_thresh))
                 and ((sum(neighbors_upright)>=pix_thresh) & (sum(neighbors_downleft)<=pix_thresh_low) | (sum(neighbors_upright)<=pix_thresh_low) & (sum(neighbors_downleft)>=pix_thresh))
                 and ((sum(neighbors_left)>=pix_thresh) & (sum(neighbors_right)<=pix_thresh_low) | (sum(neighbors_left)<=pix_thresh_low) & (sum(neighbors_right)>=pix_thresh))
                 and ((sum(neighbors_up)>=pix_thresh) & (sum(neighbors_down)<=pix_thresh_low) | (sum(neighbors_up)<=pix_thresh_low) & (sum(neighbors_down)>=pix_thresh)))):
                    # Subtract 1 from coordinates to account for padding
                    loose_ends.append((i-1, j-1))
                    # set loose end to 3 to visualize. Should set it back to -1 once finish debugging
                    matrix[i-1,j-1] = 3
    return loose_ends,matrix

def post_process_loose_ends(loose_ends, matrix, distance_threshold=2):
    """
    Remove loose ends that are very close to each other.
    
    Args:
    loose_ends (list): List of tuples containing coordinates of loose ends.
    distance_threshold (float): Maximum distance for loose ends to be considered duplicates.
    
    Returns:
    list: Filtered list of loose ends.
    """
    if len(loose_ends) <= 1:
        return loose_ends,matrix

    # Convert loose ends to a numpy array
    points = np.array(loose_ends)
    
    # print('len of points:',len(points))
    # Compute pairwise distances
    distances = pdist(points)
    dist_matrix = squareform(distances)
    # print('shape distances: ',distances.shape)
    # Find pairs of points closer than the threshold
    close_pairs = np.argwhere(dist_matrix < distance_threshold)
    # print('len of close_pairs:',len(close_pairs))
    # Only keep pairs where first index is smaller (to avoid duplicates)
    close_pairs = close_pairs[close_pairs[:, 0] < close_pairs[:, 1]]
    # print('len of close_pairs:',len(close_pairs),close_pairs[0],close_pairs[1],close_pairs[2])
    # If no close pairs found, return original list
    if len(close_pairs) == 0:
        return loose_ends,matrix
    
    # Create a set of indices to remove
    to_remove = set()
    for pair in close_pairs:
        # Always remove the second point in the pair
        to_remove.add(pair[1])
        # print(pair)
        # matrix[np.ix_(pair)] = 1 # set to 1 (grain) if this boundary point is removed

    # Create new list of loose ends, excluding the ones marked for removal
    filtered_loose_ends = [point for i, point in enumerate(loose_ends) if i not in to_remove]
    for point in filtered_loose_ends:
        matrix[point] = 4 # temporarily set to 4 for debugging.
    # set the close loose ends to 1 (grain)
    for i in to_remove:
        matrix[loose_ends[i]] = 2

    return filtered_loose_ends,matrix

def draw_wide_line(matrix, rr, cc, width=1):
    # Ensure that the line is within the bounds of the matrix
    for r, c in zip(rr, cc):
        for dr in range(-width // 2, width // 2 + 1):
            for dc in range(-width // 2, width // 2 + 1):
                if 0 <= r + dr < matrix.shape[0] and 0 <= c + dc < matrix.shape[1]:
                   matrix[r + dr, c + dc] = -2  # Set the pixel to -2, separate from the boundary

def uniform_thickness_skeleton(matrix, target_thickness=2):
    if isinstance(matrix, torch.Tensor):
        matrix_np = matrix.cpu().numpy()
    else:
        matrix_np = matrix
    # Convert 1s to 0s and -1s to 1s for standard binary image processing
    binary = (matrix_np == -1).astype(np.uint8)
    
    # Perform skeletonization
    skeleton = skeletonize(binary)
    
    # Create a circular structuring element for dilation
    radius = target_thickness // 2
    se = disk(radius)
    
    # Dilate the skeleton
    dilated = ndimage.binary_dilation(skeleton, structure=se)
    
    # Convert back to original format (1 for background, -1 for lines)
    result = 1 - 2 * dilated.astype(int)
    
    return torch.from_numpy(result).float()

def trim_branches(matrix):
    results = matrix.clone()
    for i in range(matrix.shape[0]-3):
        for j in range(matrix.shape[1]-3):
            if(matrix[i,j]==-1 or matrix[i,j]==4):
                # branches under a horizontal branch
                if( matrix[i,j-1]==1 and matrix[i,j+1]==1 and matrix[i+1,j-1]==-1 and matrix[i+1,j]==-1 and matrix[i+1,j+1]==-1):
                    results[i,j] = 1
                hori_redu_branch = ( matrix[i,j-3] +matrix[i,j-2] + matrix[i,j-1]+ matrix[i,j+1] + matrix[i,j+2]+ matrix[i,j+3] )
                hori_main_branch = ( matrix[i+1,j-3]+matrix[i+1,j-2] + matrix[i+1,j-1]+matrix[i+1,j]+ matrix[i+1,j+1] + matrix[i+1,j+2]+matrix[i+1,j+3] )
                hori_vert_connect = ( matrix[i,  j-1] + matrix[i,  j] + matrix[i,  j+1] +
                                      matrix[i-1,j-1] + matrix[i-1,j] + matrix[i-1,j+1] + 
                                      matrix[i-2,j-1] + matrix[i-2,j] + matrix[i-2,j+1])
                if(hori_main_branch<=-5 and hori_redu_branch>=1 and hori_vert_connect>=0):
                    results[i,j] = 1

                # branches above a horizontal branch
                if( matrix[i,j-1]==1 and matrix[i,j+1]==1 and matrix[i-1,j-1]==-1 and matrix[i-1,j]==-1 and matrix[i-1,j+1]==-1):
                    results[i,j] = 1
                hori_redu_branch = ( matrix[i,j-2] + matrix[i,j-1]+ matrix[i,j+1] + matrix[i,j+2] )
                hori_main_branch = ( matrix[i-1,j-3] + matrix[i-1,j-2] + matrix[i-1,j-1]+ matrix[i-1,j]+ matrix[i-1,j+1] + matrix[i-1,j+2] + matrix[i-1,j+3] )
                hori_vert_connect = ( matrix[i+2,j-1] + matrix[i+2,j] + matrix[i+2,j+1] +
                                      matrix[i+1,j-1] + matrix[i+1,j] + matrix[i+1,j+1] + 
                                      matrix[i,  j-1] + matrix[i,  j] + matrix[i,  j+1])
                if(hori_main_branch<=-5 and hori_redu_branch>=1 and hori_vert_connect>=0):
                    results[i,j] = 1
                
                # branches to the right a vertical branch
                vert_redu_branch = ( matrix[i-3,j] + matrix[i-2,j] + matrix[i-1,j]+ matrix[i+1,j] + matrix[i+2,j] + matrix[i+3,j] )
                vert_main_branch = ( matrix[i-3,j-2] + matrix[i-3,j-1] + 
                                     matrix[i-2,j-2] + matrix[i-2,j-1] + 
                                     matrix[i-1,j-2] + matrix[i-1,j-1] + 
                                     matrix[i,  j-2] + matrix[i,  j-1] + 
                                     matrix[i+1,j-2] + matrix[i+1,j-1] + 
                                     matrix[i+2,j-2] + matrix[i+2,j-1] + 
                                     matrix[i+3,j-2] + matrix[i+3,j-1] )
                vert_hori_connect = ( matrix[i+1,j] + matrix[i+1,j+1] + matrix[i+1,j+2] + 
                                      matrix[i  ,j] + matrix[i  ,j+1] + matrix[i  ,j+2] +
                                      matrix[i-1,j] + matrix[i-1,j+1] + matrix[i-1,j+2])
                if(vert_main_branch<=-4 and vert_redu_branch>=0 and vert_hori_connect>=0):
                    results[i,j] = 1
                # branches to the left a vertical branch
                vert_redu_branch = ( matrix[i-3,j] + matrix[i-2,j] + matrix[i-1,j]+ matrix[i+1,j] + matrix[i+2,j] + matrix[i+3,j] )
                vert_main_branch = ( matrix[i-3,j+1] + matrix[i-3,j+2] + 
                                     matrix[i-2,j+1] + matrix[i-2,j+2] + 
                                     matrix[i-1,j+1] + matrix[i-1,j+2] + 
                                     matrix[i,  j+1] + matrix[i,  j+2] + 
                                     matrix[i+1,j+1] + matrix[i+1,j+2] + 
                                     matrix[i+2,j+1] + matrix[i+2,j+2] + 
                                     matrix[i+3,j+1] + matrix[i+3,j+2] )
                vert_hori_connect = ( matrix[i+1,j] + matrix[i+1,j-1] + matrix[i+1,j-2] + 
                                      matrix[i  ,j] + matrix[i  ,j-1] + matrix[i  ,j-2] +
                                      matrix[i-1,j] + matrix[i-1,j-1] + matrix[i-1,j-2])
                if(vert_main_branch<=-4 and vert_redu_branch>=0 and vert_hori_connect>=0):
                    results[i,j] = 1
    return results


def find_center_of_mass(arr, start_row, start_col, search_radius=6):
    rows, cols = arr.shape
    total_weight = 0
    weighted_sum_row = 0
    weighted_sum_col = 0

    initial_radius = search_radius - 3
    direction_matrix = arr[max(0, start_row - initial_radius):min(rows, start_row + initial_radius + 1),max(0, start_col - initial_radius):min(cols, start_col + initial_radius + 1)]
    # print(direction_matrix)
    # print("start_row: ",start_row, ", start_col: ",start_col)
    for r in range(max(0, start_row - initial_radius), min(rows, start_row + initial_radius + 1)):
        for c in range(max(0, start_col - initial_radius), min(cols, start_col + initial_radius + 1)):
            if arr[r, c] == -1:
                # distance = ((r - start_row) ** 2 + (c - start_col) ** 2) ** 0.5
                # weight = 1 / (distance +0.5)  # Adding 1 to avoid division by zero
                total_weight += 1
                weighted_sum_row += r #* weight
                weighted_sum_col += c #* weight
    # temporary solution, if there is an island 4, simply return inf
    if(total_weight==0):
        return start_row, start_col
    center_row = weighted_sum_row / total_weight
    center_col = weighted_sum_col / total_weight
    
    # print(direction_matrix)
    # print("center_row: ",center_row,", center_col: ",center_col, ", start_row: ",start_row, ", start_col: ",start_col)
    # print(center_row>=start_row,center_col<=start_col)
    # print(arr[start_row-1:min(rows, start_row + search_radius + 2), max(0, start_col - search_radius):start_col+2])
    total_weight = 0
    weighted_sum_row = 0
    weighted_sum_col = 0
    if (center_row>=start_row and center_col<=start_col):
        for r in range(start_row-1, min(rows, start_row + search_radius + 2)):
            for c in range(max(0, start_col - search_radius), start_col+2):
                if arr[r, c] == -1:
                    # distance = ((r - start_row) ** 2 + (c - start_col) ** 2) ** 0.5
                    # weight = 1 / (distance + 0)  # Adding 1 to avoid division by zero
                    total_weight += 1 #weight
                    weighted_sum_row += r #* weight
                    weighted_sum_col += c #* weight
        if total_weight == 0:
            return start_row, start_col  # Return starting position if no -1s found
        center_row = weighted_sum_row / total_weight
        center_col = weighted_sum_col / total_weight
        # print(center_row,center_col)
    
    elif (center_row>=start_row and center_col>=start_col):
#         print(arr[max(0, start_row-1):min(rows, start_row + search_radius + 2),max(0, start_col-1):min(cols, start_col + search_radius + 2)]
# )
        for r in range(start_row, min(rows, start_row + search_radius + 2)):
            for c in range(start_col,min(cols, start_col + search_radius + 2)):
                if arr[r, c] == -1:
                    # distance = ((r - start_row) ** 2 + (c - start_col) ** 2) ** 0.5
                    # weight = 1 / (distance + 0)  # Adding 1 to avoid division by zero
                    total_weight += 1 #weight
                    weighted_sum_row += r #* weight
                    weighted_sum_col += c #* weight
        if total_weight == 0:
            return start_row, start_col  # Return starting position if no -1s found
        center_row = weighted_sum_row / total_weight
        center_col = weighted_sum_col / total_weight

    elif (center_row<=start_row and center_col>=start_col):
        for r in range(max(0, start_row - search_radius),start_row+2):
            for c in range(start_col-1,min(cols, start_col + search_radius + 2)):
                if arr[r, c] == -1:
                    # distance = ((r - start_row) ** 2 + (c - start_col) ** 2) ** 0.5
                    # weight = 1 / (distance + 0)  # Adding 1 to avoid division by zero
                    total_weight += 1 #weight
                    weighted_sum_row += r #* weight
                    weighted_sum_col += c #* weight
        if total_weight == 0:
            return start_row, start_col  # Return starting position if no -1s found
        center_row = weighted_sum_row / total_weight
        center_col = weighted_sum_col / total_weight

    elif (center_row<=start_row and center_col<=start_col):
        for r in range(max(0, start_row - search_radius),start_row+2):
            for c in range(max(0, start_col - search_radius), start_col+2):
                if arr[r, c] == -1:
                    # distance = ((r - start_row) ** 2 + (c - start_col) ** 2) ** 0.5
                    # weight = 1 / (distance + 0)  # Adding 1 to avoid division by zero
                    total_weight += 1 #weight
                    weighted_sum_row += r #* weight
                    weighted_sum_col += c #* weight
        if total_weight == 0:
            return start_row, start_col  # Return starting position if no -1s found
        center_row = weighted_sum_row / total_weight
        center_col = weighted_sum_col / total_weight
        
    else:
        print("not implemented yet")
    return center_row, center_col

def calculate_slope(start, end):
    dy = end[0] - start[0]
    dx = end[1] - start[1]
    # print(dx,dy)
    if dx == 0 and dy != 0:
        return float('inf') if dy > 0 else float('-inf')
    elif dx==0 and dy==0:
        return float('nan')
    return dy / dx


def draw_line_with_slope(arr, start, center_row, center_col, slope):
    row, col = start
    rows, cols = arr.shape
    # print("start: ", row, col, ", slope: ", slope)

    def is_valid_and_not_negative_one(r, c):
        return 0 <= r < rows and 0 <= c < cols and arr[r, c] != -1 and arr[r, c] != -2
    
    def draw_pixel(r, c):
        if is_valid_and_not_negative_one(r, c):
            arr[r, c] = 5
        if is_valid_and_not_negative_one(r, c+1):  # Draw second pixel to the right
            arr[r, c+1] = 5
        if is_valid_and_not_negative_one(r, c-1):  # Draw second pixel to the left
            arr[r, c-1] = 5
        if is_valid_and_not_negative_one(r-1, c):  # Draw third pixel below
            arr[r-1, c] = 5
        if is_valid_and_not_negative_one(r+1, c):  # Draw third pixel above
            arr[r+1, c] = 5
    
    if slope == float('inf') or slope == float('-inf'):
        # Handle vertical line
        step = -1 if slope == float('inf') else 1
        for r in range(row, rows if step > 0 else 1, step):
            if (not is_valid_and_not_negative_one(r, col) or 
                not is_valid_and_not_negative_one(r, col+1) or 
                not is_valid_and_not_negative_one(r, col-1)):
                break
            draw_pixel(r, col)
    elif slope == 0 or slope == -0:
        # print("slope: ",slope,center_col,col)
        # Handle horizontal line
        if (center_col<col):
            step = 1
        else:
            step = -1
        for c in range(col, cols if step>0 else 1, step):
            # print("row, c: ",row, c,step)
            if (not is_valid_and_not_negative_one(row, c) or 
                not is_valid_and_not_negative_one(row+1, c) or 
                not is_valid_and_not_negative_one(row-1, c)):
                break
            # print("drawing ",row, c)
            draw_pixel(row, c)
    elif abs(slope) >= 1:
        if (slope >0 and center_col<col):
            dir, step =-1, 1
        elif(slope >0 and center_col>col):
            dir,step = -1, -1
        elif(slope <0 and center_col>col):
            dir,step = -1, 1
        elif(slope <0 and center_col<col):
            dir, step = -1, -1
        
        # print("slope ",slope,", step: ",step)
        if center_row > row:
            start = row + step
            for r in range(start, -1 if dir < 0 else rows, step):
                c = int(col + (r - row) / slope)
                # print("r,c: ", r, c, ", arr[r, c]: ", arr[r, c])
                if (not is_valid_and_not_negative_one(r, c) or not is_valid_and_not_negative_one(r, c+1) or 
                    not is_valid_and_not_negative_one(r, c-1) or not is_valid_and_not_negative_one(r+1, c) or
                    not is_valid_and_not_negative_one(r-1, c)):
                    break
                draw_pixel(r, c)
        else:
            start = row + step
            # print("start: ",start,dir)
            for r in range(start, rows if dir < 0 else -1, step):
                c = int(col + (r - row) / slope)
                # print("r,c: ", r, c, ", arr[r, c]: ", arr[r, c])
                if (not is_valid_and_not_negative_one(r, c) or not is_valid_and_not_negative_one(r, c+1) or 
                    not is_valid_and_not_negative_one(r, c-1) or not is_valid_and_not_negative_one(r+1, c) or
                    not is_valid_and_not_negative_one(r-1, c)):
                    break
                draw_pixel(r, c)
    elif(math.isnan(slope)):
        arr[row,col] = 1 # if nan -> this is an island 4 -> set it to grain value (1)
    else:
        if (slope > 0 and center_col > col):
            dir, step = -1, -1
        elif (slope > 0 and center_col < col):
            dir, step = 1, 1
        elif (slope < 0 and center_col > col):
            dir, step = -1, -1
        elif (slope < 0 and center_col < col):
            dir, step = 1, 1
        if dir == -1:
            start = col - 1  
        else:
            start = col + 1
        # print("slope: ",slope,",dir: ",dir,", step: ",step,", start ",start)
        for c in range(start, cols if dir > 0 else -1, step):
            r = int(row + (c - col) * slope)
            # print("r,c: ", r, c, ", arr[r, c]: ", arr[r, c])
            # if(r==7 and c==6):
            #     print(is_valid_and_not_negative_one(r, c))
            if (not is_valid_and_not_negative_one(r, c) or not is_valid_and_not_negative_one(r, c+1) or 
                not is_valid_and_not_negative_one(r, c-1) or not is_valid_and_not_negative_one(r+1, c) or
                not is_valid_and_not_negative_one(r-1, c)):
                # print("breaking at r c: ", r,c)
                break
            draw_pixel(r, c)
    
    return arr


def draw_line_to_negative_one_group(arr,search_radius=5):
    results = arr.clone()
    # Find the position of 4
    positions = torch.where(arr == 4)
    for start_row, start_col in zip(positions[0], positions[1]):
        start_row, start_col = start_row.item(), start_col.item()
        
        # Find the center of mass of nearby -1s
        center_row, center_col = find_center_of_mass(arr, start_row, start_col, search_radius=search_radius)
        
        # Calculate the slope
        # print("(start_row, start_col), (center_row, center_col): ",(start_row, start_col), int(round(center_row)), int(round(center_col)))
        slope = calculate_slope((start_row, start_col), (int(round(center_row)), int(round(center_col))))
        # print("center index: ", (int(round(center_row)), int(round(center_col))), ", slope: ",slope)
        # Draw the line
        results = draw_line_with_slope(results, (start_row, start_col), center_row, center_col, slope)
    
    return results


def connect_broken_boundaries_optimized(matrix, max_gap=2):
    # Create a kernel to detect nearby boundaries
    device = matrix.device

    # Create a kernel to detect nearby boundaries
    kernel_size = 2 * max_gap + 1
    kernel = torch.ones((1, 1, kernel_size, kernel_size), dtype=torch.float32, device=device)
    
    # Pad the matrix
    padded = F.pad(matrix.float().unsqueeze(0).unsqueeze(0), 
                   (max_gap, max_gap, max_gap, max_gap), mode='constant', value=0)
    
    # Create a mask for boundaries
    boundary_mask = (padded == -1).float()
    
    # Convolve to count nearby boundaries
    nearby_count = F.conv2d(boundary_mask, kernel)
    
    # Find points that are not boundaries but have at least 2 nearby boundaries
    candidates = (matrix != -1) & (nearby_count.squeeze() >= 2)
    
    # Function to check if a point is disconnected
    def is_disconnected(r, c):
        neighborhood = matrix[max(0, r-1):min(r+2, matrix.shape[0]), 
                              max(0, c-1):min(c+2, matrix.shape[1])]
        return (neighborhood == -1).sum() == 1

    # Apply the disconnected check only to candidate points
    changes = torch.zeros_like(matrix, dtype=torch.bool)
    candidate_indices = torch.nonzero(candidates)
    for r, c in candidate_indices:
        if is_disconnected(r.item(), c.item()):
            changes[r, c] = True

    # Apply changes
    matrix[changes] = -1
    return matrix

def analyze_flynns(binary_matrix,pixel_area=1.0):
    if isinstance(binary_matrix, torch.Tensor):
        binary_matrix = binary_matrix.cpu().numpy()
    else:
        binary_matrix = binary_matrix
    # 1. Convert the matrix to a binary image where Flynn interiors are 1 and boundaries are 0
    flynn_image = (binary_matrix == 1).astype(int)
    flynn_image[:2,:],flynn_image[-2:,:],flynn_image[:,:2],flynn_image[:,-2:] = 0,0,0,0
    # print(f'in analyze_flynns unique flynn_image: {np.unique(flynn_image)}')
    # 2. Label connected components (Flynns)
    labeled_array, num_features = ndimage.label(flynn_image)
    # 3. Count the number of Flynns
    num_flynns = num_features
    # 4. Calculate areas of all Flynns
    areas = pixel_area * ndimage.sum(flynn_image, labeled_array, range(1, num_features + 1))
    # 5. Calculate mean and standard deviation of Flynn areas
    mean_area = np.mean(areas)
    std_area = np.std(areas)
    return num_flynns, mean_area, std_area, labeled_array

def calculate_grainsize(grains,domain_length):
    labeled_grains, num_features = label(grains)
    used_colors = set()
    label_colored = np.zeros((*labeled_grains.shape, 3), dtype=np.uint8)
    # We will map each label to a color
    for label_id in range(1, num_features + 1):
        # Assign a random color for each grain
        unique_color = generate_unique_color(used_colors)
        label_colored[labeled_grains == label_id] = unique_color
    mean_grainarea = np.sum(grains == 1)/num_features
    mean_grainsize = np.sqrt(mean_grainarea/3.1415926)*2*domain_length/256
    # print("unique grains: ", num_features,", mean diameter: ",mean_grainsize)

    return label_colored, mean_grainsize, num_features

def calculate_grainsize_gpu(grains, domain_length):
    """
    Calculate grain size on GPU.
    
    Args:
    grains (torch.Tensor): Binary tensor of grains on GPU.
    domain_length (float): Length of the domain.
    
    Returns:
    tuple: (labeled_grains, mean_grainsize, num_features)
    """
    # Use connected components labeling
    labeled_grains = label_gpu(grains)
    
    num_features = torch.max(labeled_grains).item()
    
    # Calculate mean grain area and size
    total_grain_area = torch.sum(grains == 1).float()
    mean_grainarea = total_grain_area / num_features
    mean_grainsize = torch.sqrt(mean_grainarea / 3.1415926) * 2 * domain_length / grains.shape[0]
    
    return labeled_grains, mean_grainsize.item(), num_features

def label_gpu(binary_image):
    """
    Perform connected components labeling on GPU.
    
    Args:
    binary_image (torch.Tensor): Binary image tensor on GPU.
    
    Returns:
    torch.Tensor: Labeled image.
    """
    # Ensure the input is binary and float
    binary_image = (binary_image > 0).float()
    
    # Create kernels for 4-connectivity
    kernel = torch.tensor([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=torch.float32, device=binary_image.device)
    
    labeled = torch.zeros_like(binary_image, dtype=torch.int32)
    current_label = 1
    
    while torch.any(binary_image > 0):
        # Find an unlabeled pixel
        seed = torch.nonzero(binary_image, as_tuple=False)[0]
        
        # Grow region
        region = torch.zeros_like(binary_image)
        region[seed[0], seed[1]] = 1
        
        while True:
            dilated = F.conv2d(region.unsqueeze(0).unsqueeze(0), kernel.unsqueeze(0).unsqueeze(0), padding=1)[0, 0]
            new_region = ((dilated > 0) & (binary_image > 0)).float()
            if torch.all(new_region == region):
                break
            region = new_region
        
        # Label the region and remove it from binary_image
        labeled[region > 0] = current_label
        binary_image[region > 0] = 0
        current_label += 1
    
    return labeled

def plot_kde_time_series(kde_data,grainsize_range,ax,fig):
    '''
    data in: time series of kde
    shape requiremenet: [num of bins, num of time steps]
    '''
    n_points, n_timesteps = kde_data.shape
    print(kde_data.shape,n_points, n_timesteps)
    # Create a custom colormap from light pink to dark red
    colors = ['#FFDAE0', '#990000']  # Light pink to red
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("custom_red", colors, N=n_bins)
    
    # fig, ax = plt.subplots(figsize=(7, 4),dpi=300)
    
    # Plot each time step
    for i in range(n_timesteps):
        color = cmap(i / (n_timesteps - 1))
        ax.plot(grainsize_range, kde_data[:,i], color=color,linewidth=0.5, alpha=0.5)
        # print(f'i {i}, {kde_data[:6,i]}')

    ax.set_xlabel(r'Grain Area [mm$^2$]')
    ax.set_ylabel('Distribution Density')
    # ax.set_ylim([kde_grainsize.min(),0.11])
    # ax.set_title('KDE Time Series')
    
    # Create a colorbar to show the time progression
    # n_timesteps *= 1.2/30
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=n_timesteps))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label('Time [month]')
    max_month = int(n_timesteps) # careful, if time is too short, after int() max_month may = 0
    print(max_month)
    cbar.locator = ticker.MaxNLocator(nbins=min(max_month, 6), integer=True)
    cbar.update_ticks()
    
    # plt.tight_layout()
    # plt.show()

#################################################
#
# Utilities
#
#################################################
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# if torch.cuda.is_available():
#     torch.set_default_tensor_type('torch.cuda.FloatTensor')
# else:
#     torch.set_default_tensor_type('torch.FloatTensor')

# reading data
class MatReader(object):
    def __init__(self, file_path, to_torch=True, to_cuda=False, to_float=True):
        super(MatReader, self).__init__()

        self.to_torch = to_torch
        self.to_cuda = to_cuda
        self.to_float = to_float

        self.file_path = file_path

        self.data = None
        self.old_mat = None
        self._load_file()

    def _load_file(self):
        try:
            self.data = scipy.io.loadmat(self.file_path)
            self.old_mat = True
        except:
            self.data = h5py.File(self.file_path)
            self.old_mat = False

    def load_file(self, file_path):
        self.file_path = file_path
        self._load_file()

    def read_field(self, field):
        x = self.data[field]

        if not self.old_mat:
            x = x[()]
            x = np.transpose(x, axes=range(len(x.shape) - 1, -1, -1))

        if self.to_float:
            x = x.astype(np.float32)

        if self.to_torch:
            x = torch.from_numpy(x)

            if self.to_cuda:
                x = x.cuda()

        return x

    def set_cuda(self, to_cuda):
        self.to_cuda = to_cuda

    def set_torch(self, to_torch):
        self.to_torch = to_torch

    def set_float(self, to_float):
        self.to_float = to_float

# normalization, pointwise gaussian
class UnitGaussianNormalizer(object):
    def __init__(self, x, eps=0.00001, time_last=True):
        super(UnitGaussianNormalizer, self).__init__()

        # x could be in shape of ntrain*n or ntrain*T*n or ntrain*n*T in 1D
        # x could be in shape of ntrain*w*l or ntrain*T*w*l or ntrain*w*l*T in 2D
        self.mean = torch.mean(x, 0)
        self.std = torch.std(x, 0)
        self.eps = eps
        self.time_last = time_last # if the time dimension is the last dim

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        # sample_idx is the spatial sampling true_image
        if sample_idx is None:
            std = self.std + self.eps # n
            mean = self.mean
        else:
            if self.mean.ndim == sample_idx.ndim or self.time_last:
                std = self.std[sample_idx] + self.eps  # batch*n
                mean = self.mean[sample_idx]
            if self.mean.ndim > sample_idx.ndim and not self.time_last:
                    std = self.std[...,sample_idx] + self.eps # T*batch*n
                    mean = self.mean[...,sample_idx]
        # x is in shape of batch*(spatial discretization size) or T*batch*(spatial discretization size)
        x = (x * std) + mean
        return x

    def to(self, device):
        if torch.is_tensor(self.mean):
            self.mean = self.mean.to(device)
            self.std = self.std.to(device)
        else:
            self.mean = torch.from_numpy(self.mean).to(device)
            self.std = torch.from_numpy(self.std).to(device)
        return self

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()

# normalization, Gaussian
class GaussianNormalizer(object):
    def __init__(self, x, eps=0.00001):
        super(GaussianNormalizer, self).__init__()

        self.mean = torch.mean(x)
        self.std = torch.std(x)
        self.eps = eps

    def encode(self, x):
        x = (x - self.mean) / (self.std + self.eps)
        return x

    def decode(self, x, sample_idx=None):
        x = (x * (self.std + self.eps)) + self.mean
        return x

    def cuda(self):
        self.mean = self.mean.cuda()
        self.std = self.std.cuda()

    def cpu(self):
        self.mean = self.mean.cpu()
        self.std = self.std.cpu()


# normalization, scaling by range
class RangeNormalizer(object):
    def __init__(self, x, low=0.0, high=1.0):
        super(RangeNormalizer, self).__init__()
        mymin = torch.min(x, 0)[0].view(-1)
        mymax = torch.max(x, 0)[0].view(-1)

        self.a = (high - low)/(mymax - mymin)
        self.b = -self.a*mymax + high

    def encode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = self.a*x + self.b
        x = x.view(s)
        return x

    def decode(self, x):
        s = x.size()
        x = x.view(s[0], -1)
        x = (x - self.b)/self.a
        x = x.view(s)
        return x

#loss function with rel/abs Lp loss
class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]
        print("in abs:",x.shape,y.shape)
        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0)

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]
        # print("in rel:",x.shape,y.shape)
        # exit()
        # compute the norm of order self.p, 1 meaning norm is calculated across the second dimension 
        # (the flattened feature dimension) for each example in the batch
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    # EL added bross entropy loss for binary prediction
    def binary_cross_entropy(self, y_pred, y_true):
        """
        Calculate binary cross entropy loss.
        Args:
            y_pred (torch.Tensor): Predicted probabilities.
            y_true (torch.Tensor): Ground truth labels.
        Returns:
            torch.Tensor: The binary cross entropy loss.
        """
        num_examples = y_pred.size()[0]
        y_pred,y_true = y_pred.reshape(num_examples,-1),y_true.reshape(num_examples,-1)
        # a = y_pred.cpu().detach().numpy()
        # print("EL - in loss, unique y_pred: ",a.max(),",",a.min(),", y_true: ",np.unique(y_true.cpu().detach().numpy()))
        # Ensure predictions are in the range (0, 1)
        y_pred = torch.clamp(y_pred, min=1e-7, max=1-1e-7)
        
        # Calculate binary cross entropy loss manually
        loss = - (y_true * torch.log(y_pred) + (1 - y_true) * torch.log(1 - y_pred))
        
        if self.reduction:
            if self.size_average:
                return torch.mean(loss)
            else:
                return torch.sum(loss)
        # loss = F.binary_cross_entropy(y_pred.reshape(num_examples,-1), y_true.reshape(num_examples,-1), reduction='mean' if self.size_average else 'sum')

        return loss

    def __call__(self, x, y):
        # EL - here defines what loss function to use
        return self.rel(x, y)
        # return self.binary_cross_entropy(x, y)

# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss(object):
    def __init__(self, d=2, p=2, k=1, a=None, group=False, size_average=True, reduction=True):
        super(HsLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a == None:
            a = [1,] * k
        self.a = a

    def rel(self, x, y):
        num_examples = x.size()[0]
        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)
        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)
        return diff_norms/y_norms

    def __call__(self, x, y, a=None):
        nx = x.size()[1]
        ny = x.size()[2]
        k = self.k
        balanced = self.balanced
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = torch.cat((torch.arange(start=0, end=nx//2, step=1),torch.arange(start=-nx//2, end=0, step=1)), 0).reshape(nx,1).repeat(1,ny)
        k_y = torch.cat((torch.arange(start=0, end=ny//2, step=1),torch.arange(start=-ny//2, end=0, step=1)), 0).reshape(1,ny).repeat(nx,1)
        k_x = torch.abs(k_x).reshape(1,nx,ny,1).to(x.device)
        k_y = torch.abs(k_y).reshape(1,nx,ny,1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if balanced==False:
            weight = 1
            if k >= 1:
                weight += a[0]**2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1]**2 * (k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x*weight, y*weight)
        else:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x*weight, y*weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(k_x**4 + 2*k_x**2*k_y**2 + k_y**4)
                loss += self.rel(x*weight, y*weight)
            loss = loss / (k+1)

        return loss

# A simple feedforward neural network
class DenseNet(torch.nn.Module):
    def __init__(self, layers, nonlinearity, out_nonlinearity=None, normalize=False):
        super(DenseNet, self).__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.layers = nn.ModuleList()

        for j in range(self.n_layers):
            self.layers.append(nn.Linear(layers[j], layers[j+1]))

            if j != self.n_layers - 1:
                if normalize:
                    self.layers.append(nn.BatchNorm1d(layers[j+1]))

                self.layers.append(nonlinearity())

        if out_nonlinearity is not None:
            self.layers.append(out_nonlinearity())

    def forward(self, x):
        for _, l in enumerate(self.layers):
            x = l(x)

        return x


# print the number of parameters
def count_params(model):
    c = 0
    for p in list(model.parameters()):
        c += reduce(operator.mul, 
                    list(p.size()+(2,) if p.is_complex() else p.size()))
    return c

# check num of tensors and shapes in data lader
def check_dataloader(train_loader):
    for batch in train_loader:
        # print(batch)
        print([x.shape for x in batch])  # If they are tensors, check their shapes
        break

def calculate_iou(true_image, pred_image):
    # Ensure the images are binary (0 and 1 only)
    # true_image = true_image.astype(bool)
    # pred_image = pred_image.astype(bool)
    true_image = (true_image == 1).astype(int)
    pred_image = (pred_image == 1).astype(int)

    # Calculate intersection and union
    intersection = np.logical_and(true_image, pred_image)
    union = np.logical_or(true_image, pred_image)
    # print(f'in iou, unique pred: {np.unique(pred_image)}, unique true: {np.unique(true_image)}, inter: {np.sum(intersection)}, union: {np.sum(union)}')
    # Calculate IoU
    iou = np.sum(intersection) / np.sum(union)

    return iou

def calculate_deviation(true_image, pred_image):
    """
    Calculates deviation metrics between boundaries represented by zero pixels in true_image and pred_image.
    Works for multiple boundaries.
    """
    # Generate Nx2 matrix of pixels that represent the boundaries
    # print("EL - unique pred: ",np.unique(pred_image),np.unique(true_image))
    true_image = (true_image == 1).astype(int)
    pred_image = (pred_image == 1).astype(int)
    
    x1, y1 = np.where(pred_image == 0)
    x2, y2 = np.where(true_image == 0)
    pred_coords = np.array(list(zip(x1, y1)))
    true_coords = np.array(list(zip(x2, y2)))
    
    # If no boundary is detected in either pred or true image, return default values
    if pred_coords.shape[0] == 0 or true_coords.shape[0] == 0:
        print("!!!! No boundary detected !!!!")
        print("pred_coords.shape: ",pred_coords.shape)
        print("true_coords.shape: ",true_coords.shape)
        return np.nan, np.nan, np.nan, np.array([])
    
    # Generate the pairwise distances between each point and the closest point in the other array
    distances1 = distance.cdist(pred_coords, true_coords).min(axis=1)
    distances2 = distance.cdist(true_coords, pred_coords).min(axis=1)
    distances = np.concatenate((distances1, distances2))
    
    # Calculate the metrics
    mean_deviation = np.mean(distances)
    median_deviation = np.median(distances)
    std_deviation = np.std(distances)
    # print(f'inside deviation, median: {median_deviation},{distances.max()},{distances.min()}')
    # print(distances)
    # fig = plt.figure(figsize=(18, 10))
    # plt.hist(distances, bins=40, edgecolor='black', alpha=0.7)
    # plt.title('Error Distribution')
    # plt.xlabel('Error')
    # plt.ylabel('Frequency')
    # fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
    # fig.savefig('error_hist.png')
    # exit(0)
    return mean_deviation, median_deviation, std_deviation, distances