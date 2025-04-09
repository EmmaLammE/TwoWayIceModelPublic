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
from scipy.spatial import distance
from scipy.ndimage import label, generate_binary_structure


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

def clean_clustered_array_torch(data, threshold=0.05, min_region_size=10):
    # Ensure the input is a PyTorch tensor
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, device='cuda')
    
    # Move to GPU if not already
    data = data.cuda()
    
    # Normalize data to [0, 1] range
    normalized_data = (data - data.min()) / (data.max() - data.min())
    
    batch_size, height, width, _ = data.shape
    
    # Prepare for batch processing
    clean_data = torch.zeros_like(data)
    n_clusters_list = []
    
    for b in range(batch_size):
        # Work on 2D slice
        slice_2d = normalized_data[b, :, :, 0]
        
        # Create an empty mask for labeling
        mask = torch.zeros_like(slice_2d, dtype=torch.bool)
        labels = torch.zeros_like(slice_2d, dtype=torch.int32)
        next_label = 1
        
        # Define 8-connectivity kernel
        kernel = torch.tensor([[1, 1, 1],
                               [1, 0, 1],
                               [1, 1, 1]], device='cuda').float()
        
        for i in range(height):
            for j in range(width):
                if mask[i, j]:
                    continue
                
                # Start a new region
                region_mask = torch.zeros_like(slice_2d, dtype=torch.bool)
                region_mask[i, j] = True
                
                while True:
                    # Expand region
                    expanded = F.conv2d(region_mask.float().unsqueeze(0).unsqueeze(0), 
                                        kernel.unsqueeze(0).unsqueeze(0), 
                                        padding=1).squeeze() > 0
                    
                    # Find new pixels within threshold
                    new_pixels = expanded & ~region_mask & ~mask & \
                                 (torch.abs(slice_2d - slice_2d[i, j]) <= threshold)
                    
                    if not new_pixels.any():
                        break
                    
                    region_mask |= new_pixels
                
                if region_mask.sum() >= min_region_size:
                    mask |= region_mask
                    labels[region_mask] = next_label
                    next_label += 1
        
        # Assign mean values to each region
        for label in range(1, next_label):
            region = labels == label
            clean_data[b, :, :, 0][region] = data[b, :, :, 0][region].mean()
        
        n_clusters_list.append(next_label - 1)
    
    return torch.tensor(n_clusters_list, device='cuda'), clean_data

def clean_clustered_array(data, threshold=5, min_region_size=10):
    # Normalize data to [0, 1] range
    normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
    
    # Create an empty mask for labeling
    mask = np.zeros_like(data, dtype=bool)
    labels = np.zeros_like(data, dtype=int)
    next_label = 1
    
    # Define 8-connectivity
    s = generate_binary_structure(2, 2)
    
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if mask[i, j]:
                continue
            
            # Start a new region
            region_mask = np.zeros_like(data, dtype=bool)
            region_mask[i, j] = True
            front = [(i, j)]
            
            while front:
                x, y = front.pop(0)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < data.shape[0] and 0 <= ny < data.shape[1] and
                            not mask[nx, ny] and not region_mask[nx, ny] and
                            abs(normalized_data[nx, ny] - normalized_data[i, j]) <= threshold / 100):
                            region_mask[nx, ny] = True
                            front.append((nx, ny))
            
            if np.sum(region_mask) >= min_region_size:
                mask[region_mask] = True
                labels[region_mask] = next_label
                next_label += 1
    
    # Assign mean values to each region
    clean_data = np.zeros_like(data)
    for label in range(1, next_label):
        region = labels == label
        clean_data[region] = np.mean(data[region])
    
    return next_label - 1, clean_data

def generate_unique_color(existing_colors):
    while True:
        color = tuple(np.random.choice(range(256), size=3))
        existing_colors.add(color)
        return color

def reinitialize(grain_pred):
    grain_pred_plt = np.copy(grain_pred)
    grain_mean = 1.0*np.mean(grain_pred_plt)
    # print(grain_mean)
    grain_pred_plt[grain_pred>grain_mean] = 1 # grains
    grain_pred_plt[grain_pred<=grain_mean] = 0 # boundaries
    # grain_pred_plt[0:2,:], grain_pred_plt[-3:-1,:], grain_pred_plt[:,0:2], grain_pred_plt[:,-3:-1] = 0,0,0,0
    return grain_pred_plt

def calculate_grainsize(grains,domain_length):
    labeled_grains, num_features = label(grains)
    used_colors = set()
    label_colored = np.zeros((*labeled_grains.shape, 3), dtype=np.uint8)
    # We will map each label to a color
    for label_id in range(1, num_features + 1):
        # Assign a random color for each grain
        unique_color = generate_unique_color(used_colors)
        label_colored[labeled_grains == label_id] = unique_color
    mean_grainarea = np.sum(grains == 1)/num_features*(domain_length/450)**2
    mean_grainsize = np.sqrt(mean_grainarea/3.1415926)*2
    # print("unique grains: ", num_features,", mean_grainsize: ",mean_grainsize)

    return label_colored, mean_grainsize, num_features

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

def cosine_similarity(matrix1, matrix2):
    """
    Calculate cosine similarity between two matrices of angles.
    
    :param matrix1: First matrix of angles in radians
    :param matrix2: Second matrix of angles in radians
    :return: Cosine similarity (1 is perfect similarity, -1 is opposite)
    """
    # Convert angles to complex numbers on the unit circle
    complex1 = np.exp(1j * np.radians(matrix1))
    complex2 = np.exp(1j * np.radians(matrix2))
    
    # Calculate the mean of the complex numbers
    mean1 = np.mean(complex1)
    mean2 = np.mean(complex2)
    
    # Calculate cosine similarity
    similarity = np.real(mean1 * np.conj(mean2)) / (np.abs(mean1) * np.abs(mean2))
    
    return similarity

def mean_angular_error(matrix1, matrix2):
    """
    Calculate mean angular error between two matrices of angles.
    
    :param matrix1: First matrix of angles in radians
    :param matrix2: Second matrix of angles in radians
    :return: Mean angular error in radians
    """
    # Calculate the angular difference
    diff = np.radians(matrix1) - np.radians(matrix2)
    
    # Wrap the difference to [-pi, pi]
    wrapped_diff = (diff + np.pi) % (2 * np.pi) - np.pi
    
    # Calculate the mean of the absolute wrapped differences
    mae = np.mean(np.abs(wrapped_diff))
    
    return mae*180/math.pi



def euler_to_rotation_bunge(phi2, Phi, phi1):
    # Convert degrees to radians using torch
    phi1, Phi, phi2 = torch.deg2rad(torch.tensor([phi1, Phi, phi2], device=phi1.device))
    
    # Bunge (ZXZ) convention using torch trigonometric functions
    c1, s1 = torch.cos(phi1), torch.sin(phi1)
    c, s = torch.cos(Phi), torch.sin(Phi)
    c2, s2 = torch.cos(phi2), torch.sin(phi2)

    R = torch.tensor([
        [c1*c2 - s1*s2*c,   s1*c2 + c1*s2*c,   s2*s],
        [-c1*s2 - s1*c2*c, -s1*s2 + c1*c2*c,   c2*s],
        [s1*s,             -c1*s,              c]
    ], device=phi1.device)
    
    return R

def calculate_orientation_tensor(c_axes):
    """Calculate the orientation tensor from c-axes orientations."""
    # Use torch.mean and torch.outer for tensor operations
    return torch.mean(torch.stack([torch.outer(c, c) for c in c_axes]), dim=0)

def calculate_eigenvalues(orientation_tensor):
    """Calculate eigenvalues from the orientation tensor."""
    evals, _ = torch.linalg.eig(orientation_tensor)
    evals = evals.real  # Ensure we take the real part of the eigenvalues
    return torch.sort(evals, descending=True)[0]  # Sort in descending order


def calculate_local_and_global_eigenvalues(euler1, euler2, euler3, window_size=5):
    # Ensure all inputs are PyTorch tensors and detach them if necessary
    device = euler1.device
    if euler1.shape != (256, 256) or euler2.shape != (256, 256) or euler3.shape != (256, 256):
        raise ValueError("Input arrays must have shape (256, 256)")
    
    # Output tensor for local eigenvalues, ensuring it is on the correct device
    local_eigenvalues = torch.zeros((256, 256, 3), dtype=torch.float32, device=device)
    
    # Pad the input tensors to handle edge cases for local calculation
    pad = window_size // 2
    euler1_pad = torch.nn.functional.pad(euler1.unsqueeze(0), (pad, pad, pad, pad), mode='replicate').squeeze(0)
    euler2_pad = torch.nn.functional.pad(euler2.unsqueeze(0), (pad, pad, pad, pad), mode='replicate').squeeze(0)
    euler3_pad = torch.nn.functional.pad(euler3.unsqueeze(0), (pad, pad, pad, pad), mode='replicate').squeeze(0)

    
    # Unit vector along c-axis (for hexagonal ice), ensuring it is on the correct device
    c_axis = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
    
    # # Calculate local eigenvalues
    # for i in range(256):
    #     for j in range(256):
    #         # Extract local window
    #         e1_local = euler1_pad[i:i+window_size, j:j+window_size].flatten()
    #         e2_local = euler2_pad[i:i+window_size, j:j+window_size].flatten()
    #         e3_local = euler3_pad[i:i+window_size, j:j+window_size].flatten()
            
    #         # Calculate c-axes orientations for all grains in the window
    #         c_axes = [euler_to_rotation_bunge(e1, e2, e3) @ c_axis 
    #                   for e1, e2, e3 in zip(e1_local, e2_local, e3_local)]
            
    #         # Calculate orientation tensor and eigenvalues
    #         A_local = calculate_orientation_tensor(c_axes)
    #         local_eigenvalues[i, j] = calculate_eigenvalues(A_local)
    
    # Calculate global eigenvalues
    global_c_axes = [euler_to_rotation_bunge(e1, e2, e3) @ c_axis 
                     for e1, e2, e3 in zip(euler1.flatten(), euler2.flatten(), euler3.flatten())]
    A_global = calculate_orientation_tensor(global_c_axes)
    global_eigenvalues = calculate_eigenvalues(A_global)
    return local_eigenvalues, global_eigenvalues


def euler_to_orientation_tensor(phi1, Phi, phi2):
    """
    Calculate orientation tensor and its eigenvalues from Euler angles (Bunge convention) using PyTorch tensors.
    
    Parameters:
    phi1, Phi, phi2: torch tensors of Euler angles in degrees
        phi1: first rotation around Z ([-180, 180])
        Phi: rotation around X' ([0, 90])
        phi2: second rotation around Z' ([-180, 180])
    
    Returns:
    eigenvalues: sorted eigenvalues of the orientation tensor (λ1 ≥ λ2 ≥ λ3)
    orientation_tensor: 3x3 orientation tensor
    orientation tensor in convention of:
    xx, xy, xz
    yx, yy, yz
    zx, zy, zz
    """
    # CRSS order must be consistent with A11,A12, A13... order in computing stress tensor!
    # current order is xx,yy,xy
    CRSS = torch.tensor([[20/41*3, 0.0, 0.0 ],[0.0, 1.0/41*3, 0.0],[0.0, 0.0, 20.0/41*3]]) # [7.0/19*1.7, 0.0, 0.0 ],[0.0, 10.0/19*1.7, 0.0],[0.0, 0.0, 1.0/19*1.7]
    # Convert angles from degrees to radians
    phi1_rad = torch.deg2rad(phi1)
    Phi_rad = torch.deg2rad(Phi)
    phi2_rad = torch.deg2rad(phi2)
    
    # Calculate direction cosines (c-axis direction)
    c_axes = torch.zeros((phi1.shape[0], 3), device=phi1.device)
    
    c1, s1 = torch.cos(phi1_rad), torch.sin(phi1_rad)
    c2, s2 = torch.cos(Phi_rad), torch.sin(Phi_rad)
    c3, s3 = torch.cos(phi2_rad), torch.sin(phi2_rad)
    
    # R11 = c1 * c3 - s1 * c2 * s3
    # R12 = -c1 * s3 - s1 * c2 * c3
    R13 = s2 * c1 # s1 * s2
    # R21 = s1 * c3 + c1 * c2 * s3
    # R22 = -s1 * s3 + c1 * c2 * c3
    R23 = s1 * s2 #c1 * s2 
    # R31 = s2 * s3
    # R32 = s2 * c3
    R33 = c2

    # Transform [0, 0, 1] by rotation matrix
    c_axes[:, 0] = R13
    c_axes[:, 1] = R23
    c_axes[:, 2] = R33
    # Calculate second order orientation tensor: average of outer products of c-axis directions
    orientation_tensor = torch.zeros((3, 3), device=phi1.device)
    Rotation = torch.zeros((3,3), device=phi1.device) # rotate by the angle between principle c and z
    N = phi1.shape[0]
    
    for i in range(3):
        for j in range(3):
            orientation_tensor[i, j] = torch.sum(c_axes[:, i] * c_axes[:, j]) / N
    scale = 1/(orientation_tensor[0,0] + orientation_tensor[1,1] + orientation_tensor[2,2])
    orientation_tensor *= scale
    # Calculate eigenvalues
    # first, second, third column of eigenvectors correspond to principle x y z axes
    eigenvalues, eigenvectors = torch.linalg.eigh(orientation_tensor)
    # weakening = torch.mm(torch.mm(eigenvectors,CRSS),eigenvectors.T)
    print(f'orientation_tensor: {orientation_tensor}')
    print(f'eigenvalues: {eigenvalues}')
    print(f'eigenvectors: {eigenvectors}')
    # print(f'weakening: {weakening}')
    principle_c_in_xz_cos = eigenvectors[2,0]/torch.sqrt(eigenvectors[0,0]**2+eigenvectors[2,0]**2)
    principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
        principle_c_in_xz *= -1
    elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
        principle_c_in_xz = np.pi - principle_c_in_xz
    Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2
    weakening1 = torch.mm(torch.mm(Rotation.T,CRSS),Rotation)
    # print(f'angle 1: {principle_c_in_xz*180/np.pi}')
    principle_c_in_xz_cos = eigenvectors[2,1]/torch.sqrt(eigenvectors[0,1]**2+eigenvectors[2,1]**2)
    principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
        principle_c_in_xz *= -1
    elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
        principle_c_in_xz = np.pi - principle_c_in_xz
    Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2  
    weakening2 = torch.mm(torch.mm(Rotation.T,CRSS),Rotation)
    # print(f'angle 2: {principle_c_in_xz*180/np.pi}')
    principle_c_in_xz_cos = eigenvectors[2,2]/torch.sqrt(eigenvectors[0,2]**2+eigenvectors[2,2]**2)
    principle_c_in_xz = torch.arccos(principle_c_in_xz_cos)
    if (eigenvectors[0,0]<0 and eigenvectors[2,0]>0):
        principle_c_in_xz *= -1
    elif (eigenvectors[0,0]<0 and eigenvectors[2,0]<0):
        principle_c_in_xz = np.pi - principle_c_in_xz
    Rotation[0,0] =  torch.cos(principle_c_in_xz)**2
    Rotation[0,1] =  torch.sin(principle_c_in_xz)**2
    Rotation[0,2] = 2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[1,0] =  torch.sin(principle_c_in_xz)**2
    Rotation[1,1] =  torch.cos(principle_c_in_xz)**2
    Rotation[1,2] = -2*torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,0] = -torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,1] =  torch.sin(principle_c_in_xz)*torch.cos(principle_c_in_xz)
    Rotation[2,2] =  torch.cos(principle_c_in_xz)**2 - torch.sin(principle_c_in_xz)**2
    weakening3 = torch.mm(torch.mm(Rotation.T,CRSS),Rotation)
    # eigenvalues = eigenvalues.flip(dims=(0,))  # Sort in descending order
    # scale = np.array([weakening[0,0],weakening[1,1],weakening[2,2]]).min()
    # weakening[0,:] = weakening[0,:]/weakening[0,0]
    # weakening[1,:] = weakening[1,:]/weakening[1,1]
    # weakening[2,:] = weakening[2,:]/weakening[2,2]
    # print(f'angle 3: {principle_c_in_xz*180/np.pi}')
    return eigenvalues, 1*(eigenvalues[0]*weakening1 + eigenvalues[1]*weakening2 +  eigenvalues[2]*weakening3) # 


def f1(lambda_val, n=2):
     return 1 - 3 * (lambda_val - 1/3)**n

def f2(lambda_i, lambda_j, m=1):
    return 1 - torch.abs(lambda_i - lambda_j)**1.2

def woodcock(lambda_1, lambda_2, lambda_3):
    return torch.sqrt(torch.log(lambda_3/lambda_2)/torch.log(lambda_2/lambda_1))

def get_orient_tensor2D(euler1, euler2):
    # euler angles in shape of: [grid_reso * grid_reso]
    euler1 = torch.deg2rad(euler1)
    euler2 = torch.deg2rad(euler2)
    
    cx = torch.sin(euler1) * torch.cos(euler2)
    # cy = torch.sin(euler1) * torch.sin(euler2)
    cz = torch.cos(euler1)

    axx = torch.mean(cx*cx)
    axz = torch.mean(cx*cz)
    azz = torch.mean(cz*cz)
    scale = 1/(axx+azz)
    # print(f'axx: {axx}, axz:{axz}, azz:{azz}, scale: {1/(axx+azz)}')
    # exit()
    return axx*scale,axz*scale,azz*scale

def get_orient_tensor1D(euler1, euler2):
    euler1 = torch.deg2rad(euler1)
    euler2 = torch.deg2rad(euler2)
    
    cx = torch.sin(euler1) * torch.cos(euler2)
    # cy = torch.sin(euler1) * torch.sin(euler2)
    cz = torch.cos(euler1)
    
    axx = cx*cx
    axz = cx*cz
    azz = cz*cz
    scale = 1/(axx+azz)
    return axx,axz,azz


