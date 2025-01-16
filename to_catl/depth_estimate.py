import cv2
import torch
import numpy as np
import open3d as o3d
import os
from PIL import Image
import sys
sys.path.append('/home/catl/Desktop/Depth_Anything_V2_main')
from metric_depth.depth_anything_v2.dpt import DepthAnythingV2

#from depth_anything_v2.dpt import DepthAnythingV2

#DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
DEVICE = 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'
model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'Depth_Anything_V2_main/checkpoints/depth_anything_v2_metric_hypersim_vitl.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

def get_pointcloud(image, valid_mask, model=model, device=DEVICE):
    raw_img = image
    color_image = Image.fromarray(image)
    width, height = color_image.size

    depth = model.infer_image(raw_img) # HxW raw depth map in numpy
    import matplotlib.pyplot as plt
    
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - width / 2) / 217
    y = (y - height / 2) / 217
    # Apply the valid_mask to keep only the valid points
    z = np.array(depth)
    valid_mask = valid_mask.astype(bool)
    x = x[valid_mask]
    y = y[valid_mask]
    z = z[valid_mask]
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1)
    colors = np.array(color_image).reshape(-1, 3)[valid_mask.reshape(-1)] / 255.0
    # Create the point cloud and save it to the output directory
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    # Visualize the point cloud
    #o3d.visualization.draw_geometries([pcd])
    
if __name__ == '__main__':
    image_path = 'RoboTwin/data/block_hammer_beat_L515/episode0/camera/color/head/187.png'
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    valid_mask = np.random.choice([0, 1], size=image.shape[:2], p=[0.5, 0.5])
    pointcloud = get_pointcloud(image, valid_mask)
    