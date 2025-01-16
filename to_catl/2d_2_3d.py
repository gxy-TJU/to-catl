import RoboTwin.crop_image as crop
import Depth_Anything_V2_main.depth_estimate as depth_estimate
import zarr
import cv2
import numpy as np
import random
import time

image_path = 'RoboTwin/data/block_hammer_beat_L515/episode0/camera/color/head/59.png'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
text_prompt = 'background.objects.hammer.cube'

start_time = time.time()
valid_mask = crop.crop_image(image, text_prompt)
end_time = time.time()
print('grounding_time', end_time - start_time)

def farthest_point_sampling(points, n_samples):
    N, D = points.shape
    centroids = np.zeros((n_samples,))
    distances = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(n_samples):
        centroids[i] = farthest
        centroid = points[farthest, :]
        dist = np.sum((points - centroid) ** 2, axis=1)
        mask = dist < distances
        distances[mask] = dist[mask]
        farthest = np.argmax(distances)
    return points[centroids.astype(np.int32)]

# Flatten the valid_mask to get the coordinates of valid points
valid_points = np.column_stack(np.where(valid_mask))

start_time = time.time()
# Perform FPS sampling
if len(valid_points) > 1024:
    sampled_points = farthest_point_sampling(valid_points, 1024)
else:
    sampled_points = valid_points
    while len(sampled_points) < 1024:
        idx = np.random.choice(len(valid_points))
        sampled_points = np.vstack([sampled_points, valid_points[idx]])
end_time = time.time()

print("FPS sampling time:", end_time - start_time)
valid_mask = np.zeros_like(valid_mask)
valid_mask[sampled_points[:, 0], sampled_points[:, 1]] = 1

start_time = time.time()
pointcloud = depth_estimate.get_pointcloud(image, valid_mask)
end_time = time.time()
print('depth_estimate_time', end_time - start_time)

