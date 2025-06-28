import numpy as np

def generate_heatmap(keypoints, image_size=(256, 256), sigma=4, num_keypoints=1):
    heatmaps = np.zeros((image_size[0], image_size[1], num_keypoints), dtype=np.float32)
    for idx in range(num_keypoints):
        x, y, v = keypoints[idx*3:idx*3+3]
        if v > 0:
            xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
            heatmaps[:, :, idx] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    return heatmaps