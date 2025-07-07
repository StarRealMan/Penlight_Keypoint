import numpy as np

def generate_heatmap(keypoints, image_size=(192, 192), sigma=4):
    """
    生成热力图
    Args:
        keypoints: 关键点坐标 [x, y, v] 或 None (负样本)
        image_size: 图像尺寸 (height, width)
        sigma: 高斯核标准差
    
    Returns:
        heatmap: 热力图，形状为 (height, width, 1)
    """
    heatmaps = np.zeros((image_size[0], image_size[1], 1), dtype=np.float32)
    
    # 如果keypoints为None，返回全零热力图（负样本）
    if keypoints is None:
        return heatmaps
    
    x, y, v = keypoints[0], keypoints[1], keypoints[2]
    if v > 0:
        xx, yy = np.meshgrid(np.arange(image_size[1]), np.arange(image_size[0]))
        heatmaps[:, :, 0] = np.exp(-((xx - x)**2 + (yy - y)**2) / (2 * sigma**2))
    
    return heatmaps