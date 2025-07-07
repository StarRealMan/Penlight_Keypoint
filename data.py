import tensorflow as tf
import cv2
import json
import os
import numpy as np
from tqdm import tqdm
from utils import generate_heatmap

# json file in Datumaro format

def create_debug_visualization(img, heatmap, keypoints):
    """
    创建调试可视化图像，包含原图、热力图叠加和关键点标记
    """
    # 复制原图
    vis_img = img.copy()
    
    # 创建热力图叠加
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # 叠加热力图到原图（透明度混合）
    alpha = 0.6
    overlay = cv2.addWeighted(vis_img, alpha, heatmap_colored, 1-alpha, 0)
    
    # 检查是否为负样本（keypoints包含NaN值）
    if np.any(np.isnan(keypoints)):
        # 负样本标记
        cv2.putText(overlay, "NEGATIVE SAMPLE", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(overlay, "No keypoints detected", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    else:
        # 标记关键点位置
        x, y = int(keypoints[0]), int(keypoints[1])
        visibility = keypoints[2]
        
        if visibility > 0:  # 如果关键点可见
            # 绘制关键点
            cv2.circle(overlay, (x, y), 5, (255, 255, 255), 2)  # 白色外圈
            cv2.circle(overlay, (x, y), 3, (0, 255, 0), -1)     # 绿色填充
            
            # 添加坐标文本
            coord_text = f"({x},{y})"
            cv2.putText(overlay, coord_text, (x+10, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # 添加可见性信息
            vis_text = f"Vis: {visibility}"
            cv2.putText(overlay, vis_text, (x+10, y+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 在图像顶部添加信息
    cv2.putText(overlay, "Training Data Visualization", (10, 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return overlay


def apply_augmentations(img, keypoints, image_size):
    """
    应用数据增强
    Args:
        img: 输入图像 (H, W, C)
        keypoints: 关键点坐标 [x, y, v] 或 None (负样本)
        image_size: 图像尺寸 (height, width)
    
    Returns:
        augmented_img, augmented_keypoints
    """
    height, width = image_size
    aug_img = img.copy()
    aug_keypoints = keypoints.copy() if keypoints is not None and not np.any(np.isnan(keypoints)) else keypoints
    
    # 1. 水平翻转增强 (50%概率)
    if np.random.random() < 0.5:
        # 翻转图像
        aug_img = cv2.flip(aug_img, 1)  # 1表示水平翻转
        
        # 翻转关键点坐标 (只对有效关键点)
        if aug_keypoints is not None and not np.any(np.isnan(aug_keypoints)):
            aug_keypoints[0] = width - aug_keypoints[0]  # x坐标翻转
            # y坐标和可见性保持不变
    
    # 2. 水平裁剪增强 (30%概率)
    if np.random.random() < 0.3:
        # 随机选择裁剪比例 m (0-20%)
        crop_ratio = np.random.uniform(0, 0.20)
        
        # 检查关键点是否在左右20%范围内
        should_crop = True
        if aug_keypoints is not None and not np.any(np.isnan(aug_keypoints)):
            x_ratio = aug_keypoints[0] / width
            # 如果关键点在左右20%范围内，不进行裁剪
            if x_ratio <= 0.2 or x_ratio >= 0.8:
                should_crop = False
        
        if should_crop:
            # 计算裁剪区域
            crop_width = int(width * crop_ratio)
            start_x = crop_width
            end_x = width - crop_width
            
            # 裁剪图像
            aug_img = aug_img[:, start_x:end_x, :]
            
            # 调整关键点坐标
            if aug_keypoints is not None and not np.any(np.isnan(aug_keypoints)):
                aug_keypoints[0] = aug_keypoints[0] - start_x
                # 如果关键点超出裁剪区域，设为不可见
                if aug_keypoints[0] < 0 or aug_keypoints[0] >= (end_x - start_x):
                    aug_keypoints[2] = 0  # 设为不可见
            
            # 重新调整图像大小到目标尺寸
            aug_img = cv2.resize(aug_img, (width, height))
            
            # 调整关键点坐标到新尺寸
            if aug_keypoints is not None and not np.any(np.isnan(aug_keypoints)):
                scale_x = width / (end_x - start_x)
                aug_keypoints[0] = aug_keypoints[0] * scale_x
    
    return aug_img, aug_keypoints


def load_image_and_heatmap_with_aug(image_path, keypoints, original_width, original_height, image_size, enable_augmentation=True):
    """
    延迟加载图像并生成热力图的函数，包含数据增强
    """
    # 读取并调整图像大小
    img = cv2.imread(image_path.numpy().decode('utf-8'))
    if img is None:
        raise ValueError(f"无法读取图像: {image_path.numpy().decode('utf-8')}")
    
    img = cv2.resize(img, image_size)
    img = img.astype(np.float32) / 255.0
    
    # 处理keypoints为None的情况（负样本）
    if keypoints.numpy() is None or np.all(np.isnan(keypoints.numpy())):
        adjusted_keypoints = None
    else:
        # 调整关键点坐标到新尺寸
        adjusted_keypoints = keypoints.numpy().copy()
        adjusted_keypoints[0] = adjusted_keypoints[0] * image_size[1] / original_width.numpy()
        adjusted_keypoints[1] = adjusted_keypoints[1] * image_size[0] / original_height.numpy()
    
    # 根据参数决定是否应用数据增强
    if enable_augmentation:
        # 应用数据增强
        aug_img, aug_keypoints = apply_augmentations(img, adjusted_keypoints, image_size)
    else:
        # 不应用数据增强（用于验证集）
        aug_img, aug_keypoints = img, adjusted_keypoints
    
    # 生成热力图
    if aug_keypoints is None:
        # 负样本：生成全零热力图
        heatmap = generate_heatmap(None, image_size, sigma=10)
    else:
        # 正样本：根据增强后的关键点生成热力图
        heatmap = generate_heatmap(aug_keypoints, image_size, sigma=10)
    
    return aug_img, heatmap


def load_dataset(json_path, image_size=(192, 192), shuffle=1000, batch_size=32, enable_augmentation=True):
    with open(json_path, 'r') as f:
        annos = json.load(f)
        
    # 只存储路径和关键点信息，不加载图像
    data_info = []
    for idx, ann in tqdm(enumerate(annos), desc="准备数据信息"):
        # 处理keypoints为None的情况（负样本）
        if ann['keypoints'] is None:
            # 使用NaN值表示无效的关键点，这样TensorFlow可以处理
            keypoints = np.array([np.nan, np.nan, 0.0], dtype=np.float32)
        else:
            keypoints = np.array(ann['keypoints'], dtype=np.float32)
        
        data_info.append({
            'image_path': ann['image_path'],
            'keypoints': keypoints,
            'original_width': ann['original_width'],
            'original_height': ann['original_height']
        })

    # 记录数据集大小信息
    dataset_size = len(data_info)
    print(f"数据集大小: {dataset_size} 个样本")
    print(f"数据增强: {'启用' if enable_augmentation else '禁用'}")
    
    # 计算每个epoch的步数
    steps_per_epoch = dataset_size // batch_size
    if dataset_size % batch_size != 0:
        steps_per_epoch += 1
    print(f"每个epoch步数: {steps_per_epoch}")

    # 创建数据集生成器
    def data_generator():
        for info in data_info:
            yield (
                info['image_path'],
                info['keypoints'],
                info['original_width'],
                info['original_height']
            )

    # 创建TensorFlow数据集
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_types=(tf.string, tf.float32, tf.int32, tf.int32),
        output_shapes=((), (3,), (), ())
    )
    
    # 使用tf.py_function进行延迟加载
    def tf_load_image_and_heatmap(image_path, keypoints, original_width, original_height):
        img, heatmap = tf.py_function(
            func=lambda path, kp, ow, oh: load_image_and_heatmap_with_aug(path, kp, ow, oh, image_size, enable_augmentation),
            inp=[image_path, keypoints, original_width, original_height],
            Tout=[tf.float32, tf.float32]
        )
        
        # 设置形状信息
        img.set_shape((image_size[0], image_size[1], 3))
        heatmap.set_shape((image_size[0], image_size[1], 1))
        
        return img, heatmap
    
    # 应用延迟加载
    dataset = dataset.map(tf_load_image_and_heatmap, num_parallel_calls=tf.data.AUTOTUNE)
    
    # 每个epoch都重新shuffle数据
    if shuffle > 0:
        # 使用合理的shuffle buffer，确保数据充分打乱但不占用过多内存
        shuffle_buffer_size = min(shuffle, dataset_size)  # 不超过数据集大小
        dataset = dataset.shuffle(
            buffer_size=shuffle_buffer_size,
            reshuffle_each_iteration=True  # 每个epoch都重新shuffle
        )
        print(f"启用数据shuffle，buffer大小: {shuffle_buffer_size}，每个epoch重新shuffle")
    
    final_dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    
    # 返回数据集和步数信息，以便训练时使用
    return final_dataset
