import cv2
import numpy as np
import tensorflow as tf
from model import build_pose_model
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def extract_keypoint_from_heatmap(heatmap, image_size=(192, 192), original_size=(1920, 1080)):
    """
    从热力图中提取关键点坐标
    """
    # 找到热力图中的最大值位置
    y, x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    
    # 获取置信度（最大值）
    confidence = heatmap[y, x]
    
    # 将坐标从模型输出尺寸缩放到原始图像尺寸
    x_original = int(x * original_size[1] / image_size[1])
    y_original = int(y * original_size[0] / image_size[0])
    
    return x_original, y_original, confidence


def preprocess_frame(frame, target_size=(192, 192)):
    """
    预处理视频帧以符合模型输入要求
    """
    # 调整大小
    frame_resized = cv2.resize(frame, target_size)
    # 归一化到[0,1]
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    # 添加批次维度
    frame_batch = np.expand_dims(frame_normalized, axis=0)
    
    return frame_batch


def draw_keypoint(image, x, y, confidence, color=(0, 255, 0), radius=8, thickness=3):
    """
    在图像上绘制关键点
    """
    # 绘制圆点
    cv2.circle(image, (x, y), radius, color, thickness)
    
    # 绘制置信度文本
    confidence_text = f"Conf: {confidence:.3f}"
    text_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
    
    # 计算文本位置，避免超出图像边界
    text_x = max(10, x - text_size[0] // 2)
    text_y = max(30, y - 20)
    if text_y < 30:
        text_y = y + 40
    
    # 绘制文本背景
    cv2.rectangle(image, (text_x - 5, text_y - 20), 
                  (text_x + text_size[0] + 5, text_y + 5), (0, 0, 0), -1)
    
    # 绘制文本
    cv2.putText(image, confidence_text, (text_x, text_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # 绘制坐标文本
    coord_text = f"({x}, {y})"
    coord_y = text_y + 25
    cv2.rectangle(image, (text_x - 5, coord_y - 20), 
                  (text_x + len(coord_text) * 10, coord_y + 5), (0, 0, 0), -1)
    cv2.putText(image, coord_text, (text_x, coord_y), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def save_heatmap_visualization(heatmap, frame, output_path, keypoint_coords=None):
    """
    保存热力图可视化结果
    """
    # 确保heatmap是2D的
    if len(heatmap.shape) == 3:
        heatmap = heatmap[:, :, 0]
    
    # 创建包含4个子图的图像：原图、热力图、叠加图、3D热力图
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 原始帧（调整大小到192x192用于显示）
    frame_resized = cv2.resize(frame, (192, 192))
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    axes[0, 0].imshow(frame_rgb)
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # 热力图
    im1 = axes[0, 1].imshow(heatmap, cmap='hot', interpolation='nearest')
    axes[0, 1].set_title('Heatmap')
    axes[0, 1].axis('off')
    plt.colorbar(im1, ax=axes[0, 1])
    
    # 叠加图：原图+热力图
    # 将热力图归一化到0-255并转换为彩色
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    heatmap_resized = cv2.resize(heatmap_colored, (frame.shape[1], frame.shape[0]))
    
    # 叠加（透明度混合）
    alpha = 0.6
    overlay = cv2.addWeighted(frame, alpha, heatmap_resized, 1-alpha, 0)
    overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    
    # 如果有关键点坐标，在叠加图上标记
    if keypoint_coords:
        x, y, conf = keypoint_coords
        cv2.circle(overlay_rgb, (x, y), 8, (255, 255, 255), 3)
        cv2.circle(overlay_rgb, (x, y), 6, (0, 255, 0), -1)
    
    axes[1, 0].imshow(overlay_rgb)
    axes[1, 0].set_title('Overlay (Frame + Heatmap)')
    axes[1, 0].axis('off')
    
    # 3D热力图
    x_mesh = np.arange(heatmap.shape[1])
    y_mesh = np.arange(heatmap.shape[0])
    X, Y = np.meshgrid(x_mesh, y_mesh)
    
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    surf = ax_3d.plot_surface(X, Y, heatmap, cmap='hot', alpha=0.8)
    ax_3d.set_title('3D Heatmap')
    ax_3d.set_xlabel('X')
    ax_3d.set_ylabel('Y')
    ax_3d.set_zlabel('Confidence')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_individual_heatmap(heatmap, output_path):
    """
    单独保存热力图为图像文件
    """
    # 确保heatmap是2D的
    if len(heatmap.shape) == 3:
        heatmap = heatmap[:, :, 0]
    
    # 归一化到0-255
    heatmap_normalized = (heatmap * 255).astype(np.uint8)
    
    # 应用颜色映射
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
    
    # 保存
    cv2.imwrite(output_path, heatmap_colored)


def process_video_with_keypoint_detection(model_path, video_path, output_dir):
    """
    处理视频，进行关键点检测并保存结果
    """
    # 加载模型
    print("Loading model...")
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return
    
    # 获取视频信息
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Video info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Resolution: {frame_width}x{frame_height}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    heatmap_dir = os.path.join(output_dir, "heatmaps")
    visualization_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(heatmap_dir, exist_ok=True)
    os.makedirs(visualization_dir, exist_ok=True)
    
    # 创建结果视频写入器（可选）
    output_video_path = os.path.join(output_dir, "output_with_keypoints.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_video = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))
    
    # 统计信息
    detection_count = 0
    confidence_threshold = 0.1  # 置信度阈值
    
    print(f"Processing video frames...")
    
    # 处理每一帧
    frame_idx = 0
    progress_bar = tqdm(total=total_frames, desc="Processing frames")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预处理帧
        processed_frame = preprocess_frame(frame, target_size=(192, 192))
        
        # 进行推理
        heatmap = model.predict(processed_frame, verbose=0)[0]  # 移除批次维度
        
        # 保存原始热力图
        heatmap_filename = f"heatmap_{frame_idx:06d}.jpg"
        heatmap_path = os.path.join(heatmap_dir, heatmap_filename)
        save_individual_heatmap(heatmap[:, :, 0], heatmap_path)
        
        # 提取关键点
        x, y, confidence = extract_keypoint_from_heatmap(
            heatmap[:, :, 0],  # 取第一个（也是唯一一个）关键点的热力图
            image_size=(192, 192),
            original_size=(frame_height, frame_width)
        )
        
        # 复制帧用于绘制
        frame_with_keypoint = frame.copy()
        
        # 如果置信度足够高，绘制关键点
        if confidence > confidence_threshold:
            # 根据置信度设置颜色（绿色到红色）
            if confidence > 0.7:
                color = (0, 255, 0)  # 绿色 - 高置信度
            elif confidence > 0.4:
                color = (0, 255, 255)  # 黄色 - 中等置信度
            else:
                color = (0, 165, 255)  # 橙色 - 低置信度
            
            draw_keypoint(frame_with_keypoint, x, y, confidence, color)
            detection_count += 1
        else:
            # 置信度太低，显示"No detection"
            cv2.putText(frame_with_keypoint, "No detection", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 添加帧信息
        info_text = f"Frame: {frame_idx+1}/{total_frames}"
        cv2.putText(frame_with_keypoint, info_text, (20, frame_height - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 保存帧
        frame_filename = f"frame_{frame_idx:06d}_detected.jpg"
        frame_path = os.path.join(output_dir, frame_filename)
        cv2.imwrite(frame_path, frame_with_keypoint)
        
        # 保存热力图可视化（每隔10帧保存一次以节省时间和空间）
        if frame_idx % 10 == 0:
            vis_filename = f"visualization_{frame_idx:06d}.png"
            vis_path = os.path.join(visualization_dir, vis_filename)
            keypoint_coords = (x, y, confidence) if confidence > confidence_threshold else None
            save_heatmap_visualization(heatmap[:, :, 0], frame, vis_path, keypoint_coords)
        
        # 写入视频
        out_video.write(frame_with_keypoint)
        
        # 保存热力图可视化结果
        heatmap_output_path = os.path.join(output_dir, f"heatmap_{frame_idx:06d}.png")
        save_heatmap_visualization(heatmap[:, :, 0], frame, heatmap_output_path, keypoint_coords=(x, y, confidence))
        
        # 更新进度
        progress_bar.update(1)
        frame_idx += 1
    
    # 清理
    cap.release()
    out_video.release()
    progress_bar.close()
    
    # 打印统计信息
    print(f"\nProcessing completed!")
    print(f"  - Total frames processed: {frame_idx}")
    print(f"  - Frames with detections: {detection_count}")
    print(f"  - Detection rate: {detection_count/frame_idx*100:.1f}%")
    print(f"  - Output frames saved to: {output_dir}")
    print(f"  - Heatmaps saved to: {heatmap_dir}")
    print(f"  - Visualizations saved to: {visualization_dir}")
    print(f"  - Output video saved to: {output_video_path}")


def test_model_on_sample_frames(model_path, video_path, num_samples=10):
    """
    在几帧样本上测试模型，用于快速验证
    """
    print("Testing model on sample frames...")
    
    # 加载模型
    model = tf.keras.models.load_model(model_path)
    
    # 创建测试输出目录
    test_output_dir = "/root/Penlight_Keypoint/quick_test_results"
    os.makedirs(test_output_dir, exist_ok=True)
    
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 选择均匀分布的帧进行测试
    sample_indices = np.linspace(0, total_frames-1, num_samples, dtype=int)
    
    print(f"Testing on frames: {sample_indices}")
    
    for i, frame_idx in enumerate(sample_indices):
        # 跳转到指定帧
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 预处理和推理
        processed_frame = preprocess_frame(frame)
        heatmap = model.predict(processed_frame, verbose=0)[0]
        
        # 提取关键点
        x, y, confidence = extract_keypoint_from_heatmap(
            heatmap[:, :, 0],
            original_size=(frame.shape[0], frame.shape[1])
        )
        
        print(f"Frame {frame_idx}: Keypoint at ({x}, {y}), confidence: {confidence:.3f}")
        
        # 保存这一帧的可视化结果
        vis_filename = f"test_frame_{frame_idx:06d}_visualization.png"
        vis_path = os.path.join(test_output_dir, vis_filename)
        keypoint_coords = (x, y, confidence) if confidence > 0.1 else None
        save_heatmap_visualization(heatmap[:, :, 0], frame, vis_path, keypoint_coords)
        
        # 保存原始热力图
        heatmap_filename = f"test_frame_{frame_idx:06d}_heatmap.jpg"
        heatmap_path = os.path.join(test_output_dir, heatmap_filename)
        save_individual_heatmap(heatmap[:, :, 0], heatmap_path)
    
    cap.release()
    print(f"\nQuick test results saved to: {test_output_dir}")


def main():
    # 配置路径
    MODEL_PATH = "model_checkpoints/model_epoch_30.h5"
    VIDEO_PATH = "/root/Penlight_Keypoint/videos/WIN_20250627_21_44_32_Pro.mp4"
    OUTPUT_DIR = "/root/Penlight_Keypoint/test_results"
    
    print("=== Penlight Keypoint Detection Test ===")
    print(f"Model: {MODEL_PATH}")
    print(f"Video: {VIDEO_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print()
    
    # 检查文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model directory not found at {MODEL_PATH}")
        return
    
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return
    
    # 选择测试模式
    print("Choose test mode:")
    print("1. Quick test (10 sample frames)")
    print("2. Full video processing")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        test_model_on_sample_frames(MODEL_PATH, VIDEO_PATH, num_samples=10)
    elif choice == "2":
        process_video_with_keypoint_detection(MODEL_PATH, VIDEO_PATH, OUTPUT_DIR)
    else:
        print("Invalid choice. Running quick test by default.")
        test_model_on_sample_frames(MODEL_PATH, VIDEO_PATH, num_samples=10)


if __name__ == "__main__":
    main()
