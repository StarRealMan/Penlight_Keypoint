import tensorflow as tf
import cv2
import numpy as np
import os
from datetime import datetime


class PredictionVisualizationCallback(tf.keras.callbacks.Callback):
    """
    训练过程中保存预测结果和loss可视化的回调函数
    """
    
    def __init__(self, validation_data, save_dir="training_predictions", save_frequency=5, num_samples=10):
        super().__init__()
        self.validation_data = validation_data
        self.save_dir = save_dir
        self.save_frequency = save_frequency  # 每多少个epoch保存一次
        self.num_samples = num_samples
        self.epoch_losses = []
        self.batch_losses = []
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        self.predictions_dir = os.path.join(save_dir, "predictions")
        self.loss_dir = os.path.join(save_dir, "loss_visualization")
        os.makedirs(self.predictions_dir, exist_ok=True)
        os.makedirs(self.loss_dir, exist_ok=True)
        
        # 准备验证样本
        self.val_samples = self._prepare_validation_samples()
        
    def _prepare_validation_samples(self):
        """准备用于可视化的验证样本"""
        samples = []
        count = 0
        for batch in self.validation_data:
            if count >= self.num_samples:
                break
            images, heatmaps = batch
            for i in range(min(images.shape[0], self.num_samples - count)):
                samples.append({
                    'image': images[i].numpy(),
                    'ground_truth': heatmaps[i].numpy(),
                    'sample_id': count
                })
                count += 1
                if count >= self.num_samples:
                    break
        return samples
    
    def on_epoch_end(self, epoch, logs=None):
        """在每个epoch结束时调用"""
        current_loss = logs.get('loss', 0)
        val_loss = logs.get('val_loss', 0)
        
        self.epoch_losses.append({
            'epoch': epoch,
            'train_loss': current_loss,
            'val_loss': val_loss
        })
        
        # 每隔指定频率保存预测结果
        if (epoch + 1) % self.save_frequency == 0:
            self._save_predictions(epoch, current_loss, val_loss)
            self._save_loss_visualization(epoch)
    
    def on_batch_end(self, batch, logs=None):
        """在每个batch结束时记录loss"""
        if logs:
            self.batch_losses.append({
                'batch': len(self.batch_losses),
                'loss': logs.get('loss', 0)
            })
    
    def _save_predictions(self, epoch, train_loss, val_loss):
        """保存当前epoch的预测结果"""
        print(f"\nSaving predictions for epoch {epoch + 1}...")
        
        epoch_dir = os.path.join(self.predictions_dir, f"epoch_{epoch+1:03d}")
        os.makedirs(epoch_dir, exist_ok=True)
        
        for sample in self.val_samples:
            # 获取模型预测
            image_batch = np.expand_dims(sample['image'], axis=0)
            prediction = self.model.predict(image_batch, verbose=0)[0]
            
            # 计算当前样本的loss
            sample_loss = tf.keras.losses.mse(
                sample['ground_truth'], 
                prediction
            ).numpy()
            
            # 确保sample_loss是标量值
            if isinstance(sample_loss, np.ndarray):
                sample_loss = float(sample_loss.mean())
            else:
                sample_loss = float(sample_loss)
            
            # 创建可视化
            vis_image = self._create_prediction_visualization(
                sample['image'], 
                sample['ground_truth'], 
                prediction,
                sample_loss,
                train_loss,
                val_loss,
                epoch
            )
            
            # 保存
            filename = f"sample_{sample['sample_id']:02d}_loss_{sample_loss:.4f}.jpg"
            save_path = os.path.join(epoch_dir, filename)
            cv2.imwrite(save_path, vis_image)
        
        print(f"Predictions saved to: {epoch_dir}")
    
    def _create_prediction_visualization(self, image, ground_truth, prediction, sample_loss, train_loss, val_loss, epoch):
        """创建预测结果的可视化"""
        # 转换图像到显示格式
        display_image = (image * 255).astype(np.uint8)
        display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        
        # 处理热力图
        gt_heatmap = ground_truth[:, :, 0]
        pred_heatmap = prediction[:, :, 0]
        
        # 归一化热力图到0-255
        gt_vis = (gt_heatmap * 255).astype(np.uint8)
        pred_vis = (pred_heatmap * 255).astype(np.uint8)
        
        # 应用颜色映射
        gt_colored = cv2.applyColorMap(gt_vis, cv2.COLORMAP_JET)
        pred_colored = cv2.applyColorMap(pred_vis, cv2.COLORMAP_JET)
        
        # 创建叠加图
        alpha = 0.6
        gt_overlay = cv2.addWeighted(display_image, alpha, gt_colored, 1-alpha, 0)
        pred_overlay = cv2.addWeighted(display_image, alpha, pred_colored, 1-alpha, 0)
        
        # 找到关键点位置
        gt_y, gt_x = np.unravel_index(np.argmax(gt_heatmap), gt_heatmap.shape)
        pred_y, pred_x = np.unravel_index(np.argmax(pred_heatmap), pred_heatmap.shape)
        
        # 在叠加图上标记关键点
        cv2.circle(gt_overlay, (gt_x, gt_y), 5, (255, 255, 255), 2)
        cv2.circle(gt_overlay, (gt_x, gt_y), 3, (0, 255, 0), -1)
        
        cv2.circle(pred_overlay, (pred_x, pred_y), 5, (255, 255, 255), 2)
        cv2.circle(pred_overlay, (pred_x, pred_y), 3, (0, 0, 255), -1)
        
        # 调整大小用于拼接
        display_size = (192, 192)
        display_image = cv2.resize(display_image, display_size)
        gt_colored = cv2.resize(gt_colored, display_size)
        pred_colored = cv2.resize(pred_colored, display_size)
        gt_overlay = cv2.resize(gt_overlay, display_size)
        pred_overlay = cv2.resize(pred_overlay, display_size)
        
        # 添加标题
        cv2.putText(display_image, "Original", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(gt_colored, "GT Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(pred_colored, "Pred Heatmap", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(gt_overlay, "GT Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(pred_overlay, "Pred Overlay", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 创建差异图
        diff_heatmap = np.abs(gt_heatmap - pred_heatmap)
        diff_vis = (diff_heatmap * 255).astype(np.uint8)
        diff_colored = cv2.applyColorMap(diff_vis, cv2.COLORMAP_HOT)
        diff_colored = cv2.resize(diff_colored, display_size)
        cv2.putText(diff_colored, "Difference", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 拼接成2x3网格
        top_row = np.hstack([display_image, gt_colored, pred_colored])
        bottom_row = np.hstack([gt_overlay, pred_overlay, diff_colored])
        result = np.vstack([top_row, bottom_row])
        
        # 添加信息文本
        info_height = 80
        info_img = np.zeros((info_height, result.shape[1], 3), dtype=np.uint8)
        
        # 添加loss信息
        cv2.putText(info_img, f"Epoch: {epoch+1}", (20, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(info_img, f"Sample Loss: {sample_loss:.6f}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(info_img, f"Train Loss: {train_loss:.6f}", (300, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"Val Loss: {val_loss:.6f}", (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 添加关键点坐标信息
        cv2.putText(info_img, f"GT: ({gt_x},{gt_y})", (500, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(info_img, f"Pred: ({pred_x},{pred_y})", (500, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # 计算距离误差
        distance_error = np.sqrt((gt_x - pred_x)**2 + (gt_y - pred_y)**2)
        cv2.putText(info_img, f"Distance Error: {distance_error:.2f}px", (700, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        final_result = np.vstack([info_img, result])
        return final_result
    
    def _save_loss_visualization(self, epoch):
        """保存loss曲线图"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建loss曲线图
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Epoch loss曲线
            epochs = [x['epoch'] + 1 for x in self.epoch_losses]
            train_losses = [x['train_loss'] for x in self.epoch_losses]
            val_losses = [x['val_loss'] for x in self.epoch_losses]
            
            ax1.plot(epochs, train_losses, 'b-', label='Training Loss', linewidth=2)
            ax1.plot(epochs, val_losses, 'r-', label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Batch loss曲线（最近1000个batch）
            if len(self.batch_losses) > 0:
                recent_batches = self.batch_losses[-1000:]
                batch_nums = [x['batch'] for x in recent_batches]
                batch_loss_vals = [x['loss'] for x in recent_batches]
                
                ax2.plot(batch_nums, batch_loss_vals, 'g-', alpha=0.7, linewidth=1)
                ax2.set_xlabel('Batch')
                ax2.set_ylabel('Loss')
                ax2.set_title('Recent Batch Loss (Last 1000 batches)')
                ax2.grid(True)
            
            plt.tight_layout()
            
            # 保存图像
            loss_plot_path = os.path.join(self.loss_dir, f"loss_curves_epoch_{epoch+1:03d}.png")
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"Loss visualization saved to: {loss_plot_path}")
            
        except ImportError:
            print("Matplotlib not available, skipping loss visualization")
    
    def on_train_end(self, logs=None):
        """训练结束时保存最终的loss信息"""
        print("\nSaving final training summary...")
        
        # 保存loss数据到文件
        import json
        
        summary = {
            'epoch_losses': self.epoch_losses,
            'final_train_loss': self.epoch_losses[-1]['train_loss'] if self.epoch_losses else 0,
            'final_val_loss': self.epoch_losses[-1]['val_loss'] if self.epoch_losses else 0,
            'total_epochs': len(self.epoch_losses),
            'total_batches': len(self.batch_losses)
        }
        
        summary_path = os.path.join(self.save_dir, "training_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"Training summary saved to: {summary_path}")
        
        # 最终的loss可视化
        if len(self.epoch_losses) > 0:
            self._save_loss_visualization(len(self.epoch_losses) - 1)
