#!/usr/bin/env python3
"""
分析训练结果的脚本
用于查看预测结果、loss曲线和训练统计信息
"""

import os
import json
import cv2
import numpy as np
import argparse
from glob import glob


def analyze_training_results(predictions_dir="training_predictions"):
    """
    分析训练预测结果
    """
    print(f"Analyzing training results from: {predictions_dir}")
    
    if not os.path.exists(predictions_dir):
        print(f"Error: Directory {predictions_dir} not found!")
        return
    
    # 读取训练摘要
    summary_path = os.path.join(predictions_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        print("\n=== Training Summary ===")
        print(f"Total epochs: {summary['total_epochs']}")
        print(f"Total batches: {summary['total_batches']}")
        print(f"Final training loss: {summary['final_train_loss']:.6f}")
        print(f"Final validation loss: {summary['final_val_loss']:.6f}")
        
        # 分析loss趋势
        epoch_losses = summary['epoch_losses']
        if len(epoch_losses) > 1:
            initial_train_loss = epoch_losses[0]['train_loss']
            final_train_loss = epoch_losses[-1]['train_loss']
            train_improvement = ((initial_train_loss - final_train_loss) / initial_train_loss) * 100
            
            initial_val_loss = epoch_losses[0]['val_loss']
            final_val_loss = epoch_losses[-1]['val_loss']
            val_improvement = ((initial_val_loss - final_val_loss) / initial_val_loss) * 100
            
            print(f"Training loss improvement: {train_improvement:.2f}%")
            print(f"Validation loss improvement: {val_improvement:.2f}%")
    
    # 分析预测目录
    predictions_subdir = os.path.join(predictions_dir, "predictions")
    if os.path.exists(predictions_subdir):
        epoch_dirs = sorted([d for d in os.listdir(predictions_subdir) 
                           if os.path.isdir(os.path.join(predictions_subdir, d))])
        
        print(f"\n=== Prediction Results ===")
        print(f"Found predictions for {len(epoch_dirs)} epochs")
        
        if epoch_dirs:
            print(f"Epochs with predictions: {', '.join([d.replace('epoch_', '') for d in epoch_dirs])}")
            
            # 分析最后一个epoch的预测结果
            last_epoch_dir = os.path.join(predictions_subdir, epoch_dirs[-1])
            prediction_files = glob(os.path.join(last_epoch_dir, "*.jpg"))
            
            print(f"\nLast epoch ({epoch_dirs[-1]}) analysis:")
            print(f"Number of samples: {len(prediction_files)}")
            
            # 提取loss信息
            losses = []
            for file_path in prediction_files:
                filename = os.path.basename(file_path)
                if "loss_" in filename:
                    try:
                        loss_str = filename.split("loss_")[1].split(".jpg")[0]
                        loss_val = float(loss_str)
                        losses.append(loss_val)
                    except (IndexError, ValueError):
                        continue
            
            if losses:
                avg_loss = np.mean(losses)
                min_loss = np.min(losses)
                max_loss = np.max(losses)
                std_loss = np.std(losses)
                
                print(f"Sample losses - Avg: {avg_loss:.6f}, Min: {min_loss:.6f}, Max: {max_loss:.6f}, Std: {std_loss:.6f}")
                
                # 找到最好和最差的样本
                best_idx = np.argmin(losses)
                worst_idx = np.argmax(losses)
                
                print(f"Best sample: {os.path.basename(prediction_files[best_idx])} (loss: {losses[best_idx]:.6f})")
                print(f"Worst sample: {os.path.basename(prediction_files[worst_idx])} (loss: {losses[worst_idx]:.6f})")
    
    # 检查loss可视化
    loss_vis_dir = os.path.join(predictions_dir, "loss_visualization")
    if os.path.exists(loss_vis_dir):
        loss_plots = glob(os.path.join(loss_vis_dir, "*.png"))
        print(f"\n=== Loss Visualizations ===")
        print(f"Found {len(loss_plots)} loss curve plots")
        if loss_plots:
            latest_plot = max(loss_plots, key=os.path.getctime)
            print(f"Latest plot: {os.path.basename(latest_plot)}")


def create_training_report(predictions_dir="training_predictions", output_path="training_report.html"):
    """
    创建HTML格式的训练报告
    """
    print(f"Creating training report: {output_path}")
    
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Training Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            .section { margin: 20px 0; border: 1px solid #ccc; padding: 15px; }
            .prediction-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 10px; }
            .prediction-item { text-align: center; }
            .prediction-item img { max-width: 100%; height: auto; }
            .loss-chart { text-align: center; margin: 20px 0; }
            table { border-collapse: collapse; width: 100%; }
            th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
            th { background-color: #f2f2f2; }
        </style>
    </head>
    <body>
        <h1>Penlight Keypoint Detection Training Report</h1>
    """
    
    # 添加训练摘要
    summary_path = os.path.join(predictions_dir, "training_summary.json")
    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        html_content += f"""
        <div class="section">
            <h2>Training Summary</h2>
            <table>
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Epochs</td><td>{summary['total_epochs']}</td></tr>
                <tr><td>Total Batches</td><td>{summary['total_batches']}</td></tr>
                <tr><td>Final Training Loss</td><td>{summary['final_train_loss']:.6f}</td></tr>
                <tr><td>Final Validation Loss</td><td>{summary['final_val_loss']:.6f}</td></tr>
            </table>
        </div>
        """
    
    # 添加loss可视化
    loss_vis_dir = os.path.join(predictions_dir, "loss_visualization")
    if os.path.exists(loss_vis_dir):
        loss_plots = sorted(glob(os.path.join(loss_vis_dir, "*.png")))
        if loss_plots:
            latest_plot = loss_plots[-1]
            relative_path = os.path.relpath(latest_plot, os.path.dirname(output_path))
            html_content += f"""
            <div class="section">
                <h2>Loss Curves</h2>
                <div class="loss-chart">
                    <img src="{relative_path}" alt="Loss Curves">
                </div>
            </div>
            """
    
    # 添加预测结果
    predictions_subdir = os.path.join(predictions_dir, "predictions")
    if os.path.exists(predictions_subdir):
        epoch_dirs = sorted([d for d in os.listdir(predictions_subdir) 
                           if os.path.isdir(os.path.join(predictions_subdir, d))])
        
        if epoch_dirs:
            # 显示最后几个epoch的结果
            for epoch_dir in epoch_dirs[-3:]:  # 最后3个epoch
                epoch_path = os.path.join(predictions_subdir, epoch_dir)
                prediction_files = sorted(glob(os.path.join(epoch_path, "*.jpg")))
                
                if prediction_files:
                    html_content += f"""
                    <div class="section">
                        <h2>Predictions - {epoch_dir.replace('_', ' ').title()}</h2>
                        <div class="prediction-grid">
                    """
                    
                    for pred_file in prediction_files[:6]:  # 显示前6个样本
                        relative_path = os.path.relpath(pred_file, os.path.dirname(output_path))
                        filename = os.path.basename(pred_file)
                        html_content += f"""
                        <div class="prediction-item">
                            <img src="{relative_path}" alt="{filename}">
                            <p>{filename}</p>
                        </div>
                        """
                    
                    html_content += "</div></div>"
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"Report saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--predictions_dir', type=str, default='training_predictions',
                       help='Directory containing training predictions')
    parser.add_argument('--create_report', action='store_true',
                       help='Create HTML training report')
    parser.add_argument('--report_output', type=str, default='training_report.html',
                       help='Output path for HTML report')
    
    args = parser.parse_args()
    
    # 分析训练结果
    analyze_training_results(args.predictions_dir)
    
    # 创建HTML报告
    if args.create_report:
        create_training_report(args.predictions_dir, args.report_output)


if __name__ == "__main__":
    main()
