import cv2
import os
import argparse
from pathlib import Path
from tqdm import tqdm

def extract_frames_from_video(video_path, output_dir, frame_interval=1, image_format='jpg'):
    base_name = os.path.basename(video_path).split('.')[0]
    output_dir = os.path.join(output_dir, base_name)
    os.makedirs(output_dir, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc=f"Processing {base_name}", unit="frame")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        pbar.update(1)

        if frame_count % frame_interval == 0:
            filename = f"frame_{frame_count:06d}.{image_format}"
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, frame)

            breakpoint()

        frame_count += 1
    
    cap.release()
    return True

def preprocess_videos(video_path=None, output_dir="images", frame_interval=1, image_format='jpg'):
    if os.path.isfile(video_path):
        extract_frames_from_video(video_path, output_dir, frame_interval, image_format)
    
    elif os.path.isdir(video_path):
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
        # video_files = []
        # for ext in video_extensions:
        #     video_files.extend(Path(video_path).glob(f"*{ext}"))
        #     video_files.extend(Path(video_path).glob(f"*{ext.upper()}"))
        video_files = []
        for video_file in ["/root/Penlight_Keypoint/training/1.MOV", 
                           "/root/Penlight_Keypoint/training/2.MOV", 
                           "/root/Penlight_Keypoint/training/3.mov", 
                           "/root/Penlight_Keypoint/training/4.mov", 
                           "/root/Penlight_Keypoint/training/7.MOV"
                        ]:
            extract_frames_from_video(str(video_file), output_dir, frame_interval, image_format)
    
    else:
        current_dir = os.getcwd()
        preprocess_videos(video_path=current_dir, output_dir=output_dir, 
                         frame_interval=frame_interval, image_format=image_format)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_path', type=str, default='videos')
    parser.add_argument('--output_dir', type=str, default='images')
    parser.add_argument('--frame_interval', type=int, default=1)
    parser.add_argument('--image_format', type=str, default='PNG', choices=['jpg', 'png', 'bmp', 'jpeg', 'PNG', 'JPEG'])
    
    args = parser.parse_args()
    
    preprocess_videos(
        video_path=args.video_path,
        output_dir=args.output_dir,
        frame_interval=args.frame_interval,
        image_format=args.image_format
    )

if __name__ == "__main__":
    main()
