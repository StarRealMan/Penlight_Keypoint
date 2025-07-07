import os
import json

base_path = "/root/Penlight_Keypoint/training"
negative_path = "/root/Penlight_Keypoint/training/negative_images"
images_path = "/root/Penlight_Keypoint/training/images"

if __name__ == "__main__":
    
    data = []

    # positive video
    for file_name in os.listdir(base_path):
        if file_name.endswith(".json"):
            vid_num = file_name.split(".")[0]
            json_path = os.path.join(base_path, file_name)
            video_path = os.path.join(base_path, str(vid_num) + ".mp4")
            with open(json_path, 'r') as f:
                anno = json.load(f)
            for idx, ann in enumerate(anno['items']):
                image_name = ann['image']['path']
                width = ann['image']['size'][1]
                height = ann['image']['size'][0]
                image_path = os.path.join(images_path, vid_num, image_name)
                
                if 'annotations' in ann and len(ann['annotations']) != 0:
                    keypoints = ann['annotations'][0]['points']
                    keypoints.append(ann['annotations'][0]['visibility'][0])
                else:
                    keypoints = None
                data.append({
                    "image_path": image_path,
                    "keypoints": keypoints,
                    "original_height": height,
                    "original_width": width
                })
    
    # negative video
    for neg_video_path in os.listdir(negative_path):
        neg_video_path = os.path.join(negative_path, neg_video_path)
        for neg_image in os.listdir(neg_video_path):
            image_name = os.path.join(neg_video_path, neg_image)
            data.append({
                "image_path": image_name,
                "keypoints": None,
                "original_height": 1080,
                "original_width": 1920
            })
    
    with open("data.json", 'w') as f:
        json.dump(data, f, indent=4)