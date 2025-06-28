import tensorflow as tf
import cv2, json
from utils import generate_heatmap

# json file in Datumaro format

def load_dataset(json_path, image_dir, image_size=(256, 256), shuffle=1000, batch_size=32, num_keypoints=1):
    with open(json_path, 'r') as f:
        anno = json.load(f)

    data = []
    for ann in anno['items']:
        if 'annotations' not in ann or len(ann['annotations']) == 0:
            continue
        
        image_name = ann['image']['path']
        img = cv2.imread(f"{image_dir}/{image_name}")
        img = cv2.resize(img, image_size)

        keypoints = ann['annotations'][0]['points']
        keypoints.append(ann['annotations'][0]['visibility'][0])
        heatmap = generate_heatmap(keypoints, image_size, num_keypoints=1)

        data.append((img / 255.0, heatmap))
        
    return tf.data.Dataset.from_generator(
        lambda: iter(data),
        output_types=(tf.float32, tf.float32),
        output_shapes=((image_size[0], image_size[1], 3), (image_size[0], image_size[1], num_keypoints))
    ).shuffle(shuffle).batch(batch_size).prefetch(tf.data.AUTOTUNE)
