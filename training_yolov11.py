
from ultralytics import YOLO
import zipfile
import os
import shutil
import random
import cv2
import numpy as np

dataset_zip = r'My First Project.yolov11.zip' 
extract_path = 'raw_dataset'
save_dir = 'nano_50_614_16'

os.makedirs(save_dir, exist_ok=True)

if os.path.exists(extract_path):
    shutil.rmtree(extract_path)

os.makedirs(extract_path, exist_ok=True)

with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

print("Dataset berhasil diekstrak")

all_images = []
all_labels = []

for root, dirs, files_list in os.walk(extract_path):
    for file in files_list:
        if file.endswith(('.jpg', '.png', '.jpeg')):
            img_path = os.path.join(root, file)

            label_path = img_path.replace('images', 'labels')
            label_path = os.path.splitext(label_path)[0] + '.txt'

            if os.path.exists(label_path):
                all_images.append(img_path)
                all_labels.append(label_path)

print(f"Total data: {len(all_images)}")

data = list(zip(all_images, all_labels))
random.shuffle(data)

total = len(data)
train_split = int(0.7 * total)
val_split = int(0.2 * total)

train_data = data[:train_split]
val_data = data[train_split:train_split + val_split]
test_data = data[train_split + val_split:]

base_path = 'dataset_2kelas'

if os.path.exists(base_path):
    shutil.rmtree(base_path)

for split in ['train', 'val', 'test']:
    os.makedirs(f'{base_path}/{split}/images', exist_ok=True)
    os.makedirs(f'{base_path}/{split}/labels', exist_ok=True)

def copy_data(split_data, split_name):
    for img_path, label_path in split_data:
        shutil.copy(img_path, f'{base_path}/{split_name}/images/')
        shutil.copy(label_path, f'{base_path}/{split_name}/labels/')

copy_data(train_data, 'train')
copy_data(val_data, 'val')
copy_data(test_data, 'test')

selected_classes = {0: 0, 1: 1}
class_names = ['bulat_besar', 'bulat_kecil']

for split in ['train', 'val', 'test']:
    label_dir = f'{base_path}/{split}/labels'

    for file in os.listdir(label_dir):
        file_path = os.path.join(label_dir, file)

        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []

        for line in lines:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            class_id = int(parts[0])
            if class_id in selected_classes:
                new_lines.append(' '.join([str(selected_classes[class_id])] + parts[1:]) + '\n')

        if new_lines:
            with open(file_path, 'w') as f:
                f.writelines(new_lines)
        else:
            os.remove(file_path)

yaml_content = f"""
path: {os.path.abspath(base_path)}
train: train/images
val: val/images
test: test/images

nc: 2
names: {class_names}
"""

with open(f'{base_path}/data.yaml', 'w') as f:
    f.write(yaml_content)

model = YOLO('yolo11n.pt')

model.train(
    data=f'{base_path}/data.yaml',
    epochs=50,
    imgsz=640,
    batch=16,
    optimizer='SGD',
    lr0=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    name='Ynano_50_614_16',
    exist_ok=True,

    degrees=10,         
    translate=0.05,     
    scale=0.10,         
    fliplr=0.5,         
    hsv_h=0.01,         
    hsv_s=0.20,         
    hsv_v=0.20,         
    mosaic=0.30,        
    mixup=0.0,        
)

best_model_path = 'runs/detect/Ynano_50_614_16/weights/best.pt'
save_path = os.path.join(save_dir, 'best.pt')

if os.path.exists(best_model_path):
    shutil.copy(best_model_path, save_path)
    print("Model disimpan!")

model = YOLO(save_path)

model.predict(
    source=f'{base_path}/test/images',
    conf=0.55, 
    save=True
)
