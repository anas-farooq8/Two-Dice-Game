import os
import math
import random
import numpy as np
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt
import shutil
from PIL import Image
import ultralytics
from ultralytics import YOLO
ultralytics.checks()

# Create dataset directories
os.makedirs('datasets/train/images', exist_ok=True)
os.makedirs('datasets/valid/images', exist_ok=True)
os.makedirs('datasets/test/images', exist_ok=True)
os.makedirs('datasets/train/labels', exist_ok=True)
os.makedirs('datasets/valid/labels', exist_ok=True)
os.makedirs('datasets/test/labels', exist_ok=True)

# Paths for train, valid, and test sets
train_path = 'datasets/train/'
valid_path = 'datasets/valid/'
test_path = 'datasets/test/'

# Path to the dataset
# Replace with your actual path to the dice dataset
DATASET_PATH = '/kaggle/input/d6-dice/d6-dice'

# Get all annotation files
ano_paths = []
for dirname, _, filenames in os.walk(os.path.join(DATASET_PATH, 'Annotations')):
    for filename in filenames:
        if filename.endswith('.txt'):
            ano_paths.append(os.path.join(dirname, filename))

# Filter out annotations that don't have corresponding image files
valid_ano_paths = []
for ano_path in ano_paths:
    base_name = os.path.basename(ano_path)[0:-4]
    img_path = os.path.join(DATASET_PATH, 'Images', base_name + '.jpg')
    if os.path.exists(img_path):
        valid_ano_paths.append(ano_path)
    else:
        print(f"Skipping annotation without image: {base_name}")

n = len(valid_ano_paths)
print(f"Total number of valid annotations: {n}")
N = list(range(n))
random.shuffle(N)

# Split data according to specified ratios
train_ratio = 0.7
valid_ratio = 0.2
test_ratio = 0.1

train_size = int(train_ratio * n)
valid_size = int(valid_ratio * n)

train_i = N[:train_size]
valid_i = N[train_size:train_size+valid_size]
test_i = N[train_size+valid_size:]

print(f"Train set size: {len(train_i)}")
print(f"Validation set size: {len(valid_i)}")
print(f"Test set size: {len(test_i)}")

# Copy files to respective directories
def copy_files(indices, dest_path, prefix='Moving'):
    """Copy image and annotation files to destination directory."""
    success_count = 0
    for i in tqdm(indices, desc=prefix):
        ano_path = valid_ano_paths[i]
        base_name = os.path.basename(ano_path)[0:-4]
        img_path = os.path.join(DATASET_PATH, 'Images', base_name + '.jpg')
        
        # Skip if image doesn't exist (which shouldn't happen due to filtering)
        if not os.path.exists(img_path):
            print(f"Warning: Image not found: {img_path}")
            continue
            
        # Destination paths for the current file
        dest_img_path = os.path.join(dest_path, 'images', os.path.basename(img_path))
        dest_ano_path = os.path.join(dest_path, 'labels', os.path.basename(ano_path))
        
        try:
            shutil.copy(img_path, dest_img_path)
            shutil.copy(ano_path, dest_ano_path)
            success_count += 1
        except Exception as e:
            print(f"Error copying file {img_path}: {e}")
    
    return success_count

# Copy files to train, validation, and test directories
train_copied = copy_files(train_i, train_path, 'Moving training files')
valid_copied = copy_files(valid_i, valid_path, 'Moving validation files')
test_copied = copy_files(test_i, test_path, 'Moving test files')

# Print number of files in each directory
print(f"Files in train directory: {len(os.listdir(train_path + 'images'))}")
print(f"Files in validation directory: {len(os.listdir(valid_path + 'images'))}")
print(f"Files in test directory: {len(os.listdir(test_path + 'images'))}")
print(f"Successfully copied: {train_copied} training, {valid_copied} validation, {test_copied} test files")

# Create YAML configuration for YOLO
data_yaml = dict(
    train='datasets/train/images',
    val='datasets/valid/images',
    test='datasets/test/images',
    nc=6,
    names=['1', '2', '3', '4', '5', '6']
)

# Save YAML configuration
with open('data.yaml', 'w') as outfile:
    yaml.safe_dump(data_yaml, outfile, default_flow_style=False)

print("YAML configuration saved:")
with open('data.yaml', 'r') as f:
    print(f.read())

# Initialize YOLO model
model = YOLO('yolov8x.pt')

# Train the model
results = model.train(
    data='/kaggle/working/data.yaml',
    epochs=20,
    imgsz=640,
    batch=8,
    name='dice_detector'
)

# Save the trained model
model.export(format='onnx')
model.save('dice_detector.pt')
print("Model saved successfully")

# list your test images directory
test_images = os.listdir(os.path.join(test_path, 'images'))
if test_images:
    # pick up to 6 random images
    n_images = min(6, len(test_images))
    sample_images = random.sample(test_images, n_images)

    cols = 2
    rows = math.ceil(n_images / cols)
    plt.figure(figsize=(15, rows * 7))

    for i, img_file in enumerate(sample_images):
        img_path = os.path.join(test_path, 'images', img_file)
        results = model.predict(img_path, conf=0.25)

        # create a subplot
        ax = plt.subplot(rows, cols, i + 1)
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"Test {i+1}: {img_file}", fontsize=12)
        ax.axis('off')

        # iterate each detection in this image
        for res in results:
            # extract boxes, classes, and confidences
            boxes = res.boxes.xyxy.cpu().numpy()
            classes = res.boxes.cls.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()

            for box, cls, conf in zip(boxes, classes, confs):
                x1, y1, x2, y2 = box

                # unfilled bounding box
                rect = plt.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    fill=False, edgecolor='red', linewidth=2
                )
                ax.add_patch(rect)

                # label with class name and confidence
                label = model.names[int(cls)] if hasattr(model, 'names') else str(int(cls))
                ax.text(
                    x1, y1 - 5,
                    f"{label}, {conf:.2f}",
                    fontsize=10,
                    verticalalignment='bottom',
                    bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none')
                )

    plt.tight_layout()
    plt.show()

# Evaluate the model on test set
test_results = model.val(data='/kaggle/working/data.yaml', split='test')
print("Test Results:", test_results)




