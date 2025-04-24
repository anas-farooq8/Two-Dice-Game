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



