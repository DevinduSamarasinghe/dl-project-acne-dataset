import os
import cv2
import albumentations as A
import numpy as np

# Define the augmentation pipeline
augmentation = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.5),
    A.Rotate(limit=30, p=0.5),
    # Add more augmentations as needed
], bbox_params=A.BboxParams(format='yolo'))

# Paths
input_image_dir = 'E:/My Stuff/SLIIT/Year 4/DL/valid/images'
input_label_dir = 'E:/My Stuff/SLIIT/Year 4/DL/valid/labels'
output_image_dir = 'E:/My Stuff/SLIIT/Year 4/DL/valid/aug_images'
output_label_dir = 'E:/My Stuff/SLIIT/Year 4/DL/valid/aug_labels'

if not os.path.exists(output_image_dir):
    os.makedirs(output_image_dir)
if not os.path.exists(output_label_dir):
    os.makedirs(output_label_dir)

# Function to read YOLO annotations
def read_yolo_annotation(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        bboxes = []
        for line in lines:
            values = line.strip().split()
            if len(values) >= 5:
                class_id = int(values[0])
                x_center, y_center, width, height = map(float, values[1:5])
                
                # Clip the values to be between 0 and 1 (ensures no invalid values)
                x_center = np.clip(x_center, 0, 1)
                y_center = np.clip(y_center, 0, 1)
                width = np.clip(width, 0, 1)
                height = np.clip(height, 0, 1)
                
                bboxes.append([x_center, y_center, width, height, class_id])
        return bboxes



# Function to save YOLO annotations
def save_yolo_annotation(file_path, bboxes):
    with open(file_path, 'w') as file:
        for bbox in bboxes:
            x_center, y_center, width, height, class_id = bbox
            file.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

# Function to ensure bounding boxes are valid before transformation
def ensure_valid_bbox(bboxes):
    valid_bboxes = []
    for bbox in bboxes:
        x_center, y_center, width, height, class_id = bbox
        
        # Ensure that the bounding box stays within [0, 1] range before augmentation
        x_min = x_center - width / 2
        x_max = x_center + width / 2
        y_min = y_center - height / 2
        y_max = y_center + height / 2
        
        # Clip the values to be within valid range
        x_min = np.clip(x_min, 0, 1)
        x_max = np.clip(x_max, 0, 1)
        y_min = np.clip(y_min, 0, 1)
        y_max = np.clip(y_max, 0, 1)
        
        # Recalculate the center and width/height after clipping
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        width = x_max - x_min
        height = y_max - y_min
        
        # Only keep boxes that have non-zero width and height
        if width > 0 and height > 0:
            valid_bboxes.append([x_center, y_center, width, height, class_id])
    
    return valid_bboxes

# Loop over the dataset and apply augmentations
for image_file in os.listdir(input_image_dir):
    if image_file.endswith('.jpg'):
        image_path = os.path.join(input_image_dir, image_file)
        label_path = os.path.join(input_label_dir, image_file.replace('.jpg', '.txt'))
        
        # Load the image
        image = cv2.imread(image_path)
        height, width, _ = image.shape
        
        # Load the corresponding annotations
        bboxes = read_yolo_annotation(label_path)
        
        # Ensure the bounding boxes are valid before augmentation
        bboxes = ensure_valid_bbox(bboxes)
        
        # Apply augmentation
        transformed = augmentation(image=image, bboxes=bboxes)
        augmented_image = transformed['image']
        augmented_bboxes = transformed['bboxes']
        
        # Clip the bounding boxes after augmentation to ensure they are within valid range
        clipped_bboxes = []
        for bbox in augmented_bboxes:
            x_center, y_center, box_width, box_height, class_id = bbox
            x_center = np.clip(x_center, 0, 1)
            y_center = np.clip(y_center, 0, 1)
            box_width = np.clip(box_width, 0, 1)
            box_height = np.clip(box_height, 0, 1)
            clipped_bboxes.append([x_center, y_center, box_width, box_height, class_id])
        
        # Save augmented image
        output_image_path = os.path.join(output_image_dir, 'aug_' + image_file)
        cv2.imwrite(output_image_path, augmented_image)
        
        # Save augmented annotations
        output_label_path = os.path.join(output_label_dir, 'aug_' + image_file.replace('.jpg', '.txt'))
        save_yolo_annotation(output_label_path, clipped_bboxes)


