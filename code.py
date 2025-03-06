import os

print("Current Working Directory:", os.getcwd())
from glob import glob

# File paths
detection_path = r"C:\Users\Jayanth\Downloads\Licplatesdetection_train\license_plates_detection_train"
recognition_path = r"C:\Users\Jayanth\Downloads\Licplatesrecognition_train\license_plates_recognition_train"
test_path = r"C:\Users\Jayanth\Downloads\test\test\test"

detection_images = glob(detection_path + "/*.jpg")
recognition_images = glob(recognition_path + "/*.jpg")
test_images = glob(test_path + "/*.jpg")

print(f"Detection Images: {len(detection_images)}")
print(f"Recognition Images: {len(recognition_images)}")
print(f"Test Images: {len(test_images)}")
# output
Detection Images: 900
Recognition Images: 900
Test Images: 210
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob

# Set image size (choose based on model needs)
IMG_SIZE = (128, 128)
def preprocess_image(image_path, grayscale=False):
    """Preprocess an image by resizing, converting to grayscale (optional), and normalizing."""
    img = cv2.imread(image_path)  # Load image
    img = cv2.resize(img, IMG_SIZE)  # Resize image
    
    if grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    
    img = img / 255.0  # Normalize pixel values to range [0, 1]
    
    return img
  # Load and preprocess detection images (color)
detection_images_processed = [preprocess_image(img) for img in detection_images]

# Load and preprocess recognition images (grayscale)
recognition_images_processed = [preprocess_image(img, grayscale=True) for img in recognition_images]

# Load and preprocess test images (color)
test_images_processed = [preprocess_image(img) for img in test_images]

print("Preprocessing complete!")
#output
Preprocessing complete!

def show_images(image_list, title, grayscale=False):
    """Display a few preprocessed images."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 5))
    for i, img in enumerate(image_list[:5]):  # Show first 5 images
        if grayscale:
            axes[i].imshow(img, cmap="gray")
        else:
            axes[i].imshow(img)
        axes[i].axis("off")
    plt.suptitle(title)
    plt.show()

# Display preprocessed images
show_images(detection_images_processed, "Detection Images (Resized & Normalized)")
show_images(recognition_images_processed, "Recognition Images (Grayscale & Normalized)", grayscale=True)
show_images(test_images_processed, "Test Images (Resized & Normalized)")

import pandas as pd

# Define file paths (update paths as needed)
detection_annotations_path = r"C:\Users\Jayanth\Downloads\Licplatesdetection_train.csv"
recognition_annotations_path = r"C:\Users\Jayanth\Downloads\Licplatesrecognition_train.csv"

# Load the CSV files
detection_df = pd.read_csv(detection_annotations_path)
recognition_df = pd.read_csv(recognition_annotations_path)

# Display first few rows of each dataset
print("🔹 License Plate Detection Annotations:")
print(detection_df.head())

print("\n🔹 License Plate Recognition Annotations:")
print(recognition_df.head())
# output
🔹 License Plate Detection Annotations:
    img_id  ymin  xmin  ymax  xmax
0    1.jpg   276    94   326   169
1   10.jpg   311   395   344   444
2  100.jpg   406   263   450   434
3  101.jpg   283   363   315   494
4  102.jpg   139    42   280   222

🔹 License Plate Recognition Annotations:
    img_id      text
0    0.jpg  117T3989
1    1.jpg  128T8086
2   10.jpg   94T3458
3  100.jpg  133T6719
4  101.jpg   68T5979
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import os

# Define paths 
image_folder = r"C:\Users\Jayanth\Downloads\Licplatesdetection_train\license_plates_detection_train"
annotation_file = r"C:\Users\Jayanth\Downloads\Licplatesdetection_train.csv"

# Load annotations
detection_df = pd.read_csv(annotation_file)

# Function to display an image with a bounding box
def show_sample_image(image_name, bbox):
    img_path = os.path.join(image_folder, image_name)
    
    # Read the image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Image {image_name} not found!")
        return
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for matplotlib

    # Get bounding box coordinates
    ymin, xmin, ymax, xmax = bbox

    # Draw bounding box
    cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 3)  # Blue box

    # Display image
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title(f"License Plate: {image_name}")
    plt.axis("off")
    plt.show()

# Show a few sample images with bounding boxes
for i in range(3):  # Display 3 images
    row = detection_df.iloc[i]
    show_sample_image(row["img_id"], (row["ymin"], row["xmin"], row["ymax"], row["xmax"]))
