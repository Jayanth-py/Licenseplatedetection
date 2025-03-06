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
