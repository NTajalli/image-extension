import os
import cv2
import fiftyone as fo
import fiftyone.zoo as foz

# Load the COCO 2017 dataset with the specified number of samples
train_dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="train",
    max_samples=20000,
)

# Define directories for saving the preprocessed images
input_dir = "coco_preprocessed/inputs"
output_dir = "coco_preprocessed/outputs"
os.makedirs(input_dir, exist_ok=True)
os.makedirs(output_dir, exist_ok=True)

def preprocess_image(image_path, input_dir, output_dir, base_filename):
    img = cv2.imread(image_path)
    if img is None:
        return
    
    # Convert the image from BGR to LAB color space
    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    
    h, w, _ = img_lab.shape
    if h < 256 or w < 256:
        return  # Skip images smaller than 256x256
    
    # Get the four corners
    corners = [
        (0, 0, 256, 256),
        (0, w - 256, 256, w),
        (h - 256, 0, h, 256),
        (h - 256, w - 256, h, w)
    ]
    
    for i, (y1, x1, y2, x2) in enumerate(corners):
        patch_lab = img_lab[y1:y2, x1:x2]
        input_patch_lab = patch_lab[64:192, 64:192]

        # Convert patches back to RGB for saving
        patch_rgb = cv2.cvtColor(patch_lab, cv2.COLOR_LAB2BGR)
        input_patch_rgb = cv2.cvtColor(input_patch_lab, cv2.COLOR_LAB2BGR)
        
        input_filename = os.path.join(input_dir, f"{base_filename}_corner{i}_input.png")
        output_filename = os.path.join(output_dir, f"{base_filename}_corner{i}_output.png")
        
        cv2.imwrite(input_filename, input_patch_rgb)
        cv2.imwrite(output_filename, patch_rgb)

# Process all images in the dataset
for sample in train_dataset:
    preprocess_image(sample.filepath, input_dir, output_dir, os.path.splitext(os.path.basename(sample.filepath))[0])
