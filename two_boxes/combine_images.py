import cv2
import os
import re

def combine_images(folder1, folder2, folder3, output_folder):
    # Get a sorted list of image files from both folders
    def get_sorted_files(folder):
        files = [f for f in os.listdir(folder) if (f.endswith('.png') or f.endswith('.jpg')) ]
        files.sort(key=lambda x: int(re.search(r'\d+', x).group()))
        return files

    images1 = get_sorted_files(folder1)
    images2 = get_sorted_files(folder2)
    images3 = get_sorted_files(folder3)

    # print(images1)
    # print(images2)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Combine corresponding images
    for img1_name, img2_name, img3_name in zip(images1, images2, images3):
        path1 = os.path.join(folder1, img1_name)
        path2 = os.path.join(folder2, img2_name)
        path3 = os.path.join(folder3, img3_name )
        
        img1 = cv2.imread(path1)
        img2 = cv2.imread(path2)
        img3 = cv2.imread(path3)
        
        # Check that images have the same height
        img1 = cv2.resize(img1, (int(img1.shape[1] * img2.shape[0] / img1.shape[0]), img2.shape[0]))
        img3 = cv2.resize(img3, (int(img3.shape[1] * img2.shape[0] / img3.shape[0]), img2.shape[0]))
        
        # Combine images horizontally (side by side)
        combined_image = cv2.hconcat([img1, img2, img3])
        
        # Save the combined image
        output_path = os.path.join(output_folder, f"combined_{img1_name}")
        cv2.imwrite(output_path, combined_image)

    print(f"Combined images saved to {output_folder}")

# Example usage
combine_images(
    folder1="left_box_frames",
    folder2="3D_plots",
    folder3 = "right_box_frames", 
    output_folder="combined_images"
)
