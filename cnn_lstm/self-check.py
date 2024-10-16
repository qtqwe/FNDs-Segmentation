import os
import cv2
import numpy as np


def load_binary_masks(folder_path):
    """
    Load binary mask images from a specified folder and return them as a list of numpy arrays.

    Parameters:
    folder_path: str, the path to the image folder

    Returns:
    masks: list of numpy arrays, containing all binary mask images
    """
    masks = []
    for filename in os.listdir(folder_path):
        img_path = os.path.join(folder_path, filename)
        if filename.endswith((".png", ".jpg", ".jpeg", ".bmp")):
            mask = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            masks.append((mask > 0).astype(int))  # Ensure mask is 0 and 1
    return masks


def self_check_mechanism_on_masks(masks, threshold=0.5):
    """
    Perform the self-check mechanism on a set of binary mask images to verify the authenticity of FL spots
    and generate a new mask image.

    Parameters:
    masks: list of numpy arrays, containing all binary mask images.
    threshold: float, the threshold for the occurrence ratio of FL spots in subsequences (default is 0.5).

    Returns:
    cleaned_masks: list of numpy arrays, processed image data with false signals removed.
    final_mask: numpy array, generated final mask image.
    is_blank: bool, whether the final image is blank.
    """
    predictions = np.stack(masks, axis=0)
    is_blank = True
    cleaned_predictions = predictions.copy()
    final_mask = np.zeros_like(predictions[0])

    while is_blank:
        appearance_count = np.sum(cleaned_predictions, axis=0)
        num_masks = cleaned_predictions.shape[0]

        genuine_mask = (appearance_count > (threshold * num_masks)).astype(int)

        if np.sum(genuine_mask) == 0:
            is_blank = True
        else:
            is_blank = False

        # Remove false signals and generate the final mask
        cleaned_predictions = np.tile(genuine_mask, (num_masks, 1, 1)) * predictions
        final_mask = genuine_mask

    return cleaned_predictions, final_mask, is_blank


def save_final_mask(final_mask, output_path):
    """
    Save the final mask image.

    Parameters:
    final_mask: numpy array, the generated final mask image.
    output_path: str, the path to save the image.
    """
    # Ensure the output folder exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, (final_mask * 255).astype(np.uint8))
    print(f"Final mask image saved at {output_path}")


# 使用示例
input_folder = "/media/bio/LW/new/regular/test_results"
# input_folder = "/media/bio/LW/new/sensitive/test_results"
output_folder = "/media/bio/LW/new/regular/test_results"
# output_folder = "/media/bio/LW/new/sensitive/test_results"
output_filename = "final_mask.png"
output_path = os.path.join(output_folder, output_filename)

# Load and process mask images
masks = load_binary_masks(input_folder)

if masks:
    cleaned_masks, final_mask, is_blank = self_check_mechanism_on_masks(masks)
    print("Cleaned mask images after removing false signals:")
    for idx, mask in enumerate(cleaned_masks):
        print(f"Mask image {idx + 1}:\n{mask}")

    # Save the final mask
    save_final_mask(final_mask, output_path)
    print("Final status (is blank):", is_blank)
else:
    print("No mask images found in the specified folder.")