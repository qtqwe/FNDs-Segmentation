import os
import torch
from PIL import Image
from model import UNet_ConvLSTM
import torchvision.transforms as transforms
import numpy as np

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the model
model = UNet_ConvLSTM(n_channels=3, n_classes=1, phi=2).to(device=device)
model.load_state_dict(torch.load("dataset/regular/best_model_f.pth"))
# model.load_state_dict(torch.load("dataset/sensitive/best_model_f.pth"))
model.eval()

# Test data path
test_data_path = 'dataset/regular/test_data'
# test_data_path = 'dataset/sensitive/test_data
# '
# Results save path
results_path = 'dataset/regular/test_results'
# results_path = 'dataset/sensitive/test_results'
# Define sliding window parameters
sequence_length = 5

# Iterate over folders in the test data directory
for folder_name in os.listdir(test_data_path):
    folder_path = os.path.join(test_data_path, folder_name)
    if not os.path.isdir(folder_path):
        print(f"Non-folder content: {folder_path}")
        continue

    # Create a results folder
    result_folder_path = os.path.join(results_path, folder_name)
    os.makedirs(result_folder_path, exist_ok=True)

    # Iterate over image files in subfolders
    for sub_folder_name in os.listdir(folder_path):
        sub_folder_path = os.path.join(folder_path, sub_folder_name)
        if not os.path.isdir(sub_folder_path):
            continue

        # Create results folder within the subfolder
        result_sub_folder_path = os.path.join(result_folder_path, sub_folder_name)
        os.makedirs(result_sub_folder_path, exist_ok=True)

        # Load images and perform predictions
        image_paths = sorted([os.path.join(sub_folder_path, image_name) for image_name in os.listdir(sub_folder_path)])
        num_images = len(image_paths)
        print(f"Current subfolder path: {sub_folder_path}")
        print(f"Number of images in current subfolder: {num_images}")
        if num_images < sequence_length:
            print("Number of images is less than the sequence length, skipping...")
            continue

        # Iterate with a sliding window
        for start_index in range(num_images - sequence_length + 1):
            print(f"Current sliding window start index: {start_index}")
            # Load image sequence and perform predictions
            image_sequence = [Image.open(image_paths[j]).convert('RGB') for j in
                              range(start_index, start_index + sequence_length)]

            # Image preprocessing
            transform = transforms.Compose([
                transforms.Resize((192, 192)),
                transforms.ToTensor(),
                transforms.Normalize([0.22274131, 0.15919147, 0.16728036],
                                     [0.0873689, 0.05578854, 0.05981448])
                # transforms.Normalize([0.20837964, 0.17967771, 0.1828326],
                #                      [0.07973031, 0.06490166, 0.0666421])

            ])
            image_sequence = [transform(image) for image in image_sequence]

            # Stack images and move to device
            images = torch.stack(image_sequence).unsqueeze(0).to(device=device, dtype=torch.float32)  # Add batch dimension

            # Model prediction
            with torch.no_grad():
                outputs = model(images)
                outputs = torch.sigmoid(outputs).detach().cpu().numpy()

            # Thresholding and binarization of the output
            outputs[outputs >= 0.4] = 1
            outputs[outputs < 0.4] = 0

            # Save the prediction results
            for j, output in enumerate(outputs):
                output_image = output[0] * 255  # Convert binarized output to 0-255 range
                output_image = Image.fromarray(output_image.astype(np.uint8))
                output_image.save(os.path.join(result_sub_folder_path, f"output_{start_index + j}.png"))

print("Prediction results saved!")
