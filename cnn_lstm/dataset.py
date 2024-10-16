import os
from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF


class CustomDatasetLoader:
    def __init__(self, base_dir, dataset_type='train', transform=None, sequence_length=5):
        self.base_dir = base_dir
        self.dataset_type = dataset_type
        self.transform = transform
        self.sequence_length = sequence_length
        self.resize = transforms.Resize((192, 192))
        self.normal_regular = transforms.Normalize([0.22274131, 0.15919147, 0.16728036],
                                                   [0.0873689, 0.05578854, 0.05981448])
        self.normal_sensitive = transforms.Normalize([0.20837964, 0.17967771, 0.1828326],
                                                     [0.07973031, 0.06490166, 0.0666421])

        if dataset_type in ['train', 'val']:
            self.images_dir = os.path.join(base_dir, f'{dataset_type}_data')
            self.labels_dir = os.path.join(base_dir, f'{dataset_type}_label')
        elif dataset_type == 'test':
            self.images_dir = os.path.join(base_dir, 'test_data')

        self.dataset = self.prepare_dataset()

    def prepare_dataset(self):
        dataset = []
        if self.dataset_type != 'test':
            for label_folder in os.listdir(self.labels_dir):
                label_folder_path = os.path.join(self.labels_dir, label_folder)
                for mask_file in sorted(os.listdir(label_folder_path)):
                    mask_path = os.path.join(label_folder_path, mask_file)
                    mask_base_name = mask_file.split('.')[0]
                    image_folder_base = os.path.join(self.images_dir, label_folder, mask_base_name)

                    image_sub_folder = os.path.join(image_folder_base)
                    image_files = sorted(os.listdir(image_sub_folder))
                    for i in range(len(image_files) - self.sequence_length + 1):
                        sequence_paths = [os.path.join(image_sub_folder, image_files[j]) for j in
                                          range(i, i + self.sequence_length)]
                        dataset.append((sequence_paths, mask_path))
        else:
            for subdir, dirs, files in os.walk(self.images_dir):
                for dir in dirs:
                    image_sub_folder = os.path.join(self.images_dir, dir)
                    image_files = sorted(os.listdir(image_sub_folder))
                    if len(image_files) >= self.sequence_length:
                        for i in range(len(image_files) - self.sequence_length + 1):
                            sequence_paths = [os.path.join(image_sub_folder, image_files[j]) for j in
                                              range(i, i + self.sequence_length)]
                            dataset.append(sequence_paths)
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        if self.dataset_type == 'test':
            image_paths = self.dataset[index]
            images = [Image.open(path).convert('RGB') for path in image_paths]
            images = [self.resize(image) for image in images]
            if self.transform is not None:
                images = [self.transform(image) for image in images]
            else:
                images = [TF.to_tensor(image) for image in images]
                images = [self.normal_regular(image) for image in images]
            return images
        else:
            image_paths, mask_path = self.dataset[index]
            images = [Image.open(path).convert('RGB') for path in image_paths]
            mask = Image.open(mask_path).convert('L')
            images = [self.resize(image) for image in images]
            mask = self.resize(mask)

            if self.transform is not None:
                images = [self.transform(image) for image in images]
                mask = self.transform(mask)
            else:
                images = [TF.to_tensor(image) for image in images]
                mask = TF.to_tensor(mask)  # 直接将掩码转换为张量
                mask = (mask > 0.0).float()

            images = [self.normal_regular(image) for image in images]

            return images, mask


# 使用示例
if __name__ == "__main__":
    dataset_path = 'dataset/regular'
    # dataset_path = 'dataset/sensitive'

    dataset = CustomDatasetLoader(dataset_path)
    images, mask = dataset[0]
    print(f"Images shape: {[image.shape for image in images]}")
    print(f"Mask shape: {mask.shape}")
    print(len(dataset))
    print("Images paths:", dataset.dataset[0][0])
    print("Mask path:", dataset.dataset[0][1])
