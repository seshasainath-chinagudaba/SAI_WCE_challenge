from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image, ImageDraw
import os
import random

class WCEClassDataset(Dataset):
    def __init__(self, root_dir,num_models=None,tr=True):
        super().__init__()
        self.root_dir = root_dir
        self.num_models = num_models
        
        # List of subfolders (class names)
        if tr:
            self.classes = [os.path.join(root_dir, 'bleeding','Images'), os.path.join(root_dir, 'non-bleeding','images')]
        else:
            self.classes=[os.path.join(root_dir,'')]

        # Initialize lists to hold image paths and labels
        self.image_paths = []
        self.labels = []

        # Load image paths and labels
        for class_idx, class_dir in enumerate(self.classes):
            image_files = os.listdir(class_dir)
            for image_file in image_files:
                image_path = os.path.join(class_dir, image_file)
                self.image_paths.append(image_path)
                self.labels.append(class_idx) # bleeding images given label 0 and non-bleeding 1
        combined_data = list(zip(self.image_paths, self.labels))
        random.shuffle(combined_data)
        self.image_paths, self.labels = zip(*combined_data)
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load the image using PIL
        image = Image.open(image_path)
        
        return image, label


class WCEClassSubsetDataset(Dataset):
    def __init__(self, original_dataset, subset_indices, transform=None):
        self.original_dataset = original_dataset
        self.subset_indices = subset_indices
        self.transform = transform

    def __len__(self):
        return len(self.subset_indices)

    def __getitem__(self, idx):
        # Get an item from the subset
        image, label = self.original_dataset[self.subset_indices[idx]]

        # Apply the transform if it is provided
        if self.transform:
            images = []
            for i in range(self.original_dataset.num_models):
                images.append(self.transform(image))

            return images, label

        return image, label
