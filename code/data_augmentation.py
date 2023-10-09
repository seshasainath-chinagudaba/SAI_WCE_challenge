import os
import cv2
import random
import numpy as np
import torch
import torchvision.transforms as transforms
import albumentations as A
from PIL import Image, ImageFilter 
from skimage.util import random_noise

class WCEImageTransforms:
    def __init__(self, rotation_degrees, blur_parameters):
        self.rotation_degrees = rotation_degrees
        self.blur_parameters = blur_parameters


    def apply_clahe(self,rgb_image, clip_limit=2.0, tile_grid_size=(8, 8)):
        # Convert RGB image to LAB color space
        rgb_image = cv2.convertScaleAbs(np.array(rgb_image))
        lab_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2LAB)

        # Split the LAB image into L, A, and B channels
        l_channel, a_channel, b_channel = cv2.split(lab_image)

        # Apply CLAHE to the L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        cl_channel = clahe.apply(l_channel)

        # Merge the processed L channel with the original A and B channels
        enhanced_lab_image = cv2.merge((cl_channel, a_channel, b_channel))

        # Convert the enhanced LAB image back to RGB color space
        enhanced_rgb_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced_rgb_image)
    
    def __call__(self, img):
        # Randomly decide whether to rotate the image
        should_rotate = random.choice([True, False])
        if should_rotate:
            # Randomly select a degree from the set and rotate the image
            random_degree = random.choice(self.rotation_degrees)
            img = img.rotate(random_degree, expand=True)

            
        should_add_clahe = random.choice([True, False])
        if should_add_clahe:
            img = self.apply_clahe(img)
        
        trans = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor()])
        
        img_tensor = trans(img)

        return img_tensor
