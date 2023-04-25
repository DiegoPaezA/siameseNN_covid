import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os 
from PIL import Image 

class SiameseNetworkDataset(Dataset):
    """
    This class is used to create a dataset for the Siamese Network.

    Args:
        Dataset (torch.utils.data.Dataset): Pytorch Dataset class
    Returns:
        image0, image1, label    
    
    """
    def __init__(self,root_path,transform=None):
        """
        Initialize the dataset
        Args:
            imageFolderDataset (torchvision.datasets.ImageFolder): Pytorch ImageFolder class
            transform (torchvision.transforms): Pytorch transforms
        """
        self.root_path = root_path    
        self.transform = transform
        self.classes = os.listdir(root_path)
        self.classes_to_idx = {class_name:i for i, class_name in enumerate(self.classes)} 
        self.img_paths, self.labels = self._make_samples()

    def _make_samples(self):
        samples_images = []
        samples_classes = []
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_path, cls_name)
            for file_name in os.listdir(class_dir):
                sample_path = os.path.join(class_dir, file_name)
                samples_images.append(sample_path)
                samples_classes.append(self.classes_to_idx[cls_name])
        return samples_images, samples_classes 

    def __getitem__(self,idx):
        

        path_img0 = self.img_paths[idx] 
        label0 = self.labels[idx]

        if random.randint(0,1) == 0:
            idxs = np.argwhere(np.array(self.labels) == label0).flatten()
            idxs = np.delete(idxs, label0)      #Prevent to select the same image twice. 
        else:
            idxs = np.argwhere(np.array(self.labels) != label0).flatten()
 
        other_idx = np.random.choice(idxs)
         
        path_img1 = self.img_paths[other_idx]
        label1 = self.labels[other_idx]
       
        img0 = Image.open(path_img0)
        img1 = Image.open(path_img1)

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(label0 != label1)], dtype=np.float32))
    
    def __len__(self):
        return len(self.img_paths)