import random
import numpy as np
import torch
from torch.utils.data import Dataset


class SiameseNetworkDataset(Dataset):
    """
    This class is used to create a dataset for the Siamese Network.

    Args:
        Dataset (torch.utils.data.Dataset): Pytorch Dataset class
    Returns:
        image0, image1, label    
    
    """
    def __init__(self,imageFolderDataset,transform=None):
        """
        Initialize the dataset
        Args:
            imageFolderDataset (torchvision.datasets.ImageFolder): Pytorch ImageFolder class
            transform (torchvision.transforms): Pytorch transforms
        """
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.len_dataset = len(imageFolderDataset)-1
        
    def __getitem__(self,index):
        #img0_tuple = random.choice(self.imageFolderDataset)
        idx = random.randint(0, self.len_dataset)
        img0_tuple = self.imageFolderDataset[idx]

        #We need to approximately 50% of images to be in the same class
        should_get_same_class = random.randint(0,1) 
        if should_get_same_class:
            while True:
                #Look untill the same class image is found
                idx = random.randint(0, self.len_dataset)
                img1_tuple = self.imageFolderDataset[idx] 
                if img0_tuple[1] == img1_tuple[1]:
                    break
        else:

            while True:
                #Look untill a different class image is found
                idx = random.randint(0, self.len_dataset)
                img1_tuple = self.imageFolderDataset[idx] 
                if img0_tuple[1] != img1_tuple[1]:
                    break

        img0=img0_tuple[0]
        img1=img1_tuple[0]            

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        
        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1] != img0_tuple[1])], dtype=np.float32))
    
    def __len__(self):
        return len(self.imageFolderDataset)