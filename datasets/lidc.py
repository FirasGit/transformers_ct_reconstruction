import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
from torchvision import transforms
import glob


class LIDCDataset(Dataset):
    def __init__(self, root_dir='../LIDC', augmentation=False, projection=False, projection_plane=None, return_indices=False, indices_path=None, split='train'):
        self.root_dir = root_dir
        self.split = split
        self.file_names = self.get_split()
        self.augmentation = augmentation
        self.projection = projection
        self.projection_plane = projection_plane
        self.return_indices = return_indices
        self.indices_path = indices_path

    def get_split(self):
        file_names_ = glob.glob(os.path.join(
            self.root_dir, './**/*.npy'), recursive=True)
        if self.split == 'train':
            # take 70% of the data
            file_names = file_names_[:int(len(file_names_)*0.7)]
        if self.split == 'val':
            # take 20% of the data
            file_names = file_names_[
                int(len(file_names_)*0.7):int(len(file_names_)*0.9)]
        if self.split == 'test':
            file_names = file_names_[int(len(file_names_)*0.9):]
        return file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        if self.return_indices:
            path = self.file_names[index]
            indices_path = os.path.join(
                self.indices_path, self.split, path.split('/')[-2])
            indices_ct = torch.tensor(
                np.load(os.path.join(indices_path, 'CT.npy')))
            indices_ap = torch.tensor(
                np.load(os.path.join(indices_path, 'ap.npy')))
            indices_lat = torch.tensor(
                np.load(os.path.join(indices_path, 'lat.npy')))
            return {'indices_ct': indices_ct, 'indices_ap': indices_ap, 'indices_lat': indices_lat, 'file_name': path}

        else:
            path = self.file_names[index]
            img = np.load(path)

            if self.augmentation:
                random_n = torch.rand(1)
                if random_n[0] > 0.5:
                    img = np.flip(img, 2)

            if self.projection:
                if self.projection_plane is None:
                    projection_plane = np.random.choice(['ap', 'lat'])
                else:
                    projection_plane = self.projection_plane

                 # Flip the image because for some reason the projection is flipped. 
                 # Note that we take the mean projection here for ease of use, since we have seen no difference in image quality when doing so.
                if projection_plane == 'ap':
                    img = np.flip(img.mean(axis=1), 0)
                elif projection_plane == 'lat':
                    img = np.flip(img.mean(axis=2), 0)

            imageout = torch.from_numpy(img.copy()).float()
            imageout = imageout.unsqueeze(0)

            return {'source': imageout, 'file_name': path}
