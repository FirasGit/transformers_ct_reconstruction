import os
import numpy as np
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from datasets import LIDCDataset


# --------------------------------------------- #
#                  Data Utils
# --------------------------------------------- #

def load_data(args):
    train_data = LIDCDataset(
        root_dir=args.path_to_preprocessed_data,
        augmentation=False,
        projection=False,
        return_indices=True,
        indices_path=args.path_to_data_indices,
        split='train'
    )

    val_data = LIDCDataset(
        root_dir=args.path_to_preprocessed_data,
        augmentation=False,
        projection=False,
        return_indices=True,
        indices_path=args.path_to_data_indices,
        split='val'
    )

    test_data = LIDCDataset(
        root_dir=args.path_to_preprocessed_data,
        augmentation=False,
        projection=False,
        return_indices=True,
        indices_path=args.path_to_data_indices,
        split='test'
    )

    train_loader = DataLoader(
        train_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    val_loader = DataLoader(
        val_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)

    test_loader = DataLoader(
        test_data, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False)
    return train_loader, val_loader, test_loader


# --------------------------------------------- #
#                  Module Utils
#            for Encoder, Decoder etc.
# --------------------------------------------- #

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def plot_images(images, slice_idx=60):
    x = images["input"]
    reconstruction = images["rec"]
    half_sample = images["half_sample"]
    full_sample = images["full_sample"]

    fig, axarr = plt.subplots(1, 4)
    axarr[0].imshow(x.cpu().detach().numpy()[0][0][slice_idx])
    axarr[1].imshow(reconstruction.cpu().detach().numpy()
                    [0][0][slice_idx])
    axarr[2].imshow(half_sample.cpu().detach().numpy()[0][0][slice_idx])
    axarr[3].imshow(full_sample.cpu().detach().numpy()[0][0][slice_idx])
    plt.show()
