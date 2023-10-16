"""Adapted from https://github.com/peterhan91/cycleGAN/blob/db8f1d958c0879c29cf3932cae74a166317be812/prepro.py#L39"""

import os
import numpy as np
from glob import glob
import pydicom
import scipy.ndimage
from pathlib import Path
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm



class CTExtractor:
    def __init__(self, input_path, out_path):
        super(CTExtractor, self).__init__()

        self.MIN_BOUND = -1000.0
        self.MAX_BOUND = 400.0
        self.PIXEL_MEAN = 0.25
        self.roi = 320
        self.size = 128

        self.path = input_path
        self.outpath = out_path
        self.slices = []
        self.fname = ''

    # Load the scans in given folder path
    def load_scan(self):
        slices_ = [pydicom.read_file(s) for s in glob(
            os.path.join(self.path, self.fname, '*/*/*.dcm'))]

        # Problem when CXR is available. This fixes it.
        num_subfolders = len(os.listdir(os.path.join(self.path, self.fname)))
        if num_subfolders > 1:
            print(f"Filename: {self.fname}, No. Subfolders: {num_subfolders}")
            slices = []
            for s in slices_:
                if s.Modality == 'CT':
                    slices.append(s)
                else:
                    print(s.Modality)
        else:
            slices = slices_

        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        try:
            slice_thickness = np.abs(
                slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
        except:
            slice_thickness = np.abs(
                slices[0].SliceLocation - slices[1].SliceLocation)

        for s in slices:
            s.SliceThickness = slice_thickness
            if s.Modality != 'CT':
                print(f"NOT A CT. This is a {s.Modality}")

        return slices

    def get_pixels_hu(self, slices):
        image = np.stack([s.pixel_array for s in slices])
        # Convert to int16 (from sometimes int16),
        # should be possible as values should always be low enough (<32k)
        image = image.astype(np.int16)

        # Set outside-of-scan pixels to 0
        # The intercept is usually -1024, so air is approximately 0
        image[image == -2000] = 0

        # Convert to Hounsfield units (HU)
        for slice_number in range(len(slices)):

            intercept = slices[slice_number].RescaleIntercept
            slope = slices[slice_number].RescaleSlope

            if slope != 1:
                image[slice_number] = slope * \
                    image[slice_number].astype(np.float64)
                image[slice_number] = image[slice_number].astype(np.int16)

            image[slice_number] += np.int16(intercept)

        return np.array(image, dtype=np.int16)

    def resample(self, image, scan, new_spacing=[1.0, 1.0, 1.0]):
        # Determine current pixel spacing
        # print(scan[0].SliceThickness)
        # print(scan[0].PixelSpacing)
        spacing = np.array([scan[0].SliceThickness] +
                           list(scan[0].PixelSpacing), dtype=np.float32)

        resize_factor = spacing / new_spacing
        new_real_shape = image.shape * resize_factor
        new_shape = np.round(new_real_shape)
        real_resize_factor = new_shape / image.shape
        new_spacing = spacing / real_resize_factor

        image = scipy.ndimage.interpolation.zoom(
            image, real_resize_factor, mode='nearest')

        return image, new_spacing

    def normalize(self, image):
        image = (image - self.MIN_BOUND) / (self.MAX_BOUND - self.MIN_BOUND)
        image[image > 1] = 1.
        image[image < 0] = 0.
        return image*2-1.

    def zero_center(self, image):
        image = image - self.PIXEL_MEAN
        return image

    def pad_center(self, pix_resampled):
        pad_z = max(self.roi - pix_resampled.shape[0], 0)
        pad_x = max(self.roi - pix_resampled.shape[1], 0)
        pad_y = max(self.roi - pix_resampled.shape[2], 0)
        try:
            pad = np.pad(pix_resampled,
                         [(pad_z//2, pad_z-pad_z//2), (pad_x//2,
                                                       pad_x-pad_x//2), (pad_y//2, pad_y-pad_y//2)],
                         mode='constant',
                         constant_values=pix_resampled[0][10][10])
        except ValueError:
            print(pix_resampled.shape)
        except IndexError:
            print(pix_resampled.shape)
            pass
        return pad

    def crop_center(self, vol, cropz, cropy, cropx):
        z, y, x = vol.shape
        startx = x//2-(cropx//2)
        starty = y//2-(cropy//2)
        startz = z//2-(cropz//2)
        return vol[startz:startz+cropz, starty:starty+cropy, startx:startx+cropx]

    def save(self):
        path = os.path.join(self.outpath, self.fname, '128.npy')
        Path(os.path.join(self.outpath, self.fname)).mkdir(
            parents=True, exist_ok=True)
        np.save(path, self.vol)

    def run(self, fname):
        self.fname = fname
        self.patient = self.load_scan()
        self.vol = self.get_pixels_hu(self.patient)
        self.vol, _ = self.resample(self.vol, self.patient)
        if self.vol.shape[0] >= self.roi and self.vol.shape[1] >= self.roi and self.vol.shape[2] >= self.roi:
            self.vol = self.crop_center(self.vol, self.roi, self.roi, self.roi)
        else:
            self.vol = self.pad_center(self.vol)
            self.vol = self.crop_center(self.vol, self.roi, self.roi, self.roi)
        assert self.vol.shape == (self.roi, self.roi, self.roi)
        self.vol = scipy.ndimage.zoom(self.vol,
                                      [self.size/self.roi, self.size /
                                          self.roi, self.size/self.roi],
                                      mode='nearest')
        assert self.vol.shape == (self.size, self.size, self.size)
        self.vol = self.normalize(self.vol)
        self.save()


def worker(fname, extractor):
    try:
        extractor.run(fname)
    except:
        print('Error extracting the lung CT')
        print(fname)


if __name__ == "__main__":
    # Argument Parsing
    parser = argparse.ArgumentParser(description='CTExtractor for processing CT scans.')
    parser.add_argument('--input_path', type=str, required=True, help='Path to the input CT scans directory')
    parser.add_argument('--path_output', type=str, required=True, help='Path to the directory to save processed CT scans')
    args = parser.parse_args()

    input_path = args.input_path  
    path_output = args.path_output  


    extractor = CTExtractor(input_path, path_output)

    def worker_partial(fname):
        return worker(fname, extractor)

    fnames = os.listdir(input_path)
    print('total # of scans', len(fnames))

    with Pool(processes=4) as pool:
        res = list(tqdm(pool.imap(
            worker_partial, iter(fnames)), total=len(fnames)))
