# Data handling utilities
import cv2
import imgaug.augmenters as iaa
import numpy as np
import pandas as pd
import torch
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
##############################################################
class OilWellDataset(Dataset):
    def __init__(self, images_list: list[str], masks_list: list[str], do_aug: bool = False, flip_lr_prob: int = 0.5, flip_ud_prob: int = 0.5, blur_prob: int = 0.5, max_blur_scale: float = 1.0) -> None:
        '''
        Create a custom OilWellDataset object.

        Parameters:
        ----------------------
        images_list: list[str]
            List of input images for dataset

        masks_list: list[str]
            List of binary segmentation masks corresponding to respective image files in images_list
        
        do_aug: bool
            Boolean flag to turn on/off data augmentation

        flip_lr_prob: float value in range [0.0, 1.0]   (default: 0.5)
            Probability with which to flip an image horizontally for data augmentation

        flip_ud_prob: float value in range [0.0, 1.0]   (default: 0.5)
            Probability with which to flip an image vertically for data augmentation
        
        blur_prob: float value in range [0.0, 1.0]   (default: 0.5)
            Random probability with which to blur an image with an isotropic Gaussian filter

        max_blur_scale: positive float value   (default: 1.0)
            Samples a standard deviation estimate "sigma" uniformly from range [0, max_blur_scale]. 
            Then, blurs an image via convolution with an isotropic 2D Gaussian filter of standard deviation "sigma."
        '''
        self.images_list = images_list
        self.masks_list = masks_list
        self.do_aug = do_aug
        self.flip_lr_prob = flip_lr_prob
        self.flip_ud_prob = flip_ud_prob
        self.blur_prob = blur_prob
        self.max_blur_scale = max_blur_scale

    def __getitem__(self, index: int) -> [torch.Tensor, torch.Tensor]:
        '''
        Returns PyTorch tensors of an image at a specific index of self.images_list as well as it corresponding mask. 
        Also, applies relevant data augmentations to both the image and the mask. 

        Parameters:
        ----------------------
        index: int 
            Index of dataset. Integer value in the range [0, len(dataset)-1].
        '''        
        # Read image using cv2. Also, flip BGR to RGB by reversing the final array index.
        im = cv2.imread(self.images_list[index])[..., ::-1]  # im.shape = (n_pixels_width, n_pixels_height, 3)
        mask = np.load(self.masks_list[index]) # mask.shape = (n_pixels_width, n_pixels_height, 1)

        if self.do_aug:
            # Set up data augmenter.
            operations = [iaa.Fliplr(self.flip_lr_prob), iaa.Flipud(self.flip_ud_prob)]
            if self.blur_prob and np.random.uniform(0,1) <= self.blur_prob:
                sigma = np.random.uniform(0, self.max_blur_scale)
                operations.append(iaa.GaussianBlur(sigma=sigma))
            augmenter = iaa.Sequential(operations)
            im, mask = augmenter(image=im, segmentation_maps=SegmentationMapsOnImage(mask, shape=mask.shape))
            mask = mask.get_arr()

        # Cast im and mask as torch tensors of shape (No. of color channels, width, height).
        im = torch.from_numpy(im.copy()).permute((2, 0, 1))
        mask = torch.from_numpy(mask.copy()).permute((2, 0, 1))

        # Scale im linearly to range [0, 1].
        im = (im - im.min())/(im.max() - im.min())

        return im, mask
    
    def __len__(self):
        '''
        Returns the length of the dataset.
        '''
        return len(self.images_list)
##############################################################
def custom_train_val_split(well_counts_csv: str, val_size: float) -> [list[str], list[str]]:
    '''
    Perform train-validation split according to the following methodology.
    1. Construct the histogram of well counts per image, where each bin covers a single integer well count. 
    2, For stratified train-val split on well counts, bins must frequencies equal to or exceeding 1/val_size.
    3. Keep note of integer well counts (or bin centers) that have occurrence frequencies less than 1/val_size.
    4. Group images with such integer well counts for random train-test split.
    5. Group remaning images for stratified train-test split on well counts.
    6. Combine train/val sets from the two groups to build the grand training/validation data sets.

    Parameters:
    ----------------------
    well_counts_csv: str
        .csv file containing tabular data of wells counts per input image
    
    val_size: float
        Fraction of overall data size to be reserved as validation data
    
    Returns:
    ----------------------
    img_train: list[str]
        Basenames of images chosen to be in the training set
    
    img_val: list[str]
        Basenames of images chosen to constitute the validation set
    '''
    # Load .csv file containing information of well count per image.
    df = pd.read_csv(well_counts_csv)
    
    # Construct bins centered at each integer well count between 0 and the max count in an image.
    bin_edges = np.arange(df['Well count'].max()+2) - 0.5
    bin_centers = np.arange(df['Well count'].max()+1)
    bin_frequencies, _ = np.histogram(df['Well count'], bins=bin_edges)
    # Minimum frequency per count to have a validation set of minimum size 1 at every well count
    min_freq = int(1/val_size)
    # Split list of images into groups: one for random train-test split, another for stratified train-test split based on well count per image.
    rand_split_well_counts = set(bin_centers[np.where(bin_frequencies < min_freq)[0]])
    rand_split_img_group = []
    stratified_split_img_group = []
    stratified_split_well_counts = []
    for i, record in df.iterrows():
        if record['Well count'] in rand_split_well_counts:
            rand_split_img_group.append(record['Image'])
        else:
            stratified_split_img_group.append(record['Image'])
            stratified_split_well_counts.append(record['Well count'])
    rand_split_img_group = np.array(rand_split_img_group)
    stratified_split_img_group = np.array(stratified_split_img_group)
    # Perform random split for first group with well count frequencies < min_freq.
    img_train_rand, img_val_rand, _, _ = train_test_split(rand_split_img_group.reshape(-1,1), rand_split_img_group, test_size=val_size)
    # Perform train-validation split by stratifying input images on well counts.
    img_train_strat, img_val_strat, _, _ = train_test_split(stratified_split_img_group.reshape(-1,1), stratified_split_well_counts, test_size=val_size, stratify=stratified_split_well_counts, random_state=42)
    # Combine groups to form training and validation data sets.
    img_train = np.concatenate((img_train_rand.squeeze(), img_train_strat.squeeze()))
    img_val = np.concatenate((img_val_rand.squeeze(), img_val_strat.squeeze()))
    return img_train, img_val
##############################################################
