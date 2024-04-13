#!/usr/bin/env python
'''
Model training script. Run using the following syntax from the repository root directory.

python scripts/train.py -i config/train.cfg
'''
import numpy as np
import pandas as pd
import os, torch, sys
import torch.optim as optim
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from tqdm import tqdm
# Custom modules
from data_utils import custom_train_val_split, OilWellDataset
from dice_loss import DiceLoss
from general_utils import  create_path, setup_logger_stdout
from read_config import read_config
from unet_model import UNet
##############################################################
# Set up default device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_device(device)
##############################################################
def train(hotpotato: dict) -> None:
    '''
    Train a deep learning model.

    Parameters:
    ---------------------------
    hotpotato: dict
        Dictionary of input parameters
    '''
    # Initiate logger.
    logger = setup_logger_stdout()
    
    img_train, img_val = custom_train_val_split(hotpotato['well_counts_csv'], hotpotato['val_size'])
    # Construct training dataset.
    train_images_list = []
    train_masks_list = []
    for basename in img_train:
        train_images_list.append(hotpotato['images_path']+'/'+basename+'.'+hotpotato['imgformat'])
        train_masks_list.append(hotpotato['masks_path']+'/'+basename+'_mask.'+hotpotato['maskformat'])
    # Turn on data augmentation for training set.
    train_dataset = OilWellDataset(train_images_list, train_masks_list, do_aug=True, flip_lr_prob=hotpotato['flip_lr_prob'], 
                               flip_ud_prob=hotpotato['flip_ud_prob'], blur_prob=hotpotato['blur_fraction'], max_blur_scale=hotpotato['max_blur_scale'])
    N_train = len(train_dataset)
    
    # Construct validation dataset.
    val_images_list = []
    val_masks_list = []
    for basename in img_val:
        val_images_list.append(hotpotato['images_path']+'/'+basename+'.'+hotpotato['imgformat'])
        val_masks_list.append(hotpotato['masks_path']+'/'+basename+'_mask.'+hotpotato['maskformat'])
    # No data augmentation for validation set
    val_dataset = OilWellDataset(val_images_list, val_masks_list, do_aug=False)
    N_val = len(val_dataset)
    
    # Define data loaders for training and validation sets.
    loader_args = dict(batch_size=hotpotato['batch_size'], pin_memory=True, num_workers=os.cpu_count())
    train_loader = DataLoader(train_dataset, shuffle=True, drop_last=False, **loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, drop_last=False, **loader_args)
    
    # Initiate model.
    model = UNet()
    model.to(device)
    pretrained_state_dict = torch.load(hotpotato['pretrained_weights_file'], map_location=device)
    # Freeze all layers.
    for param in model.parameters():
        param.requires_grad = False
    # Unfreeze weights in final few layers.
    for param in model.conv.parameters():
        param.requires_grad = True
    for param in model.decoder1.parameters():
        param.requires_grad = True
    for param in model.upconv1.parameters():
        param.requires_grad = True
    for param in model.decoder2.parameters():
        param.requires_grad = True
    for param in model.upconv2.parameters():
        param.requires_grad = True
    # Compute the number of trainable parameters
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    trainable_params = sum([np.prod(p.size()) for p in model_parameters])
    logger.info('No. of trainable model parameters = %d'% (trainable_params))

    # Set up optimizer and loss criterion.
    optimizer = optim.Adam(model.parameters(), lr=hotpotato['learning_rate'])
    criterion = DiceLoss()
    # Compute the batch-averaged training and validation loss at every epoch.
    train_loss = []
    val_loss = []
    # Whenever the validation loss drops to a new minimum, save model weights to disk.
    lowest_val_loss = float('inf')
    create_path(hotpotato['outpath'])

    # Begin model training.
    logger.info('Starting training:\n Epochs = %d \n Batch size = %d \n Learning rate = %.3g \n Training size = %d \n Validation size = %d \n Device = %s'
                % (hotpotato['N_epochs'], hotpotato['batch_size'], hotpotato['learning_rate'], N_train, N_val, device.type))
    for epoch in tqdm(range(1,hotpotato['N_epochs']+1)):
        # Training  phase
        model.train()
        current_train_loss = 0 # Stores average training loss over batches in current epoch
        for batch in train_loader:
            images, true_masks = batch
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            # Reset model parameter gradients to zero before every batch training.
            optimizer.zero_grad()
            # Compute predicted mask and loss per batch.
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks.float())
            current_train_loss += (loss.item()*true_masks.shape[0])
            # Batch gradient descent
            loss.backward()
            optimizer.step()
        current_train_loss /= N_train
        train_loss.append(current_train_loss)
    
        # Validation phase
        model.eval()
        current_val_loss = 0  # Stores average validation loss over batches in current epoch
        with torch.no_grad():
            for batch in val_loader:
                images, true_masks = batch
                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)
                # Compute predicted mask and loss per batch.
                pred_masks = model(images)
                loss = criterion(pred_masks, true_masks.float())
                current_val_loss += (loss.item()*true_masks.shape[0])
        current_val_loss /= N_val
        val_loss.append(current_val_loss)

        logger.info('Statistics for epoch %d: \n Training loss: %.4f \n Validation loss: %.4f'% (epoch, train_loss[-1], val_loss[-1]))
        
        # Write model weights to disk.
        if val_loss[-1] < lowest_val_loss:
            lowest_val_loss = val_loss[-1]
            logger.info('Achieved new minimum validation loss. Writing model weights to disk.')
            torch.save(model.state_dict(), hotpotato['outpath']+'/model_weights.pth')
    
    # Save loss curve data to disk intermittently.
    if epoch%10==0 or epoch==hotpotato['N_epochs']:
        loss_df = pd.DataFrame({'train_loss':np.array(train_loss), 'val_loss':np.array(val_loss)}, index=np.arange(1, epoch+1))
        loss_df.to_csv(hotpotato['outpath']+'/model_loss.csv', index=False)

##############################################################
def set_defaults(hotpotato: dict) -> dict:
    """
    Set default values for keys in a dictionary of input parameters.

    Parameters
    ----------
    hotpotato : dictionary
         Dictionary of input parameters read from a configuration script

    Returns
    -------
    hotpotato : dictionary
        Input dictionary with keys set to default values
    """
    # Default image file format
    if hotpotato['imgformat']=='':
        hotpotato['imgformat'] = 'png'
    # Default mask file format
    if hotpotato['maskformat']=='':
        hotpotato['maskformat'] = 'npy'
    # Default output path 
    if hotpotato['outpath']=='':
        hotpotato['outpath'] = hotpotato['images_path']
    # Default probability with which to flip an image horizontally = 0.5
    if hotpotato['flip_lr_prob']=='':
        hotpotato['flip_lr_prob'] = 0.5
    # Default probability with which to flip an image vertically = 0.5
    if hotpotato['flip_ud_prob']=='':
        hotpotato['flip_ud_prob'] = 0.5
    # Default value for the fraction of images to be blurred with an isotropic 2D Gaussian filter = 0.5
    if hotpotato['blur_fraction']=='':
        hotpotato['blur_fraction'] = 0.5
    # Default max_scale for 2D Gaussian filter
    if hotpotato['max_blur_scale']=='':
        hotpotato['max_blur_scale'] = 0.5
    # Default epoch limit = 10
    if hotpotato['N_epochs']=='':
        hotpotato['N_epochs'] = 10
    # Default fractional size of validation set = 0.2
    if hotpotato['val_size']=='':
        hotpotato['val_size'] = 0.2
    # Default batch size = 32
    if hotpotato['batch_size']=='':
        hotpotato['batch_size'] = 32
    # Default starting value of learning rate = 1.0e-5
    if hotpotato['learning_rate']=='':
        hotpotato['learning_rate'] = 1.0e-5
    return hotpotato
##############################################################################
def main():
    # Help description
    usage = 'Model training script'
    parser = ArgumentParser(description=usage)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-i', action='store', required=True, dest='inputs_cfg', type=str,
                            help="Configuration script of inputs")
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit("Missing input configuration script")

    parse_args = parser.parse_args()
    hotpotato = set_defaults(read_config(parse_args.inputs_cfg))
    train(hotpotato)
      
##############################################################################
if __name__=='__main__':
    main()
##############################################################################
