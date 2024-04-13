#!/usr/bin/env python
'''
Render binary segmentation masks in human-interpretable RGB color format.

python render_binary_masks.py -f <File format of output mask images> -m <Path to segmentation masks> -o <Output path>
'''
import glob, sys
import numpy as np
from argparse import ArgumentParser
from general_utils import create_path, setup_logger_stdout
from PIL import Image
##############################################################
def render_binary_mask(mask_list: list[str], outpath: str, imgformat: str) -> None:
    '''
    Render binary mask file in human-interpretable image format for visualization.

    Parameters:
    -----------------------------
    mask_list: list[str]
        List of .npy binary mask files to process
    
    outpath: str
        Output path to which mask images should be stored
    
    imgformat: str
        Image file format (e.g., 'png', 'jpg', 'bmp') for output figures
    '''
    N_masks = len(mask_list)
    logger = setup_logger_stdout()
    for i, mask_file in enumerate(mask_list):
        try:
            mask = np.load(mask_file).squeeze()
        except:
            logger.error('%s not found.' %(mask_file))
            continue
        if i%10==0 or i==N_masks-1:
           logger.info('Reading mask file %d out of %d'% (i+1, N_masks))
        basename = mask_file.split('/')[-1] if '/' in mask_file else mask_file
        basename = basename.split('.npy')[0]
        # Map [0, 1] to [0, 255]. RGB = (0, 0, 0) for black and RGB = (255, 255, 255) for white.
        mask *= 255
        mask = Image.fromarray(mask)
        mask.save(outpath + '/' + basename + '_rgb.' + imgformat)

##############################################################
def main() -> None:
    # Help description
    usage = 'Render binary segmentation masks in human-interpretable RGB color format.'
    parser = ArgumentParser(description=usage)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-f', action='store', required=True, dest='imgformat', type=str,
                            help="File format of output RGB images")   
    required.add_argument('-m', action='store', required=True, dest='maskpath', type=str,
                            help="Path to segmentation masks")        
    required.add_argument('-o', action='store', required=True, dest='outpath', type=str,
                            help="Output path (will be created if non-existent)") 
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit("Missing one or more required input arguments")    
    
    parse_args = parser.parse_args()
    mask_list = sorted(glob.glob(parse_args.maskpath+'/*.npy'))
    create_path(parse_args.outpath)
    render_binary_mask(mask_list, parse_args.outpath, parse_args.imgformat)
##############################################################
if __name__=='__main__':
    main()
##############################################################