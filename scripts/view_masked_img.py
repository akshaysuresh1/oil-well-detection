#!/usr/bin/env python
'''
Visualize images after application of binary semantic segmentation masks built using Roboflow Annotate.

python view_masked_img.py -f <RGB image file format> -i <Path to RGB image files>  -m <Path to segmentation masks> -o <Output path>
'''
import cv2
import glob, sys
import numpy as np
from argparse import ArgumentParser
from general_utils import create_path, setup_logger_stdout
from PIL import Image
##############################################################
def apply_mask(img_list: list[str], maskpath: str, outpath: str, imgformat: str) -> None:
    '''
    Create image view after application of a binary semantic segmentation mask.

    Parameters:
    -----------------------------
    img_list: str
        List of RGB image files
    
    maskpath: str
        Path to mask files
    
    outpath: str
        Output path
    
    imgformat: str
        Image file format
    '''
    N_files = len(img_list)
    logger = setup_logger_stdout()
    for i, img_file in enumerate(img_list):
        try:
            # The cv2 package assumes image data in BGR channel ordering by default. Reverse BGR to RGB by flipping the last array index.
            im = cv2.imread(img_file)[...,::-1]
        except:
            logger.error('%s not found.' %(img_file))
            continue
        if i%10==0 or i==N_files-1:
            logger.info('Reading image file %d out of %d'% (i+1, N_files))
        basename = img_file.split('/')[-1] if '/' in img_file else img_file
        basename = basename.split('.'+imgformat)[0]
        mask_file = maskpath+'/'+basename+'_mask.npy'
        try:
            mask = np.load(mask_file)
        except:
            logger.error('Mask file not found for %s.'% (img_file))
            continue
        # Duplicate single channel data to 3 channels to mimic RGB format.
        mask = np.concatenate((mask, mask, mask), axis=-1)
        # Set masked areas to black.
        im[mask==0] = 0
        im = Image.fromarray(im)
        im.save(outpath+'/'+basename+'_masked.'+imgformat)

##############################################################
def main() -> None:
    # Help description
    usage = 'Visualize images after application of binary semantic segmentation masks built using Roboflow Annotate.'
    parser = ArgumentParser(description=usage)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-f', action='store', required=True, dest='imgformat', type=str,
                            help="File format of RGB images")   
    required.add_argument('-i', action='store', required=True, dest='imgpath', type=str,
                            help="Path to RGB image files")
    required.add_argument('-m', action='store', required=True, dest='maskpath', type=str,
                            help="Path to binary segmentation masks")        
    required.add_argument('-o', action='store', required=True, dest='outpath', type=str,
                            help="Output path (will be created if non-existent)") 
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit("Missing one or more required input arguments")    
    
    parse_args = parser.parse_args()
    img_list = sorted(glob.glob(parse_args.imgpath+'/*.'+parse_args.imgformat))
    create_path(parse_args.outpath)
    apply_mask(img_list, parse_args.maskpath, parse_args.outpath, parse_args.imgformat)
##############################################################
if __name__=='__main__':
    main()
##############################################################