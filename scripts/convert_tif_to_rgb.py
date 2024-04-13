#!/usr/bin/env python
'''
Convert .tif images into a RGB file format for visualization and ease of annotation with Roboflow Annotate.

Run scipt using the following syntax.
python convert_tif_to_rgb.py -tif <Path to .tif images> 
                             -out <Path to which output data products must be saved>
                             -format <Output image format>
'''
import glob, sys
from PIL import Image
from argparse import ArgumentParser
from general_utils import create_path, setup_logger_stdout
##############################################################
def convert_tif(tif_list: list[str], outpath: str, format: str) -> None:
    '''
    Reads in. tif images one by one and writes the image data into a .png file.

    Parameters:
    -----------------------------
    tif_list: list[str]
        List of .tif images to read
    
    outpath: str
        Output path
    
    format: str
        Output file format. For example, 'png', 'jpg'
    '''
    logger = setup_logger_stdout()
    for i, tif_file in enumerate(tif_list):
        basename = tif_file.split('/')[-1] if '/' in tif_file else tif_file
        basename = basename.split('.tif')[0]
        try:
            tif_im = Image.open(tif_file)
            if i%10==0 or i==len(tif_list)-1:
                logger.info('Processing .tif file %d out of %d'% (i+1, len(tif_list)))
            rgb_im = tif_im.convert("RGB")
            rgb_im.save(outpath+'/'+basename+ '.'+format)
        except:
            logging.error('%s not found.'% (tif_file))
##############################################################
def main() -> None:
    # Help description
    usage = 'Convert .tif images into a RGB file format for visualization and ease of annotation with Roboflow Annotate.'
    parser = ArgumentParser(description=usage)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-tif', action='store', required=True, dest='tif_glob', type=str,
                            help="Glob string (including path) to parse .tif images")
    required.add_argument('-out', action='store', required=True, dest='outpath', type=str,
                            help="Output path (will be created if non-existent)") 
    required.add_argument('-format', action='store', required=True, dest='format', type=str,
                            help="Output image format (e.g., png, jpg)")    
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit("Missing one or more required input arguments")    
    
    parse_args = parser.parse_args()
    tif_list = sorted(glob.glob(parse_args.tif_glob+'/*.tif'))
    outpath = parse_args.outpath
    format = parse_args.format
    create_path(outpath)

    convert_tif(tif_list, outpath, format)
##############################################################
if __name__=='__main__':
    main()
##############################################################