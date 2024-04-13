#!/usr/bin/env python
'''
This script reads in a collection of .tif images and a .csv file of labeled oil well locations (latitude and longitude). 
For every input .tif image, the script returns a .csv file of labeled oil wells that lie within the image boundary. 
In addition, the script generates a summary .csv file listing the count of labeled oil wells per image.

Run scipt using the following syntax.
python parse_tif.py -tif <Path to .tif images> 
                    -loc <Name (including path) of .csv file of labeled oil well locations> 
                    -out <Path to which output data products must be saved>

Example (paths applicable when running script from repository root directory):
python scripts/parse_tif.py -tif data_raw/images -loc data_raw/well_locations.csv -out eda
'''
from __future__ import print_function
from __future__ import absolute_import
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio as rs
import glob, sys
from argparse import ArgumentParser
from general_utils import create_path, setup_logger_stdout
from rasterio.plot import show
##############################################################
def plot_tif(tif_img_src: rs.DatasetReader, tif_wells_lon: list[float], tif_wells_lat: list[float], outpath: str, basename: str) -> None:
    '''
    Generate a raster image of a. tif file ans save it to disk. Light blue scatter points highlight labeled wells contained inside the .tif boundary.

    Parameters:
    -----------------------------
    tif_img_src: rs.DatasetReader
        Rasterio dataset object of .tif image

    tif_wells_lon: list[float]
        Longitudes of labeled wells

    tif_wells_lat: list[float]
        Latitudes of labeled wells
    
    outpath: str
        Output path for plot
    
    basename: str
        Basename for output plot
    '''
    left, bottom, right, top = tif_img_src.bounds
    extent = [left, right, bottom, top]
    fig, ax = plt.subplots(nrows=1,ncols=1, num=1, tight_layout=True)
    show(tif_img_src, extent=extent, ax=ax, cmap='pink')
    plt.scatter(tif_wells_lon, tif_wells_lat, marker='o', color='aqua', s=10)
    plt.axis('off')
    plt.savefig(outpath+'/'+basename+'_wells.png',bbox_inches='tight')
    plt.close()
    
def process_tif(tif_list: list[str], wells_df: pd.DataFrame, outpath: str) -> None:
    '''
    Reads in. tif images one by one and checks for labeled wells that fall within the image boundary.

    Parameters:
    -----------------------------
    tif_list: list[str]
        List of .tif images to read

    wells_df: pd.DataFrame
        Pandas DataFrame of labeled oil well locations (latitude and longitude)
    
    outpath: str
        Output path
    '''
    N_wells = len(wells_df)
    wells_longitude = wells_df['lon'].to_numpy()    
    wells_latitude = wells_df['lat'].to_numpy()
    # Store count of labeled wells contained in every .tif image.
    well_counts = np.zeros(len(tif_list), dtype=int)
    # Set up logger.
    logger = setup_logger_stdout()
    for i, tif_img in enumerate(tif_list):
        if i%10==0 or i==len(tif_list)-1:
            logger.info('Processing .tif file %d out of %d'% (i+1, len(tif_list)))
        basename = tif_img.split('/')[-1] if '/' in tif_img else tif_img
        basename = basename.split('.tif')[0]
        # Read in .tif image.
        src = rs.open(tif_img)
        lon_min, lat_min, lon_max, lat_max = src.bounds
        # Store longitude and latitude of wells contained inside .tif image boundary.
        tif_wells_lon = []
        tif_wells_lat = []
        # Mark well pixel coordinates (both absolute and relative) relative to top right corner.
        # Normalized coordinates are obtained by dividing absolute x (or y) by the image width (or height).
        x_wells = []
        y_wells = [] 
        x_scaled_wells = []
        y_scaled_wells = []
        # Count labeled wells inside each image grid.
        for j in range(N_wells):
            if lon_min<=wells_longitude[j]<=lon_max and lat_min<=wells_latitude[j]<=lat_max:
                well_counts[i] += 1
                tif_wells_lon.append(wells_longitude[j])
                tif_wells_lat.append(wells_latitude[j])
                x, y = src.index(x=wells_longitude[j], y=wells_latitude[j])
                x_wells.append(x)
                y_wells.append(y)
                x_scaled_wells.append(x/src.width)
                y_scaled_wells.append(y/src.height)
        # Plot spatial distribution of wells in each image.
        plot_tif(src, tif_wells_lon, tif_wells_lat, outpath, basename)
        src.close()
        # Prepare .csv file of wells with locations contained inside .tif frame.
        tif_wells_df = pd.DataFrame({'lon':tif_wells_lon, 'lat':tif_wells_lat, 'x':x_wells, 'y':y_wells, 'x_scaled':x_scaled_wells, 'y_scaled':y_scaled_wells})
        tif_wells_df.to_csv(outpath+'/'+basename+'_wells.csv', index=False)      
    # Prepare a summary file storing labeled well counts in each .tif image.
    summary_df = pd.DataFrame({'Image':tif_list, 'Well count':well_counts})
    summary_df.to_csv(outpath+'/'+'summary.csv',index=False)
##############################################################
def main() -> None:
    # Help description
    usage = 'This script reads in a collection of .tif images one-by-one and checks for labeled oil wells enclosed within each .tif image boundary.'
    parser = ArgumentParser(description=usage)
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    required.add_argument('-tif', action='store', required=True, dest='tif_glob', type=str,
                            help="Glob string (including path) to parse .tif images")
    required.add_argument('-loc', action='store', required=True, dest='wells_csv', type=str,
                            help="Name (including path) of .csv file of labeled oil well locations")
    required.add_argument('-out', action='store', required=True, dest='outpath', type=str,
                            help="Output path (will be created if non-existent)")    
    parser._action_groups.append(optional)

    if len(sys.argv)==1:
        parser.print_help()
        sys.exit("Missing one or more required input arguments")
    
    # Read arguments from parser.
    parse_args = parser.parse_args()
    tif_list = sorted(glob.glob(parse_args.tif_glob+'/*.tif'))
    wells_df = pd.read_csv(parse_args.wells_csv)
    outpath = parse_args.outpath
    create_path(outpath)

    process_tif(tif_list, wells_df, outpath)
##############################################################
if __name__=='__main__':
    main()
##############################################################