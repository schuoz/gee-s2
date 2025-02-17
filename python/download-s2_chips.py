import ee
import logging
import multiprocessing
import os
import requests
import shutil
from retry import retry
import pandas as pd
import numpy as np
import datetime
import sys
import time
import argparse
from multiprocessing import freeze_support

from downloaded_gee_table_preprosessor import preprocess_sparse_table
import ctypes

def set_process_name(name):
    libc = ctypes.cdll.LoadLibrary('libc.so.6')
    libc.prctl(15, name.encode('utf-8'), 0, 0, 0)

# Set the process name
set_process_name(f"download_{os.getpid()}")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process some paths.')
    parser.add_argument('--output_dir', type=str, required=True, help='The output tiff path')
    parser.add_argument('--csv_file', type=str, required=True, help='The csv file path')
    parser.add_argument('--log_meesage_txt', type=str, required=True, help='log_meesage_txt')
    parser.add_argument('--year', type=str, required=True, help='year')
    return parser.parse_args()

# Parse the arguments
argments = parse_arguments()

# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.ERROR)

service_account = ''
credentials = ee.ServiceAccountCredentials(service_account, '')
ee.Initialize(credentials)

# LOCATION OF CSV FILE with coordinates
DATA = argments.csv_file
year = argments.year
output_dir = argments.output_dir
log_meesage_txt = argments.log_meesage_txt
# NUMBER OF FILES TO DOWNLOAD
# set to None to download all files
DOWNLOAD_NB_FILES = None

# RADIUS AROUND COORD IN METERS
# This is the number of meter around the point coordinate to include in the picture
RADIUS_AROUND = 1280

# MULTISPECTRAL OR RGB
ALL_BANDS = True

# RANGES FOR BANDS - Only for RGB images
# Values tested on different location in nepal
# Hypothesis: Should be the same for all images
RANGE_MIN = 0
RANGE_MAX = 2000

# RANGE FOR DATES
# CSV shows one date per observation.
# However, we need to take several pictures to select ones without clouds
# We look for RANDE_DATE weeks around the date

def determine_md_and_range(lat):
    if lat > 60:
        # Northern polar area
        md = '-07-01'
        RANGE = 4
    elif lat < -60:
        # Southern polar area
        md = '-01-01'
        RANGE = 4
    elif -29 <= lat <= 29:
        # Tropic area
        md = '-07-01'
        RANGE = 30
    elif lat > 29:
        # Northern than tropic area
        md = '-07-01'
        RANGE = 8
    else:
        # Southern than tropic area
        md = '-01-01'
        RANGE = 8
    return md, RANGE
df = pd.read_csv(DATA)
log_messages = []
print(f'{len(df)} images in the CSV file')
log_messages.append(f'{len(df)} images in the CSV file')
# print head of lat and lon
print(f'Original lat: {df["lat"].head()}')
log_messages.append(f'Original lat: {df["lat"].head()}')
print(f'Original lon: {df["lon"].head()}')
log_messages.append(f'Original lon: {df["lon"].head()}')

# Apply the function to each row in the dataframe
df[['md', 'RANGE']] = df.apply(lambda row: pd.Series(determine_md_and_range(row['lat'])), axis=1)

# Update TOI based on the 'year' column and 'md' column
df['TOI'] = pd.to_datetime(argments.year + df['md'])

# Convert TOI to the format '01/08/2023'
df['TOI'] = df['TOI'].dt.strftime('%d/%m/%Y')

#df['FID'] = df['player_id']
df['year'] = year
log_messages.append(f'Updated TOI: {df["TOI"].head()}')
print(f'Updated TOI: {df["TOI"].head()}')
print(f'Updated year: {df["year"].head()}')

dataset = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
csPlus = ee.ImageCollection('GOOGLE/CLOUD_SCORE_PLUS/V1/S2_HARMONIZED')

def get_image_name(index, longitude, latitude, year):
    return f'{str(index)}_{str(longitude)}_{str(latitude)}_{str(year)}'.replace('.', '-')

# Here, the position of lat and lon depends in the table, the lat and lon is correct or not, very important!!!
# if the lat lon label is correct, keep this order
coords = [(get_image_name(point[1][0], point[1][1], point[1][2], point[1][3]), point[1][1], point[1][2], point[1][3], point[1][4],point[1][5]) for point in df.loc[:, ['FID', 'lon', 'lat', 'year', 'TOI','RANGE']].iterrows()]

# Determine number of images to process
total_images = len(df)

@retry(tries=80, delay=2, backoff=2)
# Generate a rectangle containing the circle (centered on the coordinate) with radius RADIUS_AROUND
def get_geometry_radius(geometry_point):
    coord = np.array(geometry_point.getInfo()['coordinates'][0])
    return ee.Geometry.Rectangle([coord[:, 0].min(), coord[:, 1].min(), coord[:, 0].max(), coord[:, 1].max()])

# Generate the dates around the time of interest. Useful to have enough images to filter clouds
def date_range_to_collect(time_of_interest, RANDE_DATE, debug=False):
    d, m, y = time_of_interest.split('/')
    target_date = datetime.date(int(y), int(m), int(d))
    delta = datetime.timedelta(weeks=RANDE_DATE)
    return target_date-delta, target_date+delta
def generate_image(image_collection, image_name, x, y, year, toi, RANGE, params, log_messages, debug=True):
    try:
        if debug:
            message1 = f'Working on {image_name}: ({x}, {y})'
            print(message1)
            log_messages.append(f'{image_name},{y},{x},{message1},')
        geo = ee.Geometry.Point(x, y)
        radius = geo.buffer(params["buffer"])
        geometry_radius = get_geometry_radius(radius)

        spatialFiltered = image_collection.filterBounds(geo)

        date_range = date_range_to_collect(toi, RANGE, debug)
        if debug:
            message2 = f'date range: {str(date_range[0])} {str(date_range[1])}'
            print(message2)
            log_messages.append(f'date range: {str(date_range[0])} {str(date_range[1])}')

        temporalFiltered = spatialFiltered.filterDate(str(date_range[0]), str(date_range[1]))

        # Filter images with CLOUDY_PIXEL_PERCENTAGE below 10
        QA_BAND = 'cs'  # Example QA band, replace with the actual band name
        CLEAR_THRESHOLD = 0.5  # Example threshold, replace with the actual threshold
        # check if the band have cloudy pixel percentage
        filtered_images = temporalFiltered.filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 80)).linkCollection(csPlus, [QA_BAND]).map(lambda img: img.updateMask(img.select(QA_BAND).gte(CLEAR_THRESHOLD)))
        # Sort by CLOUDY_PIXEL_PERCENTAGE and take the first 6 images
        least_cloudy_images = filtered_images.sort('CLOUDY_PIXEL_PERCENTAGE').limit(50)

        # Get the list of images
        image_list = least_cloudy_images.toList(50)

        # Initialize variables to track the largest cloudiness and corresponding product ID
        max_cloudiness = -1
        max_product_id = None
        # Print the largest CLOUDY_PIXEL_PERCENTAGE and corresponding PRODUCT_ID
        # Take the median of the 6 images
        img = ee.ImageCollection(image_list).reduce(ee.Reducer.percentile([25])).select('B.+')
        #img = ee.ImageCollection(image_list).median().select('B.+')
    # TODO: change the resolution, the resolution now is 20
        url = img.getDownloadURL({
            'scale': params["scale"],
            'region': geometry_radius.getInfo()['coordinates'][0],
            # "dimensions": params["dimensions"],
            'format': params["format"]
        })

        r = requests.get(url, stream=True)
        if r.status_code != 200:
            r.raise_for_status()

        filename = f"{output_dir}/{image_name}.tif"
        with open(filename, "wb") as out_file:
            shutil.copyfileobj(r.raw, out_file)
        print("Done: ", image_name)
        log_messages.append(f"Done: {image_name}")
        print('Download URL: ', url)
        log_messages.append(f'Download URL: {url}')
    except ee.ee_exception.EEException as e:
        error_message = f"EEException for image: {e}"
        logging.error(error_message)
        print(error_message)
        log_messages.append(f'{image_name},{y},{x},{error_message},')
    except Exception as e:
        error_message = f"Unexpected error for image {image_name}: {e}"
        logging.error(error_message)
        print(error_message)
        log_messages.append(f'{image_name},{y},{x},{error_message},')

# Create output directory for the specific array job index
os.makedirs(output_dir, exist_ok=True)

def main():
    dataset_collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
    start_time = time.time()

    params = {
        "count": 1200000,  # How many image chips to export
        "buffer": 1280,  # The buffer distance (m) around each point
        "scale": 10,  # The scale to do stratified sampling
        "seed": 1,  # A randomization seed to use for subsampling.
        "dimensions": "256x256",  # The dimension of each image chip
        "format": "GEO_TIFF",  # The output image format, can be png, jpg, ZIPPED_GEO_TIFF, GEO_TIFF, NPY
        "processes": 24,  # How many processes to used for parallel processing
        "out_dir": output_dir,  # The output directory. Default to the current working directly
    }

    # Create a multiprocessing manager
    manager = multiprocessing.Manager()
    log_messages = manager.list()
    # Prepare arguments for generate_image function
    args = [(dataset_collection, point[0], point[1], point[2], point[3], point[4], point[5], params, log_messages, True) for point in coords[:params["count"]]]

    # Create a pool of processes
    with multiprocessing.Pool(params["processes"]) as p:
        # Use the pool to call generate_image function in parallel
        p.starmap(generate_image, args)

    # End time
    end_time = time.time()

    # Calculate total run time
    run_time = end_time - start_time

    # Count the number of images in the output folder
    folder = output_dir
    total_files = len([name for name in os.listdir(folder) if os.path.isfile(os.path.join(folder, name))])

    # Print finish time, run time, and number of images downloaded
    print(f"Process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    log_messages.append(f"Process finished at: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"Total running time: {run_time:.2f} seconds")
    log_messages.append(f"Total running time: {run_time:.2f} seconds")
    print(f"Total images downloaded: {total_files}")
    log_messages.append(f"Total images downloaded: {total_files}")

    # Save the log messages to a file
    with open(f"{log_meesage_txt}", 'w') as f:
        for line in log_messages:
            f.write(f"{line}\n")
if __name__ == '__main__':
    freeze_support()
    main()
