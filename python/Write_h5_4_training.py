import os
import numpy as np
from osgeo import gdal, osr
import tables
import pandas as pd
import rasterio
from rasterio.enums import Resampling
import argparse

# Create the parser
parser = argparse.ArgumentParser(description='Process some paths.')

# Add the arguments
parser.add_argument('--dir_path', type=str, required=True, help='The directory path')
parser.add_argument('--out_path', type=str, required=True, help='The output path')
parser.add_argument('--out_h5_path', type=str, required=True, help='The output h5 path')

# Parse the arguments
args = parser.parse_args()

# Now you can use args.dir_path, args.out_path, args.out_h5_path, args.start, and args.end in your script
dir_path = args.dir_path
out_path = args.out_path
out_h5_path = args.out_h5_path

print(dir_path)
print(out_path)
print(out_h5_path)


# Desired image size
width = 256
height = 256

def resample_images(dir_path, out_path, width=256, height=256):
    tif_files = [f for f in os.listdir(dir_path) if f.endswith('.tif')]
    sliced_tif_files = tif_files
    for tif_file in sliced_tif_files:
        with rasterio.open(os.path.join(dir_path, tif_file)) as src:
            transform = src.transform * src.transform.scale(
                (src.width / width),
                (src.height / height)
            )
            metadata = src.meta.copy()
            metadata.update({
                'driver': 'GTiff',
                'height': height,
                'width': width,
                'transform': transform
            })
            image = src.read(
                out_shape=(src.count, height, width),
                resampling=Resampling.bilinear
            )
            with rasterio.open(os.path.join(out_path, tif_file), 'w', **metadata) as dst:
                dst.write(image)

resample_images(args.dir_path, args.out_path)

try:
    for filename in os.listdir(dir_path):
        file_path = os.path.join(dir_path, filename)
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    print(f"All files in {dir_path} have been removed.")
except Exception as e:
    print(f"Error occurred while removing files in {dir_path}: {e}")

#--------------------------------------------------

def init_hdf5_file(hdf5_path, patch_size, channels, projection, geotransform, gt_attributes, expectedrows=1000,
                   complib=None, complevel=0, subgroups=None, num_samples_chunk=64, bitshuffle=False, shuffle=True):

    def init_arrays(group):
        # init the storage:
        hdf5_file.create_earray(group, 'shot_number', tables.UInt64Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)
        hdf5_file.create_earray(group, 'images', tables.UInt16Atom(), shape=img_shape, chunkshape=img_chunk_shape,
                                expectedrows=expectedrows, filters=filters)

        for attribute in gt_attributes:
            hdf5_file.create_earray(group, attribute, tables.Float32Atom(), shape=label_shape,
                                    chunkshape=label_chunk_shape, expectedrows=expectedrows, filters=filters)

        # save patch top-left pixel location with respect to original tile.
        hdf5_file.create_earray(group, 'x_topleft', tables.UInt32Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)
        hdf5_file.create_earray(group, 'y_topleft', tables.UInt32Atom(), shape=(0,), chunkshape=(num_samples_chunk,),
                                expectedrows=expectedrows)

        hdf5_file.create_earray(group, 'lat', tables.Float32Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)
        hdf5_file.create_earray(group, 'lon', tables.Float32Atom(), shape=label_shape, chunkshape=label_chunk_shape,
                                expectedrows=expectedrows, filters=filters)

        # save georeference for the original sentinel2 tile
        hdf5_file.create_group(group, 'georef', 'Georeference of the Sentinel-2 tile')

        if projection is not None:
            group.georef._v_attrs.projection = projection
        if geotransform is not None:
            group.georef._v_attrs.geotransform = geotransform

    img_shape = (0, patch_size, patch_size, channels)
    label_shape = (0, patch_size, patch_size, 1)

    # most efficient for reading single patches
    img_chunk_shape = (num_samples_chunk, patch_size, patch_size, channels)
    label_chunk_shape = (num_samples_chunk, patch_size, patch_size, 1)

    # filter: compression complib can be: None, 'zlib', 'lzo', 'blosc'
    if complib:
        filters = tables.Filters(complevel=complevel, complib=complib, bitshuffle=bitshuffle, shuffle=shuffle)
    else:
        filters = tables.Filters(complevel=0)

    # open a hdf5 file and create earrays
    with tables.open_file(hdf5_path, mode='w') as hdf5_file:
        if subgroups is None:
            # init arrays in root group
            init_arrays(group=hdf5_file.root)
        else:
            # init arrays in subgroups (e.g. used for two-step pile shuffling)
            for subgroup in subgroups:
                hdf5_file.create_group(hdf5_file.root, subgroup, 'Pile')
                init_arrays(group=hdf5_file.root[subgroup])

    hdf5_file.close()

patch_size = 256
channels = 12
gt_attributes = ('wildness', 'count')
expectedrows = 10000
complib = None
complevel = 0
subgroups = None
num_samples_chunk = 64
bitshuffle = False
shuffle = True

init_hdf5_file(hdf5_path=out_h5_path, patch_size=patch_size, channels=channels,
               projection=None, geotransform=None, gt_attributes=gt_attributes,
               expectedrows=expectedrows, complib=complib, complevel=complevel,
               subgroups=subgroups, num_samples_chunk=num_samples_chunk, bitshuffle=bitshuffle, shuffle=shuffle)

def sort_band_arrays(band_arrays, channels_last=True):
    bands = ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5", "Band_6", "Band_7", "Band_8", "Band_9", "Band_11", "Band_12", "Band_13"]
    out_arr = []
    for b in bands:
        out_arr.append(band_arrays[b])
    out_arr = np.array(out_arr)
    if channels_last:
        out_arr = np.moveaxis(out_arr, source=0, destination=-1)
    return out_arr

def to_latlon(x, y, ds):
    bag_gtrn = ds.GetGeoTransform()
    bag_proj = ds.GetProjectionRef()
    bag_srs = osr.SpatialReference(bag_proj)
    geo_srs = bag_srs.CloneGeogCS()
    if int(gdal.__version__[0]) >= 3:
        # GDAL 3 changes axis order: https://github.com/OSGeo/gdal/issues/1546
        bag_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        geo_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(bag_srs, geo_srs)

    # in a north up image:
    originX = bag_gtrn[0]
    originY = bag_gtrn[3]
    pixelWidth = bag_gtrn[1]
    pixelHeight = bag_gtrn[5]

    easting = originX + pixelWidth * x + bag_gtrn[2] * y
    northing = originY + bag_gtrn[4] * x + pixelHeight * y

    geo_pt = transform.TransformPoint(easting, northing)[:2]
    lon = geo_pt[0]
    lat = geo_pt[1]
    return lat, lon

def create_latlon_mask(height, width, refDataset, out_type=np.float32):
    # compute lat, lon of top-left and bottom-right corners
    lat_topleft, lon_topleft = to_latlon(x=0, y=0, ds=refDataset)
    lat_bottomright, lon_bottomright = to_latlon(x=width-1, y=height-1, ds=refDataset)

    # interpolate between the corners
    lat_col = np.linspace(start=lat_topleft, stop=lat_bottomright, num=height).astype(out_type)
    lon_row = np.linspace(start=lon_topleft, stop=lon_bottomright, num=width).astype(out_type)

    # expand dimensions of row and col vector to repeat
    lat_col = lat_col[:, None]
    lon_row = lon_row[None, :]

    # repeat column and row to get 2d arrays --> lat lon coordinate for every pixel
    lat_mask = np.repeat(lat_col, repeats=width, axis=1)
    lon_mask = np.repeat(lon_row, repeats=height, axis=0)

    print('lat_mask.shape: ', lat_mask.shape)
    print('lon_mask.shape: ', lon_mask.shape)

    return lat_mask, lon_mask

# List of band names
band_names = ["Band_1", "Band_2", "Band_3", "Band_4", "Band_5", "Band_6", "Band_7", "Band_8", "Band_9", "Band_11", "Band_12", "Band_13"]

# Dictionary to hold the band arrays for each file
band_array_dict = {}
latlon_patches_dict = {}

# Iterate over all files in the directory
for filename in os.listdir(out_path):
    # Check if the file is a .tif file
    if filename.endswith('.tif'):
        # Extract the FID from the filename
        fid = filename.split('-')[0]

        # Full path to the file
        file_path = os.path.join(out_path, filename)

        # Open the file with GDAL
        ds = gdal.Open(file_path)

        # Dictionary to hold the band arrays for this file
        band_arrays = {}

        # Read each band into an array
        for i, band_name in enumerate(band_names, start=1):
            band = ds.GetRasterBand(i)
            array = band.ReadAsArray()
            band_arrays[band_name] = array

        # Create latlon masks
        lat_mask, lon_mask = create_latlon_mask(ds.RasterYSize, ds.RasterXSize, ds)

        latlon_patches = {}

        # Add latlon masks to the band arrays
        latlon_patches['lat'] = lat_mask
        latlon_patches['lon'] = lon_mask
        # Get the top left latitude and longitude
        lat_topleft, lon_topleft = to_latlon(x=0, y=0, ds=ds)

        # Add the top left latitude and longitude to the band arrays
        band_arrays['y_topleft'] = lat_topleft
        band_arrays['x_topleft'] = lon_topleft

        # Add the band arrays for this file to the main dictionary
        band_array_dict[str(fid)] = band_arrays

        latlon_patches_dict[str(fid)] = latlon_patches

def write_patches_to_hdf(hdf5_path, band_arrays, image_date, image_name, latlon_patches=None, label_patches_dict=None):
    # the hdf5 file must already exist.
    with tables.open_file(hdf5_path, mode='r+') as hdf5_file:
        images_storage = hdf5_file.root.images
        lat_storage = hdf5_file.root.lat
        lon_storage = hdf5_file.root.lon
        shot_number_storage = hdf5_file.root.shot_number
        x_topleft_storage = hdf5_file.root.x_topleft
        y_topleft_storage = hdf5_file.root.y_topleft

        print('images_storage before:', type(images_storage), images_storage.dtype, images_storage.shape)
        print('lat_storage before:', type(lat_storage), lat_storage.dtype, lat_storage.shape)
        print('lon_storage before:', type(lon_storage), lon_storage.dtype, lon_storage.shape)
        if label_patches_dict is not None:
            for attribute in label_patches_dict.keys():
                if attribute in hdf5_file.root:
                    print('{}_storage before:'.format(attribute), type(hdf5_file.root[attribute]),
                        hdf5_file.root[attribute].dtype, hdf5_file.root[attribute].shape)

        count_skipped = 0
        for p_id in band_arrays:
            # Note: the shape of img, label must be 4-dim (1, patch, patch, channels)
            img_patch = np.expand_dims(sort_band_arrays(band_arrays=band_arrays[p_id]), axis=0)

            # skip the image patch if it contains any empty pixels (all bands equal zero)
            band_sum = np.sum(img_patch, axis=2)
            if (band_sum == 0).any():
                count_skipped += 1
                continue

            print(img_patch.shape)

            images_storage.append(img_patch)
            # Extract numeric part of p_id
            numeric_p_id = int(''.join(filter(str.isdigit, p_id)))
            shot_number_storage.append(np.array([numeric_p_id]))
            #shot_number_storage.append(np.array([p_id]))

            x_topleft_storage.append(np.array([band_arrays[p_id]['x_topleft']]))
            y_topleft_storage.append(np.array([band_arrays[p_id]['y_topleft']]))

            if label_patches_dict is not None:
                for attribute in label_patches_dict.keys():
                    label_patch = np.expand_dims(np.expand_dims(label_patches_dict[attribute][p_id], axis=-1), axis=0)
                    hdf5_file.root[attribute].append(label_patch)

            if latlon_patches is not None:
                lat_patch = np.expand_dims(np.expand_dims(latlon_patches[p_id]['lat'], axis=-1), axis=0)
                lon_patch = np.expand_dims(np.expand_dims(latlon_patches[p_id]['lon'], axis=-1), axis=0)
                lat_storage.append(lat_patch)
                lon_storage.append(lon_patch)

        print('images_storage after:', type(images_storage), images_storage.dtype, images_storage.shape)
        print('lat_storage after:', type(lat_storage), lat_storage.dtype, lat_storage.shape)
        print('lon_storage after:', type(lon_storage), lon_storage.dtype, lon_storage.shape)
        if label_patches_dict is not None:
            for attribute in label_patches_dict.keys():
                if attribute in hdf5_file.root:
                    print('{}_storage after:'.format(attribute), type(hdf5_file.root[attribute]),
                         hdf5_file.root[attribute].dtype, hdf5_file.root[attribute].shape)

        print('number of skipped patches: ', count_skipped)
    hdf5_file.close()

write_patches_to_hdf(out_h5_path, band_array_dict, 'image_date', 'image_name', latlon_patches_dict)
