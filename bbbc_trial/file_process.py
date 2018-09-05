import pandas as pd
import os

from sklearn.decomposition import PCA

from microscopium import io
from microscopium.preprocess import montage_stream
from microscopium.preprocess import correct_multiimage_illumination
from microscopium.preprocess import find_background_illumination
from microscopium.features import default_feature_map


#  C07- G11 : decompression issues
#  D05- E04 : SE image issues
#  G09      : NW image issues
BAD_GROUPS = ["C07", "E05", "E06", "E07", "F03", "G11", "D05", "C11", "E04", "G09"]

FILE_NAME_PREFIX = "Week1_150607_"
IMAGE_FILE_PATH = "../bbbc_trial/Week1_22123/"
OUTPUT_FILE_PATH = "../bbbc_trial/out/"

FEATURES_FILE = "../bbbc_trial/allFeatures.csv"
DATA_FILE = OUTPUT_FILE_PATH + "Data.csv"

# Use 50 to process all images in Week1_22123 dataset
NUM_IMAGES = 8


def main():

    ## get valid filenames and build output filenames
    file_name_groups = make_groups(NUM_IMAGES)
    filenames = get_valid_file_names(file_name_groups)

    # concatenate names including file path to output directory, keeping channel and quadrant information
    names_illum = [OUTPUT_FILE_PATH + filename[26:48] + "_illum.tif" for filename in filenames]
    # illum images
    run_illum(filenames, names_illum)

    ## montage illumed images
    names_montage = [OUTPUT_FILE_PATH + FILE_NAME_PREFIX + group + "_montaged.tif" for group in file_name_groups]
    run_montage(sorted(names_illum), names_montage)

    ## run features on images
    ims = map(io.imread, names_montage)
    output_features(ims, names_montage, FEATURES_FILE)

    ## get x y coordinates
    coords = pca_transform(FEATURES_FILE)

    ## generate CSV of coordinates
    generate_bokeh_csv(coords, file_name_groups, names_montage)




def get_valid_file_names(file_name_groups):
    """
    Get full filename relative to top level directory for each file in the BBBC trial

    :param file_name_groups: list of valid well IDs e.g. ["B02", "B03", "D03"]
    :return filenames: list of filenames relative to top level directory
    """
    filename_prefixes = [FILE_NAME_PREFIX + group for group in file_name_groups]
    filenames = []
    for filename in os.listdir(IMAGE_FILE_PATH):
        for prefix in filename_prefixes:
            if filename.startswith(prefix):
                filenames.append(IMAGE_FILE_PATH + filename)
    return filenames


def run_illum(filenames, names_out):
    """
    Find background illumination and correct all images corresponding to elements in filenames.

    Save corrected images using names_out which includes a relative path from the top level directory.

    :param filenames: list of valid filenames with relative paths from top level directory
    :param names_out: list of valid filenames for saving output with relative paths from top level directory
    """
    illum = find_background_illumination(filenames)
    corrected_images = correct_multiimage_illumination(filenames, illum=illum)
    for (image, name) in zip(corrected_images, names_out):
        io.imsave(name, image)


def run_montage(filenames, names_out):
    """
    Read images from filenames and stitch and stack their quadrants and channels before saving to new files using
    names_out

    :param filenames: list of filenames with relative paths to top level sorted by well, quadrant and channel e.g.
                        filenames = ['B02_s1_w1_illum.tif', 'B02_s1_w2_illum.tif', 'B02_s1_w4_illum.tif',
                                    'B02_s2_w1_illum.tif', 'B02_s2_w2_illum.tif', 'B02_s2_w4_illum.tif',
                                    'B02_s3_w1_illum.tif', 'B02_s3_w2_illum.tif', 'B02_s3_w4_illum.tif',
                                    'B02_s4_w1_illum.tif', 'B02_s4_w2_illum.tif', 'B02_s4_w4_illum.tif']
                        will result in one image (B02) with quadrants [[s1, s2], [s3, s4]] where each quadrant
                        is stacked in the order [w4, w2, w1]. This example assumes files at the top level directory
    :param names_out: list of filenames with relative paths to top level for output
    """
    illumed_ims = map(io.imread, filenames)
    montaged_ims = montage_stream(illumed_ims, montage_order=[[0, 1], [2, 3]], channel_order=[2, 1, 0])
    for (image, name) in zip(montaged_ims, names_out):
        io.imsave(name, image)


def output_features(ims, filenames, out_file):
    """
    Build a default feature map for each image in ims and output a dataframe of
    [filenames, features] to out_file as csv for reading in

    :param ims: opened nparray images
    :param filenames: filenames corresponding to each image in ims with relative path to top level directory
    :param out_file: name of CSV file to save dataframe, with relative path to top level directory
    """
    # generate filenames column to exist as first column of feature DF
    filenames_col = ["Filenames"]
    filenames_col.extend(filenames)
    filenames_col = pd.DataFrame(filenames_col)

    all_image_features = pd.DataFrame()
    # set up flag to only add header row once
    flag = True
    for im, im_name in zip(ims, filenames):
        image_features, feature_names = default_feature_map(im)
        # make sure header row is added to dataframe in first iteration
        if flag:
            all_image_features = all_image_features.append(pd.DataFrame(feature_names).transpose())
            flag = False
        image_features = pd.DataFrame(image_features).transpose()
        all_image_features = all_image_features.append(image_features, ignore_index=True)

    # concatenate filenames column to the features and save to CSV.
    all_image_features = pd.concat([filenames_col, all_image_features], axis=1)
    all_image_features.to_csv(out_file)


def make_groups(num_images):
    """
    Concatenate strings corresponding to the filename IDs in the BBBC trial dataset.
    Will generate as many groups as the number of images requested

    :return: list of filename groups e.g. ["B02", "B02", "D03"]
    """
    file_name_groups = []
    for letter in "BCDEFG":
        for num in range(2, 12):
            group = letter + "{:02}".format(num)
            if group not in BAD_GROUPS:
                file_name_groups.append(group)
                if len(file_name_groups) == num_images:
                    return file_name_groups
    return file_name_groups


def pca_transform(features_filename):
    """
    Read a file of image features into dataframe and perform a 2 component PCA, returning the 2 component values
    of each image

    :param features_filename: filename of CSV containing image features
    :return coords: np array of 2 components for each image
    """
    all_image_features = pd.read_csv(features_filename)
    pca = PCA(2)
    coords = pca.fit_transform(all_image_features.iloc[1:, 2:])

    return coords


def generate_bokeh_csv(coords, file_name_groups, names):
    """
    Generate a CSV of columns
        index,info,url,x,y
    to work with Bokeh app.

    :param coords: the x,y components of each data point
    :param file_name_groups: the valid filename IDs generated for this application e.g. ["B02", "B03", "D02"]
    :param names: the names of the images you wish to load into bokeh, relative to the top level directory
    """
    coords_df = pd.DataFrame(coords)

    indices = pd.DataFrame([FILE_NAME_PREFIX + group for group in file_name_groups])
    info = pd.DataFrame([FILE_NAME_PREFIX + group + "_info" for group in file_name_groups])
    # strip relative path from filename since CSV will be stored in same folder
    urls = pd.DataFrame([name[18:] for name in names])

    coord_csv = pd.concat([indices, info, urls, coords_df], axis=1)
    coord_csv.columns = ["index", "info", "url", "x", "y"]
    coord_csv.to_csv(DATA_FILE)

main()
