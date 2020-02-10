#!/bin/env python

# standard library
import os
import sys
import argparse
import ast

# dependencies
from importlib import import_module

import numpy as np
import pandas as pd
from skimage import img_as_ubyte
import toolz as tz
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler


# local imports
from . import config
from . import io
from . import screens
from .screens import cellomics
from . import preprocess as pre
from . import cluster
from .io import temporary_hdf5_dataset


parser = argparse.ArgumentParser(description="Run the microscopium functions.")
subpar = parser.add_subparsers()


def get_command(argv):
    """Get the command name used from the command line.

    Parameters
    ----------
    argv : [string]
        The argument vector.

    Returns
    -------
    cmd : string
        The command name.
    """
    return argv[1]


def main():
    """Run the command-line interface."""
    args = parser.parse_args()
    cmd = get_command(sys.argv)
    if cmd == 'crop':
        run_crop(args)
    elif cmd == 'mask':
        run_mask(args)
    elif cmd == 'illum':
        run_illum(args)
    elif cmd == 'montage':
        run_montage(args)
    elif cmd == 'features':
        run_features(args)
    else:
        sys.stderr.write("Error: command %s not found. Run %s -h for help." %
                         (cmd, sys.argv[0]))
        sys.exit(2) # 2 is commonly a command-line error


crop = subpar.add_parser('crop', help="Crop images.")
crop.add_argument('crops', nargs=4, metavar='INT',
                  help='xstart, xstop, ystart, ystop. "None" also allowed.')
crop.add_argument('images', nargs='+', metavar='IM', help="The input images.")
crop.add_argument('-o', '--output-suffix',
                  default='.crop.tif', metavar='SUFFIX',
                  help="What suffix to attach to the cropped images.")
crop.add_argument('-O', '--output-dir',
                  help="Directory in which to output the cropped images.")
crop.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                  help='Use TIFF compression in the range 0 (no compression) '
                       'to 9 (max compression, slowest) (default 1).')
def run_crop(args):
    """Run image cropping."""
    crops = []
    for c in args.crops:
        try:
            crops.append(int(c))
        except ValueError:
            crops.append(None)
    xstart, xstop, ystart, ystop = crops
    slices = (slice(xstart, xstop), slice(ystart, ystop))
    for imfn in args.images:
        im = io.imread(imfn)
        imout = pre.crop(im, slices)
        fnout = os.path.splitext(imfn)[0] + args.output_suffix
        if args.output_dir is not None:
            fnout = os.path.join(args.output_dir, os.path.split(fnout)[1])
        io.imsave(fnout, imout, compress=args.compress)


mask = subpar.add_parser('mask', help="Estimate a mask over image artifacts.")
mask.add_argument('images', nargs='+', metavar='IM', help="The input images.")
mask.add_argument('-o', '--offset', metavar='INT', default=0, type=int,
                  help='Offset the automatic mask threshold by this amount.')
mask.add_argument('-v', '--verbose', action='store_true',
                  help='Print runtime information to stdout.')
mask.add_argument('-c', '--close', metavar='RADIUS', default=0, type=int,
                  help='Perform morphological closing of masks of this radius.')
mask.add_argument('-e', '--erode', metavar='RADIUS', default=0, type=int,
                  help='Perform morphological erosion of masks of this radius.')
def run_mask(args):
    """Run mask generation."""
    n, m = pre.write_max_masks(args.images, args.offset, args.close, args.erode)
    if args.verbose:
        print("%i masks created out of %i images processed" % (n, m))


illum = subpar.add_parser('illum',
                          help="Estimate and correct illumination.")
illum.add_argument('images', nargs='*', metavar='IMG', default=[],
                   help="The input images.")
illum.add_argument('-f', '--file-list', type=lambda x: open(x, 'r'),
                   metavar='FN',
                   help='Text file with one image filename per line.')
illum.add_argument('-o', '--output-suffix',
                   default='.illum.tif', metavar='SUFFIX',
                   help="What suffix to attach to the corrected images.")
illum.add_argument('-l', '--stretchlim', metavar='[0.0-1.0]', type=float,
                   default=0.0, help='Stretch image range before all else.')
illum.add_argument('-L', '--stretchlim-output', metavar='[0.0-1.0]', type=float,
                   default=0.0, help='Stretch image range before output.')
illum.add_argument('-q', '--quantile', metavar='[0.0-1.0]', type=float,
                   default=0.05,
                   help='Use this quantile to determine illumination.')
illum.add_argument('-i', '--input-bitdepth', type=int, default=None,
                   help='Input bit-depth of images.')
illum.add_argument('-r', '--radius', metavar='INT', type=int, default=51,
                   help='Radius in which to find quantile.')
illum.add_argument('-s', '--save-illumination', metavar='FN',
                   help='Save the illumination field to a file.')
illum.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                   help='Use TIFF compression in the range 0 (no compression) '
                        'to 9 (max compression, slowest) (default 1).')
illum.add_argument('-v', '--verbose', action='store_true',
                   help='Print runtime information to stdout.')
illum.add_argument('--random-seed', type=int, default=None,
                   help='The random seed for sampling illumination image.')
def run_illum(args):
    """Run illumination correction.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """
    if args.file_list is not None:
        args.images.extend([fn.rstrip() for fn in args.file_list])
    il = pre.find_background_illumination(args.images, args.radius,
                                          args.input_bitdepth, args.quantile,
                                          args.stretchlim)
    if args.verbose:
        print('illumination field:', type(il), il.dtype, il.min(), il.max())
    if args.save_illumination is not None:
        io.imsave(args.save_illumination, il / il.max())
    base_fns = [pre.basefn(fn) for fn in args.images]
    ims_out = [fn + args.output_suffix for fn in base_fns]
    corrected = pre.correct_multiimage_illumination(args.images, il,
                                                    args.stretchlim_output,
                                                    args.random_seed)
    for im, fout in zip(corrected, ims_out):
        io.imsave(fout, im, compress=args.compress)


montage = subpar.add_parser('montage',
                            help='Montage and channel stack images.')
montage.add_argument('images', nargs='*', metavar='IM',
                     help="The input images.")
montage.add_argument('-c', '--compress', metavar='INT', type=int, default=1,
                     help='Use TIFF compression in the range 0 '
                          '(no compression) '
                          'to 9 (max compression, slowest) (default 1).')
montage.add_argument('-o', '--montage-order', type=ast.literal_eval,
                     default=cellomics.SPIRAL_CLOCKWISE_RIGHT_25,
                     help='The shape of the final montage.')
montage.add_argument('-O', '--channel-order', type=ast.literal_eval,
                     default=[0, 1, 2],
                     help='The position of red, green, and blue channels '
                          'in the stream.')
montage.add_argument('-s', '--suffix', default='.montage.tif',
                     help='The suffix for saved images after conversion.')
montage.add_argument('-d', '--output-dir', default=None,
                     help='The output directory for the images. Defaults to '
                          'the input directory.')
def run_montage(args):
    """Run montaging and channel concatenation.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """
    ims = map(io.imread, args.images)
    ims_out = pre.montage_stream(ims, montage_order=args.montage_order,
                                 channel_order=args.channel_order)
    def out_fn(fn):
        sem = cellomics.cellomics_semantic_filename(fn)
        out_fn = '_'.join([str(sem[k])
                           for k in sem
                           if k not in ['field', 'channel', 'suffix']
                           and sem[k] != ''])
        outdir = (args.output_dir if args.output_dir is not None
                  else sem['directory'])
        out = os.path.join(outdir, out_fn) + args.suffix
        return out
    step = np.array(args.montage_order).size * len(args.channel_order)
    out_fns = (out_fn(fn) for fn in args.images[::step])
    for im, fn in zip(ims_out, out_fns):
        try:
            io.imsave(fn, im, compress=args.compress)
        except ValueError:
            im = img_as_ubyte(pre.stretchlim(im, 0.001, 0.999))
            io.imsave(fn, im, compress=args.compress)


features = subpar.add_parser('features',
                             help="Map images to feature vectors.")
features.add_argument('source', type=str,
                      help="CSV of image URLs")
features.add_argument('-S', '--settings', type=str, default=None,
                      help='Settings file')

def run_features(args):
    """Run image feature computation and dimension reduction on batch of images using yaml settings.

    Parameters
    ----------
    args : argparse.Namespace
        The arguments parsed by the argparse library.
    """

    # load config
    settings = config.load_config(args.settings)

    # load images from location specified in input CSV
    current_dir = os.path.dirname(args.settings)
    src = pd.read_csv(args.source)
    src['url'] = src['url'].apply(lambda x: os.path.join(current_dir, x))
    ims = map(io.imread, src['url'])

    # import specified feature mapping function
    p, m = settings['feature-computation']['feature-function']['function'].rsplit('.', 1)
    feature_mod = import_module(p)
    feature_map = getattr(feature_mod, m)

    out_path = settings['global']['output-dir']
    print(current_dir)
    print(out_path)
    if out_path == '':
        # make sister directory to input data
        if not os.path.isdir(os.path.join(current_dir, 'output')):
            os.mkdir(os.path.join(current_dir, 'output'))
            out_path = os.path.join(current_dir, 'output')
    elif os.path.isabs(out_path):
        # make absolute path to features.csv
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    else:
        # make path relative to parent directory of input
        cur_path = os.path.split(os.path.abspath(current_dir))
        out_path = os.path.join(cur_path[0], out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    out_path = os.path.join(out_path, 'features.csv')

    # for each image, compute features (as per config function), save
    image_features = output_features(ims, src['url'], out_path, feature_map)

    # temporarily load output csv so as not to compute features

    # strip column names and filenames column leaving just features

    # scale features using StandardScaler().fit_transform(stripped_features.to_numpy())

    # import chosen dim reduction functions

    # perform dim reductions on scaled values

    # concatenate initial csv df with the transformed coords


def output_features(ims, filenames, out_file, feature_map):
    """Build a feature map for each image in ims and output a dataframe of
    [filenames, features] to out_file as csv for reading in.


    Parameters
    ----------
    ims: 3D np.ndarray of float or uint8.
        the input images
    filenames: python string list
        filenames corresponding to each image in ims with relative path to top level directory
    out_file: string
        name of CSV file to save dataframe, with relative path to top level directory
    feature_map: function
        feature map function to apply

    Returns
    -------
    all_image_features: pandas DataFrame
        raw image features
    """
    # generate filenames column to exist as first column of feature DF
    filenames_col = ["Filenames"]
    filenames_col.extend(filenames)
    filenames_col = pd.DataFrame(filenames_col)

    all_image_features = []
    feature_names = []
    i = len(filenames)
    for im in ims:
        image_features, feature_names = feature_map(im, sample_size=100)
        all_image_features.append(image_features)
        print("Image processed, {} images remaining.".format(i - 1))
        i -= 1

    # convert to dataframe and assign column names
    all_image_features = pd.DataFrame(all_image_features)
    all_image_features.columns = feature_names

    # concatenate filenames column to the features and save to CSV.
    all_image_features = pd.concat([filenames_col, all_image_features], axis=1)
    all_image_features.to_csv(out_file)

    return all_image_features


if __name__ == '__main__':
    main()

