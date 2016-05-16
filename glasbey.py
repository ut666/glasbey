#!/usr/bin/env python
# encoding: utf-8

import sys
import argparse
import numpy as np
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_objects import LabColor, sRGBColor
from colormath.color_conversions import convert_color
from colorspacious import cspace_convert
from view_palette import palette_to_image

try:
    from progressbar import Bar, ETA, Percentage, ProgressBar
except ImportError:
    class Bar: pass
    class ETA: pass
    class Percentage: pass
    class ProgressBar:
        def __init__(self, **kwargs): pass
        def start(self): return self
        def update(self, i): pass
        def finish(self): pass


MAX = 256
NUM_COLORS = MAX * MAX * MAX


def lab_from_rgb(r, g, b):
    rgb = sRGBColor(r, g, b, is_upscaled=True)
    lab = convert_color(rgb, LabColor, target_illuminant='d50')
    return lab.get_value_tuple()


def rgb_from_lab(lab):
    """
    Convert a color from CIELab space into RGB space. The RGB components are
    upscaled (in [0..255] interval).
    Parameters
    ----------
    lab : list, tuple, or numpy array
        A 3-element array with L, a, and b components of the color.
    """
    lab = LabColor(lab[0], lab[1], lab[2])
    rgb = convert_color(lab, sRGBColor, target_illuminant='d50')
    return tuple(round(k * 255) for k in rgb.get_value_tuple())

def generate_color_table():
    """
    Generate a lookup table with all possible RGB colors, encoded in
    perceptually uniform CAM02-UCS color space.
    Table rows correspond to individual RGB colors, columns correspond to J',
    a', and b' components. The table is stored as a NumPy array.
    """

    widgets = ['Generating color table: ',
               Percentage(), ' ',
               Bar(), ' ',
               ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=(MAX * MAX)).start()

    i = 0
    colors = np.empty(shape=(NUM_COLORS, 3), dtype=float)
    for r in range(MAX):
        for g in range(MAX):
            d = i * MAX
            for b in range(MAX):
                colors[d + b, :] = (r, g, b)
            colors[d:d + MAX] = cspace_convert(colors[d:d + MAX],
                                               'sRGB255',
                                               'CAM02-UCS')
            pbar.update(i)
            i += 1
    pbar.finish()
    return colors

def generate_palette(colors, size, base=None, no_black=False, no_white=False):
    # Initialize palette with given palette or a base color
    if base:
        palette = [colors[i, :] for i in base]
    else:
        rgb = (255, 0, 0)	#our starting color
        lab = cspace_convert(rgb,'sRGB255', 'CAM02-UCS')
        palette = [lab]
    	
    #how far from white or black do we exclude
	# Exclude colors that are close to black
    if no_black:
        MIN_DISTANCE_TO_BLACK = 45
        d = np.linalg.norm((colors - cspace_convert((0, 0, 0),'sRGB255','CAM02-UCS')), axis=1)
        colors = colors[d > MIN_DISTANCE_TO_BLACK, :]    # Exclude colors that are close to black
    if no_white:
        MIN_DISTANCE_TO_WHITE = 45
        d = np.linalg.norm((colors - cspace_convert((255, 255, 255),'sRGB255','CAM02-UCS')), axis=1)
        colors = colors[d > MIN_DISTANCE_TO_WHITE, :]    # Exclude colors that are close to white
	# Initialize distances array
    num_colors = colors.shape[0]
    distances = np.ones(shape=(num_colors, 1)) * 1000
    # A function to recompute minimum distances from palette to all colors
    def update_distances(colors, color):
        d = np.linalg.norm((colors - color), axis=1)
        np.minimum(distances, d.reshape(distances.shape), distances)
    # Build progress bar
    widgets = ['Generating palette: ',
               Percentage(), ' ',
               Bar(), ' ',
               ETA()]
    pbar = ProgressBar(widgets=widgets, maxval=size).start()
    # Update distances for the colors that are already in the palette
    for i in range(len(palette) - 1):
        update_distances(colors, palette[i])
        pbar.update(i)
    # Iteratively build palette
    while len(palette) < size:
        update_distances(colors, palette[-1])
        palette.append(colors[np.argmax(distances), :])
        pbar.update(len(palette))
    pbar.finish()
    return cspace_convert(palette, 'CAM02-UCS', 'sRGB1')


def load_palette(f):
    palette = list()
    for line in f.readlines():
        rgb = [int(c) for c in line.strip().split(',')]
        palette.append((rgb[0] * 256 + rgb[1]) * 256 + rgb[2])
    return palette

def save_palette(palette, f, fmt):
    if fmt == 'byte':
        for color in palette:
            rgb255 = tuple(int(round(k * 255)) for k in color)
            f.write('{},{},{}\n'.format(*rgb255))
    else:
        for color in palette:
            f.write('{:.6f},{:.6f},{:.6f}\n'.format(*(abs(k) for k in color)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='''
    Generate a color palette using the sequential method of Glasbey et al.ยน
    This script needs an RGB to Lab color lookup table. Generation of this
    table is a time-consuming process, therefore the first run of this script
    will take some time. The generated table will be stored and automatically
    used in next invocations of the script. Note that the approximate size of
    the table is 363 Mb.
    The palette generation method allows the user to supply a base palette. The
    output palette will begin with the colors from the supplied set. If no base
    palette is given, then white will be used as the first base color. The base
    palette should be given as a text file where each line contains a color
    description in RGB format with components separated with commas. (See files
    in the 'palettes/' folder for an example).
    If having black (and colors close to black) is undesired, then `--no-black`
    option may be used to prevent the algorithm from inserting such colors into
    the palette.
    ยน) Glasbey, C., van der Heijden, G., Toh, V. F. K. and Gray, A. (2007),
       Colour Displays for Categorical Images.
       Color Research and Application, 32: 304-309
    ''', formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--base-palette', type=argparse.FileType('r'),
                        help='file with base palette')
    parser.add_argument('--no-black', action='store_true',
                        help='avoid black and similar colors')
    parser.add_argument('--no-white', action='store_true',
                        help='avoid white and similar colors')
    parser.add_argument('--view', action='store_true',
                        help='view generated palette')
    parser.add_argument('--format', default='byte',
                        help='output format (byte or float)')
    parser.add_argument('size', type=int,
                        help='number of colors in the palette')
    parser.add_argument('output', type=argparse.FileType('w'),
                        help='output palette filename')
    args = parser.parse_args()

    if not args.format in ['byte', 'float']:
        sys.exit('Invalid output format "{}"'.format(args.format))

    # Load base palette
    base = load_palette(args.base_palette) if args.base_palette else None

    # Load or generate RGB to CAM02-UCS color lookup table
    LUT = 'rgb_cam02ucs_lut.npz'
    try:
        colors = np.load(LUT)['lut']
        # Sanity check
        assert colors.shape == (NUM_COLORS, 3)
    except:
        colors = generate_color_table()
        np.savez_compressed(LUT, lut=colors)

    palette = generate_palette(colors, args.size, base, no_black=args.no_black, no_white=args.no_white)
    save_palette(palette, args.output, args.format)

    if args.view:
        img = palette_to_image(palette)
        img.show()
