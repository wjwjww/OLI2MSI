import inspect
import math
import os
import re
import zipfile
import warnings
import sys
import time
from glob import glob
from functools import partial
from itertools import chain
from shutil import get_terminal_size
from datetime import datetime

import cv2
import pyproj
import numpy as np
from lxml.etree import parse, fromstring


import rasterio
from rasterio.io import MemoryFile
from rasterio.transform import xy, rowcol, Affine
from rasterio import features
from rasterio.coords import BoundingBox
import rasterio.warp as warp
from rasterio.transform import xy, rowcol

from shapely.geometry import Point, box, shape
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.ops import transform


class cached_property(object):
    """
    Decorator that converts a method with a single self argument into a
    property cached on the instance.
    Optional ``name`` argument allows you to make cached properties of other
    methods. (e.g.  url = cached_property(get_absolute_url, name='url') )
    """

    def __init__(self, func, name=None):
        self.func = func
        self.__doc__ = getattr(func, '__doc__')
        self.name = name or func.__name__

    def __get__(self, instance, cls=None):
        if instance is None:
            return self
        res = instance.__dict__[self.name] = self.func(instance)
        return res


class ProgressBar(object):
    """
    A progress bar which can print the progress
    modified from https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print('terminal width is too small ({}), please consider widen the terminal for better '
                  'progressbar visualization'.format(terminal_width))
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write('[{}] 0/{}, elapsed: 0s, ETA:\n{}\n'.format(
                ' ' * self.bar_width, self.task_num, 'Start...'))
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write('\033[J')  # clean the output (remove extra chars since last display)
            sys.stdout.write('[{}] {}/{}, {:.1f} task/s, elapsed: {}s, ETA: {:5}s\n{}\n'.format(
                bar_chars, self.completed, self.task_num, fps, int(elapsed + 0.5), eta, msg))
        else:
            sys.stdout.write('completed: {}, elapsed: {}s, {:.1f} tasks/s'.format(
                self.completed, int(elapsed + 0.5), fps))
        sys.stdout.flush()


# get image path list
IMG_EXTENSIONS = ['.TIF', '.tif', '.jp2']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_paths_from_l8_dir(path_dir):
    """get landsat8 image path list from image folder"""
    assert os.path.isdir(path_dir), '{:s} is not a valid directory'.format(path_dir)
    l8_file_ls = sorted(glob(os.path.join(path_dir, '*T1', '*_MTL.txt')))
    assert l8_file_ls, '{:s} has no valid image file'.format(path_dir)
    return l8_file_ls


def get_paths_from_s2_dir(path_dir):
    """get sentinel2 image path list from image folder"""
    assert os.path.isdir(path_dir), '{:s} is not a valid directory'.format(path_dir)
    s2_file_ls = sorted(glob(os.path.join(path_dir, '*.zip')))
    assert s2_file_ls, '{:s} has no valid image file'.format(path_dir)
    return s2_file_ls


def get_paths_from_images(path):
    """get image path list from image folder"""
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


def get_image_paths(data_type, dataroot):
    """get image path list
    support lmdb or image files"""
    paths, infos = None, None
    if dataroot is not None:
        if data_type == 'lmdb':
            paths, infos = get_paths_from_lmdb(dataroot)
        elif data_type == 'img':
            paths = sorted(get_paths_from_images(dataroot))
        else:
            raise NotImplementedError('data_type [{:s}] is not recognized.'.format(data_type))
    return paths, infos


# read images
def _read_img_lmdb(env, key, size):
    """read image from lmdb with key (w/ and w/o fixed size)
    size: (C, H, W) tuple"""
    with env.begin(write=False) as txn:
        buf = txn.get(key.encode('ascii'))
    img_flat = np.frombuffer(buf, dtype=np.float32)
    C, H, W = size
    img = img_flat.reshape(C, H, W)
    return img


def read_img(env, path, size=None):
    """read image by rasterio or from lmdb
    return: Numpy float32, CHW, RGB, normalized by mean and std"""
    if env is None:  # img
        with rasterio.open(path, 'r') as src:
            img = src.read()
    else:
        img = _read_img_lmdb(env, path, size)
    return img


def tif_to_png(tif_path, png_path=None, pmin=2.5, pmax=97.5):
    with rasterio.open(tif_path, 'r') as src:
        src_data = src.read()
    c, h, w = src_data.shape
    output = np.zeros((h, w, c), dtype='u1')
    for i in range(c):
        v_min = np.percentile(src_data[i], pmin)
        v_max = np.percentile(src_data[i], pmax)
        tem_arr = np.clip(src_data[i], v_min, v_max)
        output[:, :, i] = np.round((tem_arr - v_min) * 255. / (v_max - v_min)).astype('u1')
    save_path = png_path if png_path else tif_path.replace('.TIF', '.png')
    cv2.imwrite(save_path, output[:, :, ::-1], [cv2.IMWRITE_PNG_COMPRESSION, 3])
    # print("saving image to {}".format(save_path))


def sun_elevation(bounds, shape, date_collected, time_collected_utc):
    """
    Given a raster's bounds + dimensions, calculate the
    sun elevation angle in degrees for each input pixel
    based on metadata from a Landsat MTL file
    Parameters
    -----------
    bounds: BoundingBox
        bounding box of the input raster
    shape: tuple
        tuple of (rows, cols) or (depth, rows, cols) for input raster
    date_collected: str
        Format: YYYY-MM-DD
    time_collected_utc: str
        Format: HH:MM:SS.SSSSSSSSZ
    Returns
    --------
    ndarray
        ndarray with shape = (rows, cols) with sun elevation
        in degrees calculated for each pixel
    """
    utc_time = parse_utc_string(date_collected, time_collected_utc)

    if len(shape) == 3:
        _, rows, cols = shape
    else:
        rows, cols = shape

    lng, lat = _create_lnglats((rows, cols),
                               list(bounds))

    decimal_hour = time_to_dec_hour(utc_time)

    declination = calculate_declination(utc_time.timetuple().tm_yday)

    return _calculate_sun_elevation(lng, lat, declination,
                                    utc_time.timetuple().tm_yday,
                                    decimal_hour)


def parse_utc_string(collected_date, collected_time_utc):
    """
    Given a string in the format:
        YYYY-MM-DD HH:MM:SS.SSSSSSSSZ
    Parse and convert into a datetime object
    Fractional seconds are ignored
    Parameters
    -----------
    collected_date: str
        Format: YYYY-MM-DD
    collected_time_utc: str
        Format: HH:MM:SS.SSSSSSSSZ
    Returns
    --------
    datetime object
        parsed scene center time
    """
    utcstr = collected_date + ' ' + collected_time_utc

    if not re.match(r'\d{4}\-\d{2}\-\d{2}\ \d{2}\:\d{2}\:\d{2}\.\d+Z',
                    utcstr):
        raise ValueError("%s is an invalid utc time" % utcstr)

    return datetime.strptime(
        utcstr.split(".")[0],
        "%Y-%m-%d %H:%M:%S")


def _create_lnglats(shape, bbox):
    """
    Creates a (lng, lat) array tuple with cells that respectively
    represent a longitude and a latitude at that location
    Parameters
    -----------
    shape: tuple
        the shape of the arrays to create
    bbox: tuple or list
        the bounds of the arrays to create in [w, s, e, n]
    Returns
    --------
    (lngs, lats): tuple of (rows, cols) shape ndarrays
    """

    rows, cols = shape
    w, s, e, n = bbox
    xCell = (e - w) / float(cols)
    yCell = (n - s) / float(rows)

    lat, lng = np.indices(shape, dtype=np.float32)

    return ((lng * xCell) + w + (xCell / 2.0),
            (np.flipud(lat) * yCell) + s + (yCell / 2.0))


def time_to_dec_hour(parsedtime):
    """
    Calculate the decimal hour from a datetime object
    Parameters
    -----------
    parsedtime: datetime object
    Returns
    --------
    decimal hour: float
        time in decimal hours
    """
    return (parsedtime.hour +
            (parsedtime.minute / 60.0) +
            (parsedtime.second / 60.0 ** 2)
            )


def calculate_declination(d):
    """
    Calculate the declination of the sun in radians based on a given day.
    As reference +23.26 degrees at the northern summer solstice, -23.26
    degrees at the southern summer solstice.
    See: https://en.wikipedia.org/wiki/Position_of_the_Sun#Calculations
    Parameters
    -----------
    d: int
        days from midnight on January 1st
    Returns
    --------
    declination in radians: float
        the declination on day d
    """
    return np.arcsin(
                     np.sin(np.deg2rad(23.45)) *
                     np.sin(np.deg2rad(360. / 365.) *
                            (d - 81))
                    )


def _calculate_sun_elevation(longitude, latitude, declination, day, utc_hour):
    """
    Calculates the solar elevation angle
    https://en.wikipedia.org/wiki/Solar_zenith_angle
    Parameters
    -----------
    longitude: ndarray or float
        longitudes of the point(s) to compute solar angle for
    latitude: ndarray or float
        latitudes of the point(s) to compute solar angle for
    declination: float
        declination of the sun in radians
    day: int
        days of the year with jan 1 as day = 1
    utc_hour: float
        decimal hour from a datetime object
    Returns
    --------
    the solar elevation angle in degrees
    """
    hour_angle = np.deg2rad(solar_angle(day, utc_hour, longitude))

    latitude = np.deg2rad(latitude)

    return np.rad2deg(np.arcsin(
        np.sin(declination) *
        np.sin(latitude) +
        np.cos(declination) *
        np.cos(latitude) *
        np.cos(hour_angle)
    ))


def solar_angle(day, utc_hour, longitude):
    """
    Given a day, utc decimal hour, and longitudes, compute the solar angle
    for these longitudes
    Parameters
    -----------
    day: int
        days of the year with jan 1 as day = 1
    utc_hour: float
        decimal hour of the day in utc time to compute solar angle for
    longitude: ndarray or float
        longitude of the point(s) to compute solar angle for
    Returns
    --------
    solar angle in degrees for these longitudes
    """
    localtime = (longitude / 180.0) * 12 + utc_hour

    lstm = 15 * (localtime - utc_hour)

    B = np.deg2rad((360. / 365.) * (day - 81))

    eot = (9.87 *
           np.sin(2 * B) -
           7.53 * np.cos(B) -
           1.5 * np.sin(B))

    return 15 * (localtime +
                 (4 * (longitude - lstm) + eot) / 60.0 - 12)


# -------------------------------------------------------------
# landsat8 tiles reader

class MetaData(object):
    """
    A landsat metadata object. This class builds is attributes
    from the names of each tag in the xml formatted .MTL files that
    come with landsat data. So, any tag that appears in the MTL file
    will populate as an attribute of landsat_metadata.
    You can access explore these attributes by using, for example
    .. code-block:: python
        from dnppy import landsat
        meta = landsat.landsat_metadata(my_filepath) # create object
        from pprint import pprint                    # import pprint
        pprint(vars(m))                              # pretty print output
        scene_id = meta.LANDSAT_SCENE_ID             # access specific attribute
    :param filename: the filepath to an MTL file.

    copied from: https://github.com/NASA-DEVELOP/dnppy/tree/master/dnppy
    """

    def __init__(self, filename):
        """
        There are several critical attributes that keep a common
        naming convention between all landsat versions, so they are
        initialized in this class for good record keeping and reference
        """

        # custom attribute additions
        self.FILEPATH           = filename
        self.DATETIME_OBJ       = None

        # product metadata attributes
        self.LANDSAT_SCENE_ID   = None
        self.DATA_TYPE          = None
        self.ELEVATION_SOURCE   = None
        self.OUTPUT_FORMAT      = None
        self.SPACECRAFT_ID      = None
        self.SENSOR_ID          = None
        self.WRS_PATH           = None
        self.WRS_ROW            = None
        self.NADIR_OFFNADIR     = None
        self.TARGET_WRS_PATH    = None
        self.TARGET_WRS_ROW     = None
        self.DATE_ACQUIRED      = None
        self.SCENE_CENTER_TIME  = None

        # image attributes
        self.CLOUD_COVER        = None
        self.IMAGE_QUALITY_OLI  = None
        self.IMAGE_QUALITY_TIRS = None
        self.ROLL_ANGLE         = None
        self.SUN_AZIMUTH        = None
        self.SUN_ELEVATION      = None
        self.EARTH_SUN_DISTANCE = None    # calculated for Landsats before 8.

        # read the file and populate the MTL attributes
        self._read(filename)

    def _read(self, filename):
        """ reads the contents of an MTL file """

        # if the "filename" input is actually already a metadata class object, return it back.
        if inspect.isclass(filename):
            return filename

        fields = []
        values = []

        metafile = open(filename, 'r')
        metadata = metafile.readlines()

        for line in metadata:
            # skips lines that contain "bad flags" denoting useless data AND lines
            # greater than 1000 characters. 1000 character limit works around an odd LC5
            # issue where the metadata has 40,000+ characters of whitespace
            bad_flags = ["END", "GROUP"]
            if not any(x in line for x in bad_flags) and len(line) <= 1000:
                try:
                    line = line.replace("  ", "")
                    line = line.replace("\n", "")
                    field_name, field_value = line.split(' = ')
                    fields.append(field_name)
                    values.append(field_value)
                except:
                    pass

        for i in range(len(fields)):

            # format fields without quotes,dates, or times in them as floats
            if not any(['"' in values[i], 'DATE' in fields[i], 'TIME' in fields[i]]):
                setattr(self, fields[i], float(values[i]))
            else:
                values[i] = values[i].replace('"', '')
                setattr(self, fields[i], values[i])

        # create datetime_obj attribute (drop decimal seconds)
        dto_string          = self.DATE_ACQUIRED + self.SCENE_CENTER_TIME
        self.DATETIME_OBJ   = datetime.strptime(dto_string.split(".")[0], "%Y-%m-%d%H:%M:%S")

        # only landsat 8 includes sun-earth-distance in MTL file, so calculate it
        # for the Landsats 4,5,7 using solar module.
        # if not self.SPACECRAFT_ID == "LANDSAT_8":
        #
        #     # use 0s for lat and lon, sun_earth_distance is not a function of any one location on earth.
        #     s = Solar(0, 0, self.DATETIME_OBJ, 0)
        #     self.EARTH_SUN_DISTANCE = s.get_rad_vector()

        # print("Scene {0} center time is {1}".format(self.LANDSAT_SCENE_ID, self.DATETIME_OBJ))


class LandsatScene(object):
    """
    Defines a landsat scene object. Used to track band filepaths
    and pass raster objects to functions.
    band filepaths are read from the MTL file and stored as a dict.
    you may access them with
        from dnppy import landsat
        s = landsat.scene(MTL_path)
        s[1]        # the first band of scene "s"
        s[2]        # the second band of scene "s"
        s["QA"]     # the QA band of scene "s"
    Development Note:
        1) This is arcpy dependent, but in the future it should utilize
             custom functions that emulate RasterToNumPyArray and the dnppy metadata
             objects for all returns, thus eliminating all non-open source
             dependencies
        2) for good interchangability between landsat versions, it might be better
            to construct a dict whos keys are color codes or even wavelength values
            instead of band index numbers (which are do not correspond to similar colors
            between landsat missions)

    some codes are copied from: https://github.com/mapbox/rio-toa/tree/master/rio_toa
    """

    def __init__(self, MTL_path, tif_dir=None):
        """
        builds the scene.
        In some cases, users may have their MTL file located somewhere other than
        their landsat data. In this instance, users should input the path to
        the landsat images as tif_dir.
        """

        self.mtl_dir = os.path.dirname(MTL_path)  # directory of MTL file
        self.meta = MetaData(MTL_path)  # dnppy landsat_metadata object
        self.in_paths = {}  # dict of filepaths to tifs
        self.rasts = {}  # dict of arcpy raster objects
        self.profiles = {}

        if not tif_dir:
            self.tif_dir = self.mtl_dir

        self._find_bands()

    def read_data(self, idx):
        if not isinstance(idx, list):
            idx = [idx]
        for i in idx:
            tem_path = self.in_paths["B{0}".format(i)]
            if os.path.isfile(tem_path):
                if not "B{0}".format(i) in self.rasts:
                    with rasterio.open(tem_path) as src:
                        self.rasts["B{0}".format(i)] = src.read(1)
                        self.profiles["B{0}".format(i)] = src.profile
                        self.profiles["B{0}".format(i)]['bounds'] = src.bounds
                        self.profiles["B{0}".format(i)]['res'] = src.res
            else:
                raise NotImplementedError("file: {} doesn't exist".format(tem_path))

    def get_DN(self, idx):
        """ returns DN from indices """

        if not "B{0}".format(idx) in self.rasts:
            self.read_data(idx)

        return self.rasts["B{0}".format(idx)]

    def get_TOA(self, idx, per_pixel=True, clip=True):
        dn_arr = self.get_DN(idx)
        rows, cols = dn_arr.shape
        profile = self.profiles["B{0}".format(idx)]
        if per_pixel:
            col_min, row_max, col_max, row_min = 0, profile['height'], profile['width'], 0
            left, bottom = profile['transform'] * (col_min, row_max)
            right, top = profile['transform'] * (col_max, row_min)
            new_bound = warp.transform_bounds(profile['crs'], {'init': u'epsg:4326'}, left, bottom, right, top)
            bbox = BoundingBox(*new_bound)
            elevation = sun_elevation(bbox, (rows, cols), self.meta.DATE_ACQUIRED, self.meta.SCENE_CENTER_TIME)
        else:
            # We're doing whole-scene (instead of per-pixel) sun angle:
            elevation = self.meta.SUN_ELEVATION

        multi_f = getattr(self.meta, "REFLECTANCE_MULT_BAND_{0}".format(idx))  # multiplicative scaling factor
        add_f = getattr(self.meta, "REFLECTANCE_ADD_BAND_{0}".format(idx))  # additive rescaling factor

        toa = self._calculate_reflectance(dn_arr, multi_f, add_f, elevation)
        if clip:
            toa = np.clip(toa, 0, 1)
        return toa

    def xy(self, row, col, band, offset="center"):
        """Returns the coordinates ``(x, y)`` of a pixel at `row` and `col`.
        The pixel's center is returned by default, but a corner can be returned
        by setting `offset` to one of `ul, ur, ll, lr`.

        Parameters
        ----------
        row : int
            Pixel row.
        col : int
            Pixel column.
        band : int or str
            Determines the transform to be used, because different bands may have different transform
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.

        Returns
        -------
        tuple
            ``(x, y)``
        """
        if not "B{0}".format(band) in self.profiles:
            self.read_data(band)
        return xy(self.profiles["B{0}".format(band)]['transform'], row, col, offset=offset)

    def index(self, x, y, band, op=math.floor, precision=None):
        """
        Returns the (row, col) index of the pixel containing (x, y) given a
        coordinate reference system.

        Use an epsilon, magnitude determined by the precision parameter
        and sign determined by the op function:
            positive for floor, negative for ceil.

        Parameters
        ----------
        x : float
            x value in coordinate reference system
        y : float
            y value in coordinate reference system
        band : int or str
            Determines the transform to be used, because different bands may have different transform
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        precision : int, optional (default: None)
            Decimal places of precision in indexing, as in `round()`.

        Returns
        -------
        tuple
            (row index, col index)
        """
        if not "B{0}".format(band) in self.profiles:
            self.read_data(band)
        return rowcol(self.profiles["B{0}".format(band)]['transform'], x, y, op=op, precision=precision)

    @staticmethod
    def _calculate_reflectance(img, MR, AR, E, src_nodata=0):
        """Calculate top of atmosphere reflectance of Landsat 8
        as outlined here: http://landsat.usgs.gov/Landsat8_Using_Product.php
        R_raw = MR * Q + AR
        R = R_raw / cos(Z) = R_raw / sin(E)
        Z = 90 - E (in degrees)
        where:
            R_raw = TOA planetary reflectance, without correction for solar angle.
            R = TOA reflectance with a correction for the sun angle.
            MR = Band-specific multiplicative rescaling factor from the metadata
                (REFLECTANCE_MULT_BAND_x, where x is the band number)
            AR = Band-specific additive rescaling factor from the metadata
                (REFLECTANCE_ADD_BAND_x, where x is the band number)
            Q = Quantized and calibrated standard product pixel values (DN)
            E = Local sun elevation angle. The scene center sun elevation angle
                in degrees is provided in the metadata (SUN_ELEVATION).
            Z = Local solar zenith angle (same angle as E, but measured from the
                zenith instead of from the horizon).
        Parameters
        -----------
        img: ndarray
            array of input pixels of shape (rows, cols) or (rows, cols, depth)
        MR: float or list of floats
            multiplicative rescaling factor from scene metadata
        AR: float or list of floats
            additive rescaling factor from scene metadata
        E: float or numpy array of floats
            local sun elevation angle in degrees
        Returns
        --------
        ndarray:
            float32 ndarray with shape == input shape
        """

        if np.any(E < 0.0):
            raise ValueError("Sun elevation must be nonnegative "
                             "(sun must be above horizon for entire scene)")

        input_shape = img.shape

        if len(input_shape) > 2:
            img = np.rollaxis(img, 0, len(input_shape))

        rf = ((MR * img.astype(np.float32)) + AR) / np.sin(np.deg2rad(E))
        if src_nodata is not None:
            rf[img == src_nodata] = 0.0

        if len(input_shape) > 2:
            if np.rollaxis(rf, len(input_shape) - 1, 0).shape != input_shape:
                raise ValueError(
                    "Output shape %s is not equal to input shape %s"
                    % (rf.shape, input_shape))
            else:
                return np.rollaxis(rf, len(input_shape) - 1, 0)
        else:
            return rf

    def _find_bands(self):
        """
        builds filepaths for band filenames from the MTL file
        in the future, methods associated with landsat scenes,
        (for example: NDVI calculation) could save rasters in the scene
        directory with new suffixes then add attribute values to the MTL.txt
        file for easy reading later, such as:
        FILE_NAME_PROD_NDVI = [filepath to NDVI tif]
        """

        # build a band list based on the landsat version
        if self.meta.SPACECRAFT_ID == "LANDSAT_8":
            bandlist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]

            # add in landsat 8s QA band with shortened name
            QA_name = self.meta.FILE_NAME_BAND_QUALITY
            self.in_paths["BQA"] = os.path.join(self.tif_dir, QA_name)

        elif self.meta.SPACECRAFT_ID == "LANDSAT_7":
            bandlist = [1, 2, 3, 4, 5, "6_VCID_1", "6_VCID_2", 7, 8]

        elif self.meta.SPACECRAFT_ID == "LANDSAT_5":
            bandlist = [1, 2, 3, 4, 5, 6, 7]

        elif self.meta.SPACECRAFT_ID == "LANDSAT_4":
            bandlist = [1, 2, 3, 4, 5, 6, 7]

        else:
            raise ValueError

        # populate self.bands dict
        for band in bandlist:
            filename = getattr(self.meta, "FILE_NAME_BAND_{0}".format(band))
            filepath = os.path.join(self.tif_dir, filename)
            self.in_paths["B{0}".format(band)] = filepath

        return

    @staticmethod
    def _capture_bits(arr, b1, b2):
        width_int = int((b1 - b2 + 1) * "1", 2)
        return ((arr >> b2) & width_int).astype('uint8')

    def get_mask(self, kinds):
        qa_vars = {
            'fill': (0, 0),
            'terrain': (1, 1),
            'radiometricSaturation': (3, 2),
            'cloud': (4, 4),
            'cloudConf': (6, 5),
            'cirrusConf': (8, 7),
            'cloudShadowConf': (10, 9),
            'snowIceConf': (12, 11),
        }
        if not isinstance(kinds, list):
            kinds = [kinds]

        qa_arr = self.get_DN('QA')
        return [self._capture_bits(qa_arr, *qa_vars[kind]) for kind in kinds]

    def get_surface_reflectance(self):
        """ wraps dnppy.landsat.surface_reflecance functions and applies them to this scene"""

        pass

    def get_toa_reflectance(self):
        """ wraps dnppy.landsat.toa_reflectance functions and applies them to this scene"""

        pass

    def get_ndvi(self):
        """ wraps dnppy.landsat.ndvi functions and applies them to this scene"""

        pass

    def get_atsat_bright_temp(self):
        """ wraps dnppy.atsat_bright_temp functions and applies them to this scene"""

        pass


# -------------------------------------------------------------
"""
s2reader reads and processes Sentinel-2 L1C SAFE archives.
This module implements an easy abstraction to the SAFE data format used by the
Sentinel 2 misson of the European Space Agency (ESA)
most are copied from: https://github.com/ungarj/s2reader
"""


# sentinel2 granules reader
def s2_open(safe_file):
    """Return a SentinelDataSet object."""
    if os.path.isdir(safe_file) or os.path.isfile(safe_file):
        return SentinelDataSet(safe_file)
    else:
        raise IOError("file not found: %s" % safe_file)


BAND_IDS = [
    "01", "02", "03", "04", "05", "06", "07", "08", "8A", "09", "10",
    "11", "12"
]


class SentinelDataSet(object):
    """
    Return SentinelDataSet object.
    This object contains relevant metadata from the SAFE file and its
    containing granules as SentinelGranule() object.
    """

    def __init__(self, path):
        """Assert correct path and initialize."""
        filename, extension = os.path.splitext(os.path.normpath(path))
        if extension not in [".SAFE", ".ZIP", ".zip"]:
            raise IOError("only .SAFE folders or zipped .SAFE folders allowed")
        self.is_zip = True if extension in [".ZIP", ".zip"] else False
        self.path = os.path.normpath(path)

        if self.is_zip:
            self._zipfile = zipfile.ZipFile(self.path, 'r')
            self._zip_root = os.path.basename(filename)
            if self._zip_root not in self._zipfile.namelist():
                if not filename.endswith(".SAFE"):
                    self._zip_root = os.path.basename(filename) + ".SAFE/"
                else:
                    self._zip_root = os.path.basename(filename) + "/"
                if self._zip_root not in self._zipfile.namelist():
                    raise S2ReaderIOError("unknown zipfile structure")
            self.manifest_safe_path = os.path.join(
                self._zip_root, "manifest.safe")
        else:
            self._zipfile = None
            self._zip_root = None
            # Find manifest.safe.
            self.manifest_safe_path = os.path.join(self.path, "manifest.safe")

        if (
                not os.path.isfile(self.manifest_safe_path) and
                self.manifest_safe_path not in self._zipfile.namelist()
        ):
            raise S2ReaderIOError(
                "manifest.safe not found: %s" % self.manifest_safe_path
            )

    @cached_property
    def _product_metadata(self):
        if self.is_zip:
            return fromstring(self._zipfile.read(self.product_metadata_path))
        else:
            return parse(self.product_metadata_path)

    @cached_property
    def _manifest_safe(self):
        if self.is_zip:
            return fromstring(self._zipfile.read(self.manifest_safe_path))
        else:
            return parse(self.manifest_safe_path)

    @cached_property
    def product_metadata_path(self):
        """Return path to product metadata XML file."""
        data_object_section = self._manifest_safe.find("dataObjectSection")
        for data_object in data_object_section:
            # Find product metadata XML.
            if data_object.attrib.get("ID") == "S2_Level-1C_Product_Metadata":
                relpath = os.path.relpath(
                    next(data_object.iter("fileLocation")).attrib["href"])
                try:
                    if self.is_zip:
                        abspath = os.path.join(self._zip_root, relpath)
                        assert abspath in self._zipfile.namelist()
                    else:
                        abspath = os.path.join(self.path, relpath)
                        assert os.path.isfile(abspath)
                except AssertionError:
                    raise S2ReaderIOError(
                        "S2_Level-1C_product_metadata_path not found: {} ".format(abspath)
                    )
                return abspath

    @cached_property
    def product_ID(self):
        """Find and returns "Product Start Time"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.find("PRODUCT_URI").text[:-5]

    @cached_property
    def product_start_time(self):
        """Find and returns "Product Start Time"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.find("PRODUCT_START_TIME").text

    @cached_property
    def product_stop_time(self):
        """Find and returns the "Product Stop Time"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.find("PRODUCT_STOP_TIME").text

    @cached_property
    def generation_time(self):
        """Find and returns the "Generation Time"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.findtext("GENERATION_TIME")

    @cached_property
    def processing_level(self):
        """Find and returns the "Processing Level"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.findtext("PROCESSING_LEVEL")

    @cached_property
    def product_type(self):
        """Find and returns the "Product Type"."""
        for element in self._product_metadata.iter("Product_Info"):
            return element.findtext("PRODUCT_TYPE")

    @cached_property
    def spacecraft_name(self):
        """Find and returns the "Spacecraft name"."""
        for element in self._product_metadata.iter("Datatake"):
            return element.findtext("SPACECRAFT_NAME")

    @cached_property
    def sensing_orbit_number(self):
        """Find and returns the "Sensing orbit number"."""
        for element in self._product_metadata.iter("Datatake"):
            return element.findtext("SENSING_ORBIT_NUMBER")

    @cached_property
    def sensing_orbit_direction(self):
        """Find and returns the "Sensing orbit direction"."""
        for element in self._product_metadata.iter("Datatake"):
            return element.findtext("SENSING_ORBIT_DIRECTION")

    @cached_property
    def product_format(self):
        """Find and returns the Safe format."""
        for element in self._product_metadata.iter("Query_Options"):
            return element.findtext("PRODUCT_FORMAT")

    @cached_property
    def quantification_value(self):
        """Find and returns the Safe format."""
        for element in self._product_metadata.iter("Product_Image_Characteristics"):
            return int(element.findtext("QUANTIFICATION_VALUE"))

    @cached_property
    def footprint(self):
        """Return product footprint."""
        product_footprint = self._product_metadata.iter("Product_Footprint")
        # I don't know why two "Product_Footprint" items are found.
        for element in product_footprint:
            global_footprint = None
            for global_footprint in element.iter("Global_Footprint"):
                coords = global_footprint.findtext("EXT_POS_LIST").split()
                return _polygon_from_coords(coords)

    @cached_property
    def granules(self):
        """Return list of SentinelGranule objects."""
        for element in self._product_metadata.iter("Product_Info"):
            product_organisation = element.find("Product_Organisation")
        if self.product_format == 'SAFE':
            return [
                SentinelGranule(_id.find("Granules"), self)
                for _id in product_organisation.findall("Granule_List")
            ]
        elif self.product_format == 'SAFE_COMPACT':
            return [
                SentinelGranuleCompact(_id.find("Granule"), self)
                for _id in product_organisation.findall("Granule_List")
            ]
        else:
            raise Exception(
                "PRODUCT_FORMAT not recognized in metadata file, found: '" +
                str(self.product_format) +
                "' accepted are 'SAFE' and 'SAFE_COMPACT'"
            )

    def granule_paths(self, band_id):
        """Return the path of all granules of a given band."""
        band_id = str(band_id).zfill(2)
        try:
            assert isinstance(band_id, str)
            assert band_id in BAND_IDS
        except AssertionError:
            raise AttributeError(
                "band ID not valid: %s" % band_id
            )
        return [
            granule.band_path(band_id)
            for granule in self.granules
        ]

    def __enter__(self):
        """Return self."""
        return self

    def __exit__(self, t, v, tb):
        """Do cleanup."""
        try:
            self._zipfile.close()
        except AttributeError:
            pass


class S2ReaderIOError(IOError):
    """Raised if an expected file cannot be found."""


class S2ReaderMetadataError(Exception):
    """Raised if metadata structure is not as expected."""


class SentinelGranule(object):
    """This object contains relevant metadata from a granule."""

    def __init__(self, granule, dataset):
        """Prepare data paths depending on if ZIP or not."""
        self.dataset = dataset
        if self.dataset.is_zip:
            granules_path = os.path.join(self.dataset._zip_root, "GRANULE")
        else:
            granules_path = os.path.join(dataset.path, "GRANULE")
        self.granule_identifier = granule.attrib["granuleIdentifier"]
        self.granule_path = os.path.join(
            granules_path, self.granule_identifier)
        self.datastrip_identifier = granule.attrib["datastripIdentifier"]

        self.rasts = {}  # dict of arcpy raster objects
        self.profiles = {}

    def read_data(self, idx):
        if not isinstance(idx, list):
            idx = [idx]
        for i in idx:
            if self.dataset.is_zip:
                tem_path = self.band_path(i).replace('\\', '/')
                if tem_path in self.dataset._zipfile.namelist():
                    with MemoryFile(self.dataset._zipfile.read(tem_path)) as memfile:
                        with memfile.open(driver='JP2OpenJPEG') as src:
                            self.rasts["B{0}".format(i)] = src.read(1)
                            self.profiles["B{0}".format(i)] = src.profile
                            self.profiles["B{0}".format(i)]['bounds'] = src.bounds
                            self.profiles["B{0}".format(i)]['res'] = src.res
                else:
                    raise NotImplementedError("file was not found in the zip file: {}".format(tem_path))
            else:
                tem_path = self.band_path(i, absolute=True).replace('\\', '/')
                if os.path.isfile(tem_path):
                    if not "B{0}".format(i) in self.rasts:
                        with rasterio.open(tem_path, driver='JP2OpenJPEG') as src:
                            self.rasts["B{0}".format(i)] = src.read(1)
                            self.profiles["B{0}".format(i)] = src.profile
                            self.profiles["B{0}".format(i)]['bounds'] = src.bounds
                            self.profiles["B{0}".format(i)]['res'] = src.res
                else:
                    raise NotImplementedError("file was not found: {}".format(tem_path))

    def get_DN(self, idx):
        """ returns DN from indices """

        if not "B{0}".format(idx) in self.rasts:
            self.read_data(idx)

        return self.rasts["B{0}".format(idx)]

    def get_TOA(self, idx, clip=True):
        dn_arr = self.get_DN(idx).astype('f4')
        toa = dn_arr / self.dataset.quantification_value
        if clip:
            toa = np.clip(toa, 0, 1)
        return toa

    def get_cloud_mask(self, res=10.):
        cloud_geom = self.cloudmask
        info_dict = self.geo_info(res)
        t = Affine(res, 0., info_dict['ulx'],
                   0., -res, info_dict['uly'])
        clm = features.rasterize(cloud_geom, out_shape=(info_dict['nrows'], info_dict['ncols']),
                                 transform=t)
        return clm

    def xy(self, row, col, band, offset="center"):
        """Returns the coordinates ``(x, y)`` of a pixel at `row` and `col`.
        The pixel's center is returned by default, but a corner can be returned
        by setting `offset` to one of `ul, ur, ll, lr`.

        Parameters
        ----------
        row : int
            Pixel row.
        col : int
            Pixel column.
        band : int or str
            Determines the transform to be used, because different bands may have different transform
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.

        Returns
        -------
        tuple
            ``(x, y)``
        """
        if not "B{0}".format(band) in self.profiles:
            self.read_data(band)
        return xy(self.profiles["B{0}".format(band)]['transform'], row, col, offset=offset)

    def index(self, x, y, band, op=math.floor, precision=None):
        """
        Returns the (row, col) index of the pixel containing (x, y) given a
        coordinate reference system.

        Use an epsilon, magnitude determined by the precision parameter
        and sign determined by the op function:
            positive for floor, negative for ceil.

        Parameters
        ----------
        x : float
            x value in coordinate reference system
        y : float
            y value in coordinate reference system
        band : int or str
            Determines the transform to be used, because different bands may have different transform
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        precision : int, optional (default: None)
            Decimal places of precision in indexing, as in `round()`.

        Returns
        -------
        tuple
            (row index, col index)
        """
        if not "B{0}".format(band) in self.profiles:
            self.read_data(band)
        return rowcol(self.profiles["B{0}".format(band)]['transform'], x, y, op=op, precision=precision)

    def geo_info(self, res=10.):
        """Return upper left corner coordinate."""
        info_dict = {"res": res}
        elements = next(self._metadata.iter("Tile_Geocoding"))
        for e in elements.findall("Size"):
            if float(e.get('resolution')) == res:
                info_dict['nrows'] = int(e.findtext("NROWS"))
                info_dict['ncols'] = int(e.findtext("NCOLS"))
                break
        for e in elements.findall("Geoposition"):
            if float(e.get('resolution')) == res:
                info_dict['ulx'] = float(e.findtext("ULX"))
                info_dict['uly'] = float(e.findtext("ULY"))
                break
        return info_dict

    @cached_property
    def UL_PROJECTION_COORD(self, res=10.):
        info_dict = self.geo_info(res=res)
        return info_dict['ulx'], info_dict['uly']

    @cached_property
    def UR_PROJECTION_COORD(self, res=10.):
        info_dict = self.geo_info(res=res)
        t = Affine(res, 0., info_dict['ulx'],
                   0., -res, info_dict['uly'])
        return t*(info_dict['ncols'], 0)

    @cached_property
    def LL_PROJECTION_COORD(self, res=10.):
        info_dict = self.geo_info(res=res)
        t = Affine(res, 0., info_dict['ulx'],
                   0., -res, info_dict['uly'])
        return t*(0, info_dict['nrows'])

    @cached_property
    def LR_PROJECTION_COORD(self, res=10.):
        info_dict = self.geo_info(res=res)
        t = Affine(res, 0., info_dict['ulx'],
                   0., -res, info_dict['uly'])
        return t*(info_dict['ncols'], info_dict['nrows'])

    @cached_property
    def _metadata(self):
        if self.dataset.is_zip:
            return fromstring(self.dataset._zipfile.read(self.metadata_path))
        else:
            return parse(self.metadata_path)

    @cached_property
    def _nsmap(self):
        if self.dataset.is_zip:
            root = self._metadata
        else:
            root = self._metadata.getroot()
        return {
            k: v
            for k, v in root.nsmap.items()
            if k
        }

    @cached_property
    def srid(self):
        """Return EPSG code."""
        tile_geocoding = next(self._metadata.iter("Tile_Geocoding"))
        return tile_geocoding.findtext("HORIZONTAL_CS_CODE")

    @cached_property
    def sensing_time(self):
        for element in self._metadata.xpath("//SENSING_TIME"):
            time_str = element.text
            return datetime.strptime(time_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")

    @cached_property
    def tile_id(self):
        for element in self._metadata.xpath("//TILE_ID"):
            return element.text

    @cached_property
    def metadata_path(self):
        """Determine the metadata path."""
        xml_name = _granule_identifier_to_xml_name(self.granule_identifier)
        metadata_path = os.path.join(self.granule_path, xml_name)
        try:
            assert os.path.isfile(metadata_path) or \
                   metadata_path in self.dataset._zipfile.namelist()
        except AssertionError:
            raise S2ReaderIOError(
                "Granule metadata XML does not exist:", metadata_path)
        return metadata_path

    @cached_property
    def pvi_path(self):
        """Determine the PreView Image (PVI) path inside the SAFE pkg."""
        pvi_name = next(self._metadata.iter("PVI_FILENAME")).text
        pvi_name = pvi_name.split("/")
        pvi_path = os.path.join(
            self.granule_path,
            pvi_name[len(pvi_name) - 2], pvi_name[len(pvi_name) - 1]
        )
        try:
            assert os.path.isfile(pvi_path) or \
                   pvi_path in self.dataset._zipfile.namelist()
        except (AssertionError, AttributeError):
            return None
        return pvi_path

    @cached_property
    def tci_path(self):
        """Return the path to the granules TrueColorImage."""
        tci_paths = [
            path for path in self.dataset._product_metadata.xpath(
                ".//Granule[@granuleIdentifier='%s']/IMAGE_FILE/text()"
                % self.granule_identifier
            ) if path.endswith('TCI')
        ]
        try:
            tci_path = tci_paths[0]
        except IndexError:
            return None

        return os.path.join(
            self.dataset._zip_root if self.dataset.is_zip else self.dataset.path,
            tci_path
        ) + '.jp2'

    @cached_property
    def cloud_percent(self):
        """Return percentage of cloud coverage."""
        image_content_qi = self._metadata.findtext(
            (
                """n1:Quality_Indicators_Info/Image_Content_QI/"""
                """CLOUDY_PIXEL_PERCENTAGE"""
            ),
            namespaces=self._nsmap)
        return float(image_content_qi)

    @cached_property
    def footprint(self):
        """Find and return footprint as Shapely Polygon."""
        # Check whether product or granule footprint needs to be calculated.
        tile_geocoding = next(self._metadata.iter("Tile_Geocoding"))
        resolution = 10
        searchstring = ".//*[@resolution='%s']" % resolution
        size, geoposition = tile_geocoding.findall(searchstring)
        nrows, ncols = (int(i.text) for i in size)
        ulx, uly, xdim, ydim = (int(i.text) for i in geoposition)
        lrx = ulx + nrows * resolution
        lry = uly - ncols * resolution
        utm_footprint = box(ulx, lry, lrx, uly)
        project = partial(
            pyproj.transform,
            pyproj.Proj(init=self.srid),
            pyproj.Proj(init='EPSG:4326')
        )
        footprint = transform(project, utm_footprint).buffer(0)
        return footprint

    @cached_property
    def cloudmask(self):
        """Return cloudmask as a shapely geometry."""
        polys = list(self._get_mask(mask_type="MSK_CLOUDS"))
        return MultiPolygon([
            poly["geometry"]
            for poly in polys
            if poly["attributes"]["maskType"] == "OPAQUE"
        ]).buffer(0)

    @cached_property
    def nodata_mask(self):
        """Return nodata mask as a shapely geometry."""
        polys = list(self._get_mask(mask_type="MSK_NODATA"))
        return MultiPolygon([poly["geometry"] for poly in polys]).buffer(0)

    def band_path(self, band_id, for_gdal=False, absolute=False):
        """Return paths of given band's jp2 files for all granules."""
        band_id = str(band_id).zfill(2)
        if not isinstance(band_id, str) or band_id not in BAND_IDS:
            raise ValueError("band ID not valid: %s" % band_id)
        if self.dataset.is_zip and for_gdal:
            zip_prefix = "/vsizip/"
            if absolute:
                granule_basepath = zip_prefix + os.path.dirname(os.path.join(
                    self.dataset.path,
                    self.dataset.product_metadata_path
                ))
            else:
                granule_basepath = zip_prefix + os.path.dirname(
                    self.dataset.product_metadata_path
                )
        else:
            if absolute:
                granule_basepath = os.path.dirname(os.path.join(
                    self.dataset.path,
                    self.dataset.product_metadata_path
                ))
            else:
                granule_basepath = os.path.dirname(
                    self.dataset.product_metadata_path
                )
        product_org = next(self.dataset._product_metadata.iter(
            "Product_Organisation"))
        granule_item = [
            g
            for g in chain(*[gl for gl in product_org.iter("Granule_List")])
            if self.granule_identifier == g.attrib["granuleIdentifier"]
        ]
        if len(granule_item) != 1:
            raise S2ReaderMetadataError(
                "Granule ID cannot be found in product metadata."
            )
        rel_path = [
            f.text for f in granule_item[0].iter() if f.text[-2:] == band_id
        ]
        if len(rel_path) != 1:
            # Apparently some SAFE files don't contain all bands. In such a
            # case, raise a warning and return None.
            warnings.warn(
                "%s: image path to band %s could not be extracted" % (
                    self.dataset.path, band_id
                )
            )
            return
        img_path = os.path.join(granule_basepath, rel_path[0]) + ".jp2"
        # Above solution still fails on the "safe" test dataset. Therefore,
        # the path gets checked if it contains the IMG_DATA folder and if not,
        # try to guess the path from the old schema. Not happy with this but
        # couldn't find a better way yet.
        if "IMG_DATA" in img_path:
            return img_path
        else:
            if self.dataset.is_zip:
                zip_prefix = "/vsizip/"
                granule_basepath = zip_prefix + os.path.join(
                    self.dataset.path, self.granule_path)
            else:
                granule_basepath = self.granule_path
            return os.path.join(
                os.path.join(granule_basepath, "IMG_DATA"),
                "".join([
                    "_".join((self.granule_identifier).split("_")[:-1]),
                    "_B",
                    band_id,
                    ".jp2"
                ])
            )

    def _get_mask(self, mask_type=None, to_wgs84=False):
        if mask_type is None:
            raise ValueError("mask_type hast to be provided")
        exterior_str = str(
            "eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList"
        )
        interior_str = str(
            "eop:extentOf/gml:Polygon/gml:interior/gml:LinearRing/gml:posList"
        )
        for item in next(self._metadata.iter("Pixel_Level_QI")):               # next()
            if item.attrib.get("type") == mask_type:
                gml = os.path.join(
                    self.granule_path, "QI_DATA", os.path.basename(item.text)
                ).replace('\\', '/')

                if self.dataset.is_zip:
                    root = fromstring(self.dataset._zipfile.read(gml))
                else:
                    root = parse(gml).getroot()
                nsmap = {k: v for k, v in root.nsmap.items() if k}
                # try:
                for mask_member in root.iterfind(
                        "eop:maskMembers", namespaces=nsmap):
                    for feature in mask_member:
                        _type = feature.findtext(
                            "eop:maskType", namespaces=nsmap)

                        ext_elem = feature.find(exterior_str, nsmap)
                        dims = int(ext_elem.attrib.get('srsDimension', '2'))
                        ext_pts = ext_elem.text.split()
                        exterior = _polygon_from_coords(
                            ext_pts,
                            fix_geom=True,
                            swap=False,
                            dims=dims
                        )
                        try:
                            interiors = [
                                _polygon_from_coords(
                                    int_pts.text.split(),
                                    fix_geom=True,
                                    swap=False,
                                    dims=dims
                                )
                                for int_pts in feature.findall(interior_str, nsmap)
                            ]
                        except AttributeError:
                            interiors = []
                        geom = Polygon(exterior, interiors).buffer(0)
                        if to_wgs84:
                            project = partial(
                                pyproj.transform,
                                pyproj.Proj(self.srid, preserve_units=False),
                                pyproj.Proj('EPSG:4326', preserve_units=False)
                            )
                            geom = transform(project, geom)
                        yield dict(geometry=geom, attributes=dict(maskType=_type))
        # except StopIteration:
        #     yield dict(
        #         geometry=Polygon(),
        #         attributes=dict(
        #             maskType=None
        #         )
        #     )
        #     raise StopIteration()


class SentinelGranuleCompact(SentinelGranule):
    """This object contains relevant metadata from a granule."""

    def __init__(self, granule, dataset):
        """Prepare data paths depending on if ZIP or not."""
        self.dataset = dataset
        if self.dataset.is_zip:
            granules_path = self.dataset._zip_root
        else:
            granules_path = dataset.path
        self.granule_identifier = granule.attrib["granuleIdentifier"]
        # extract the granule folder name by an IMAGE_FILE name
        image_file_name = granule.find("IMAGE_FILE").text
        image_file_name_arr = image_file_name.split("/")
        self.granule_path = os.path.join(
            granules_path, image_file_name_arr[0], image_file_name_arr[1])
        self.datastrip_identifier = granule.attrib["datastripIdentifier"]

        self.rasts = {}  # dict of arcpy raster objects
        self.profiles = {}

    @cached_property
    def metadata_path(self):
        """Determine the metadata path."""
        metadata_path = os.path.join(self.granule_path, 'MTD_TL.xml').replace('\\', '/')
        try:
            assert os.path.isfile(metadata_path) or \
                   metadata_path in self.dataset._zipfile.namelist()
        except AssertionError:
            raise S2ReaderIOError(
                "Granule metadata XML does not exist:", metadata_path)
        return metadata_path

    # @cached_property
    # def pvi_path(self):
    #     """Determine the PreView Image (PVI) path inside the SAFE pkg."""
    #     return _pvi_path(self)


def _pvi_path(granule):
    """Determine the PreView Image (PVI) path inside the SAFE pkg."""
    pvi_name = next(granule._metadata.iter("PVI_FILENAME")).text
    pvi_name = pvi_name.split("/")
    pvi_path = os.path.join(
        granule.granule_path,
        pvi_name[len(pvi_name) - 2], pvi_name[len(pvi_name) - 1]
    )
    try:
        assert os.path.isfile(pvi_path) or \
               pvi_path in granule.dataset._zipfile.namelist()
    except (AssertionError, AttributeError):
        return None
    return pvi_path


def _granule_identifier_to_xml_name(granule_identifier):
    """
    Very ugly way to convert the granule identifier.
    e.g.
    From
    Granule Identifier:
    S2A_OPER_MSI_L1C_TL_SGS__20150817T131818_A000792_T28QBG_N01.03
    To
    Granule Metadata XML name:
    S2A_OPER_MTD_L1C_TL_SGS__20150817T131818_A000792_T28QBG.xml
    """
    # Replace "MSI" with "MTD".
    changed_item_type = re.sub("_MSI_", "_MTD_", granule_identifier)
    # Split string up by underscores.
    split_by_underscores = changed_item_type.split("_")
    del split_by_underscores[-1]
    cleaned = str()
    # Stitch string list together, adding the previously removed underscores.
    for i in split_by_underscores:
        cleaned += (i + "_")
    # Remove last underscore and append XML file extension.
    out_xml = cleaned[:-1] + ".xml"

    return out_xml


def _polygon_from_coords(coords, fix_geom=False, swap=True, dims=2):
    """
    Return Shapely Polygon from coordinates.
    - coords: list of alterating latitude / longitude coordinates
    - fix_geom: automatically fix geometry
    """
    assert len(coords) % dims == 0
    number_of_points = len(coords) // dims
    coords_as_array = np.array(coords)
    reshaped = coords_as_array.reshape(number_of_points, dims)
    points = [
        (float(i[1]), float(i[0])) if swap else (float(i[0]), float(i[1]))
        for i in reshaped.tolist()
    ]
    polygon = Polygon(points).buffer(0)
    try:
        assert polygon.is_valid
        return polygon
    except AssertionError:
        if fix_geom:
            return polygon.buffer(0)
        else:
            raise RuntimeError("Geometry is not valid.")


####################
# a python implementation of MATLAB 'imresize' function
####################
def triangle(x):
    x = np.array(x).astype(np.float64)
    lessthanzero = np.logical_and((x >= -1), x < 0)
    greaterthanzero = np.logical_and((x <= 1), x >= 0)
    f = np.multiply((x + 1), lessthanzero) + np.multiply((1 - x), greaterthanzero)
    return f


def cubic(x):
    x = np.array(x).astype(np.float64)
    absx = np.absolute(x)
    absx2 = np.multiply(absx, absx)
    absx3 = np.multiply(absx2, absx)
    f = np.multiply(1.5 * absx3 - 2.5 * absx2 + 1, absx <= 1) + np.multiply(-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2,
                                                                            (1 < absx) & (absx <= 2))
    return f


def _contributions(in_length, out_length, scale, kernel, k_width):
    if scale < 1:
        h = lambda x: scale * kernel(scale * x)
        kernel_width = 1.0 * k_width / scale
    else:
        h = kernel
        kernel_width = k_width
    x = np.arange(1, out_length + 1).astype('f4')
    u = x / scale + 0.5 * (1 - 1 / scale)
    left = np.floor(u - kernel_width / 2)
    P = int(math.ceil(kernel_width)) + 2
    ind = np.expand_dims(left, axis=1) + np.arange(P) - 1  # -1 because indexing from 0
    indices = ind.astype(np.int32)
    weights = h(np.expand_dims(u, axis=1) - indices - 1)  # -1 because indexing from 0
    weights = np.divide(weights, np.expand_dims(np.sum(weights, axis=1), axis=1))
    aux = np.concatenate((np.arange(in_length), np.arange(in_length - 1, -1, step=-1))).astype(np.int32)
    indices = aux[np.mod(indices, aux.size)]
    ind2store = np.nonzero(np.any(weights, axis=0))[0]
    weights = weights[:, ind2store]
    indices = indices[:, ind2store]
    return weights, indices


def imresize_mex(inimg, weights, indices, dim):
    in_shape = inimg.shape
    w_shape = weights.shape
    out_shape = list(in_shape)
    out_shape[dim] = w_shape[0]
    outimg = np.zeros(out_shape)
    if dim == 0:
        for i_img in range(in_shape[1]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[ind, i_img].astype('f4')
                outimg[i_w, i_img] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    elif dim == 1:
        for i_img in range(in_shape[0]):
            for i_w in range(w_shape[0]):
                w = weights[i_w, :]
                ind = indices[i_w, :]
                im_slice = inimg[i_img, ind].astype('f4')
                outimg[i_img, i_w] = np.sum(np.multiply(np.squeeze(im_slice, axis=0), w.T), axis=0)
    if inimg.dtype == np.uint8:
        outimg = np.clip(outimg, 0, 255)
        return np.around(outimg).astype(np.uint8)
    else:
        return outimg


def imresize_vec(in_img, weights, indices, dim):
    if dim == 0:
        weights = weights[..., np.newaxis]
        out_img = np.sum(weights * ((in_img[..., indices, :]).astype(np.float64)), axis=-2)
    elif dim == 1:
        out_img = np.sum(weights * (in_img[..., indices].astype(np.float64)), axis=-1)
    else:
        raise NotImplementedError

    return out_img


def resize_along_dim(img, dim, weights, indices, mode="vec"):
    if mode == "org":
        out = imresize_mex(img, weights, indices, dim)
    else:
        out = imresize_vec(img, weights, indices, dim)
    return out.astype('f4')


def imresize(input_img, scale_factor=None, size=None, mode='bicubic', resize_mode="vec"):
    if mode == 'bicubic':
        kernel = cubic
    elif mode == 'bilinear':
        kernel = triangle
    else:
        raise NotImplementedError
    # is_tensor = False
    # if isinstance(input_img, torch.Tensor):
    #     input_img = input_img.detach().cpu().numpy()
    #     is_tensor = True
    is_transpose = False
    if input_img.ndim == 3 and (input_img.shape[-1] == 1 or input_img.shape[-1] == 3):
        input_img = np.transpose(input_img, (2, 0, 1))
        is_transpose = True
    if input_img.ndim == 4 and (input_img.shape[-1] == 1 or input_img.shape[-1] == 3):
        raise NotImplementedError
    is_uint8 = False
    if input_img.dtype == np.uint8:
        is_uint8 = True

    kernel_width = 4.0
    # Fill scale and output_size
    in_h, in_w = input_img.shape[-2], input_img.shape[-1]
    input_size = (in_h, in_w)
    if scale_factor is not None:
        scale_factor = float(scale_factor)
        scale = [scale_factor, scale_factor]
        out_h = int(math.ceil((in_h * scale_factor)))
        out_w = int(math.ceil((in_w * scale_factor)))
        size = (out_h, out_w)
    elif size is not None:
        scale_h = 1.0 * size[0] / in_h
        scale_w = 1.0 * size[1] / in_w
        scale = [scale_h, scale_w]
    else:
        raise NotImplementedError('scalar_scale OR output_shape should be defined!')
    order = np.argsort(scale)
    weights = []
    indices = []
    for k in range(2):
        w, ind = _contributions(input_size[k], size[k], scale[k], kernel, kernel_width)
        weights.append(w)
        indices.append(ind)

    flag_2d = False
    if input_img.ndim == 2:
        input_img = input_img[np.newaxis, :]
        flag_2d = True
    for cur_dim in order:
        input_img = resize_along_dim(input_img, cur_dim, weights[cur_dim], indices[cur_dim], resize_mode)
    if is_transpose:
        input_img = np.transpose(input_img, (1, 2, 0))
    if flag_2d:
        input_img = np.squeeze(input_img)
    if is_uint8:
        input_img = np.around(np.clip(input_img, 0, 255)).astype('u1')
    # if is_tensor:
    #     input_img = torch.from_numpy(input_img)
    return input_img


class Extractor(object):
    """
    a sub-imgs extractor for sentinel2 MSI and landsat8 OLI with overlapped region
    """

    def __init__(self, l8_path, s2_path):
        self.l8_path = l8_path
        self.s2_path = s2_path
        self.rasts = {}
        self.profiles = {}

    def read_data(self, sat_type, bands):
        if sat_type == 'l8':
            l8_scene = LandsatScene(self.l8_path)
            for band in bands:
                tag = '{}_B{}'.format(sat_type, band)
                if tag in self.rasts and tag in self.profiles:
                    continue
                else:
                    self.rasts[tag] = l8_scene.get_TOA(band, per_pixel=False, clip=True)
                    self.profiles[tag] = l8_scene.profiles['B{}'.format(band)]
                    self.l8_cloud = l8_scene.get_mask('cloud')[0]
        if sat_type == 's2':
            with s2_open(self.s2_path) as s2_scene:
                for band in bands:
                    tag = '{}_B{}'.format(sat_type, band)
                    if tag in self.rasts and tag in self.profiles:
                        continue
                    else:
                        self.rasts[tag] = s2_scene.granules[0].get_TOA(band, clip=True)
                        self.profiles[tag] = s2_scene.granules[0].profiles['B{}'.format(band)]

    def get_crs(self, sat_type, bands):
        crs = self.profiles['{}_B{}'.format(sat_type, bands[0])]['crs']
        for band in bands[1:]:
            assert crs == self.profiles['{}_B{}'.format(sat_type, band)]['crs']
        return crs

    def get_bound(self, sat_type, bands):
        bound = self.profiles['{}_B{}'.format(sat_type, bands[0])]['bounds']
        for band in bands[1:]:
            assert bound == self.profiles['{}_B{}'.format(sat_type, band)]['bounds']
        return bound

    def get_res(self, sat_type, bands):
        res = self.profiles['{}_B{}'.format(sat_type, bands[0])]['res']
        for band in bands[1:]:
            assert res == self.profiles['{}_B{}'.format(sat_type, band)]['res']
        return res

    def get_transform(self, sat_type, bands):
        t = self.profiles['{}_B{}'.format(sat_type, bands[0])]['transform']
        for band in bands[1:]:
            assert t == self.profiles['{}_B{}'.format(sat_type, band)]['transform']
        return t

    def cal_translation(self, l8_data, s2_data, num_sample=5, crop_sz=480, step=240, scale=(3, 3)):
        assert crop_sz % scale[0] == 0 and crop_sz % scale[0] == 0
        c, h, w = s2_data.shape
        if h < crop_sz or w < crop_sz:
            return None
        h_space = np.arange(0, h - crop_sz + 1, step)
        w_space = np.arange(0, w - crop_sz + 1, step)

        l8_mask = np.any(l8_data == 0., axis=0)
        s2_mask = np.any(s2_data == 0., axis=0)
        res_list = []
        for x in h_space:
            for y in w_space:
                crop_s2_mask = s2_mask[x:x + crop_sz, y:y + crop_sz]
                if np.any(crop_s2_mask):
                    continue
                crop_l8_mask = l8_mask[x // scale[1]:x // scale[1] + crop_sz // scale[1],
                               y // scale[0]:y // scale[0] + crop_sz // scale[0]]
                if np.any(crop_l8_mask):
                    continue
                crop_s2_data = s2_data[:, x:x + crop_sz, y:y + crop_sz]
                crop_l8_data = l8_data[:, x // scale[1]:x // scale[1] + crop_sz // scale[1],
                               y // scale[0]:y // scale[0] + crop_sz // scale[0]]

                up_crop_l8_data = imresize(crop_l8_data, size=(crop_sz, crop_sz), mode='bicubic')

                res_list.append(self._translation(crop_s2_data, up_crop_l8_data)[1])
                if len(res_list) == num_sample:
                    return np.percentile(np.stack(res_list), 50, axis=0)
        if len(res_list) > 1:
            return np.percentile(np.stack(res_list), 50, axis=0)
        else:
            return None

    def _translation(self, arr1, arr2):
        # arr1 is the templateImage
        # TODO, the arr1 and arr2 are regarded as RGB arrays with the first axis equal to 3 by default.
        im1_gray = cv2.cvtColor(np.transpose(arr1, (1, 2, 0)), cv2.COLOR_RGB2GRAY)
        im2_gray = cv2.cvtColor(np.transpose(arr2, (1, 2, 0)), cv2.COLOR_RGB2GRAY)

        # Define the motion model
        warp_mode = cv2.MOTION_TRANSLATION
        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        # Specify the number of iterations.
        number_of_iterations = 5000
        # Specify the threshold of the increment
        # in the correlation coefficient between two iterations
        termination_eps = 1e-10
        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)
        # Run the ECC algorithm. The results are stored in warp_matrix.
        mask = (im2_gray != 0).astype('u1')
        cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, mask, 5)
        return cc, warp_matrix[:, 2]

    def reproject(self,
                  sat_type,
                  bands,
                  src_transform,
                  src_crs,
                  dst_transform,
                  dst_crs,
                  dst_height,
                  dst_width,
                  src_nodata=0,
                  dst_nodata=0,
                  resampling=2
                  ):
        src_data = np.stack([self.rasts['{}_B{}'.format(sat_type, band)] for band in bands])
        dst_data, _ = warp.reproject(src_data,
                                     np.empty((len(bands), dst_height, dst_width), dtype=src_data.dtype),
                                     src_transform=src_transform,
                                     src_crs=src_crs,
                                     src_nodata=src_nodata,
                                     dst_transform=dst_transform,
                                     dst_crs=dst_crs,
                                     dst_nodata=dst_nodata,
                                     resampling=resampling
                                     )
        return dst_data

    def write2tiff(self, src_data, crs, t, save_path, is_rgb=False):
        n_channels = len(src_data.shape)
        if n_channels == 2:
            src_data = np.expand_dims(src_data, axis=0)
        profile = {
            'driver': 'GTiff',
            'width': src_data.shape[2],
            'height': src_data.shape[1],
            'count': src_data.shape[0],
            'dtype': src_data.dtype,
            'crs': crs,
            'transform': t,
            'blockxsize': 512,
            'blockysize': 512,
            'nodata': 0,
            'tiled': True,
            'compress': 'deflate',
            'interleave': 'band'
        }
        if is_rgb:
            profile['photometric'] = 'RGB'
        with rasterio.open(save_path, 'w', **profile) as dst:
            dst.write(src_data)

    def xy(self, t, row, col, offset="center"):
        """Returns the coordinates ``(x, y)`` of a pixel at `row` and `col`.
        The pixel's center is returned by default, but a corner can be returned
        by setting `offset` to one of `ul, ur, ll, lr`.

        Parameters
        ----------
        t : Affine
            transform.
        row : int
            Pixel row.
        col : int
            Pixel column.
        offset : str, optional
            Determines if the returned coordinates are for the center of the
            pixel or for a corner.

        Returns
        -------
        tuple
            ``(x, y)``
        """
        return xy(t, row, col, offset=offset)

    def index(self, t, x, y, op=math.floor, precision=None):
        """
        Returns the (row, col) index of the pixel containing (x, y) given a
        coordinate reference system.

        Use an epsilon, magnitude determined by the precision parameter
        and sign determined by the op function:
            positive for floor, negative for ceil.

        Parameters
        ----------
        t : Affine
            transform
        x : float
            x value in coordinate reference system
        y : float
            y value in coordinate reference system
        op : function, optional (default: math.floor)
            Function to convert fractional pixels to whole numbers (floor,
            ceiling, round)
        precision : int, optional (default: None)
            Decimal places of precision in indexing, as in `round()`.

        Returns
        -------
        tuple
            (row index, col index)
        """
        return rowcol(t, x, y, op=op, precision=precision)

    def patch_generator(self, bands=(4, 3, 2), crop_sz=(480, 480), step=(240, 240), thres_sz=(48, 48),
                        filter_nodata=True, filter_cloud=True, align=True):
        if not isinstance(bands, (list, tuple)):
            bands = [bands]
        if len(bands) == 2 and isinstance(bands[0], (list, tuple)):
            l8_bands = bands[0]
            s2_bands = bands[1]
        else:
            l8_bands = bands
            s2_bands = bands
        assert len(l8_bands) == len(s2_bands)

        self.read_data('l8', l8_bands)
        self.read_data('s2', s2_bands)

        l8_crs = self.get_crs('l8', l8_bands)
        s2_crs = self.get_crs('s2', s2_bands)
        l8_geom = box(*self.get_bound('l8', l8_bands))
        s2_geom = box(*self.get_bound('s2', s2_bands))
        l8_res = self.get_res('l8', l8_bands)
        s2_res = self.get_res('s2', s2_bands)
        l8_transform = self.get_transform('l8', l8_bands)
        s2_transform = self.get_transform('s2', s2_bands)
        scale_factor = [i / j for i, j in zip(l8_res, s2_res)]
        if not l8_crs.to_epsg() == s2_crs.to_epsg():
            l8_geom = shape(warp.transform_geom(l8_crs, s2_crs, l8_geom.__geo_interface__))
        common_geom = l8_geom.intersection(s2_geom).envelope
        assert common_geom.geom_type == 'Polygon'

        upper_left_index = self.index(s2_transform, common_geom.bounds[0], common_geom.bounds[3], math.ceil)
        upper_left_xy = self.xy(s2_transform, *upper_left_index, offset='ul')
        assert common_geom.intersects(Point(upper_left_xy))
        lower_right_index = self.index(s2_transform, common_geom.bounds[2], common_geom.bounds[1], math.floor)
        lower_right_xy = self.xy(s2_transform, *lower_right_index, offset='ul')
        assert common_geom.intersects(Point(lower_right_xy))

        s2_dst_height = int((lower_right_index[0] - upper_left_index[0]) // scale_factor[1] * scale_factor[1])
        s2_dst_width = int((lower_right_index[1] - upper_left_index[1]) // scale_factor[0] * scale_factor[0])
        l8_dst_height = int(s2_dst_height // scale_factor[1])
        l8_dst_width = int((s2_dst_width // scale_factor[0]))

        l8_left_x = upper_left_xy[0] + 0. * scale_factor[0] * s2_res[0]  # 0.5 is the center point, 0 is upper left
        l8_upper_y = upper_left_xy[1] - 0. * scale_factor[1] * s2_res[1]
        l8_dst_transform = rasterio.transform.from_origin(l8_left_x, l8_upper_y, *l8_res)
        s2_dst_transform = rasterio.transform.from_origin(*upper_left_xy, *s2_res)

        # sentinel2 data can be reprojected or just indexed from the original raster,
        # the results are identical.
        s2_dst_data = self.reproject('s2', s2_bands, s2_transform, s2_crs,
                                     s2_dst_transform, s2_crs, s2_dst_height, s2_dst_width)

        # s2_dst_data = np.stack([self.rasts['{}_B{}'.format('s2', band)] for band in s2_bands])
        # s2_dst_data = s2_dst_data[:, upper_left_index[0]: upper_left_index[0] + s2_dst_height,
        #               upper_left_index[1]: upper_left_index[1] + s2_dst_width]

        l8_dst_data = self.reproject('l8', l8_bands, l8_transform, l8_crs,
                                     l8_dst_transform, s2_crs, l8_dst_height, l8_dst_width)
        if filter_cloud:
            l8_cloud, _ = warp.reproject(self.l8_cloud,
                                         np.empty((l8_dst_height, l8_dst_width),
                                                  dtype=self.l8_cloud.dtype),
                                         src_transform=l8_transform,
                                         src_crs=l8_crs,
                                         dst_transform=l8_dst_transform,
                                         dst_crs=s2_crs
                                         )
        crop_sz = crop_sz if crop_sz else (s2_dst_height, s2_dst_width)
        step = step if step else (s2_dst_height, s2_dst_width)
        thres_sz = thres_sz if thres_sz else (s2_dst_height, s2_dst_width)
        assert crop_sz[0] % scale_factor[0] == 0, 'crop size is not {:d}X multiplication.'.format(scale_factor[0])
        assert crop_sz[1] % scale_factor[1] == 0, 'crop size is not {:d}X multiplication.'.format(scale_factor[1])
        assert step[0] % scale_factor[0] == 0, 'step is not {:d}X multiplication.'.format(scale_factor[0])
        assert step[1] % scale_factor[1] == 0, 'step is not {:d}X multiplication.'.format(scale_factor[1])
        assert thres_sz[0] % scale_factor[0] == 0, 'thres_sz is not {:d}X multiplication.'.format(scale_factor[0])
        assert thres_sz[1] % scale_factor[1] == 0, 'thres_sz is not {:d}X multiplication.'.format(scale_factor[1])

        h_space = np.arange(0, s2_dst_height - crop_sz[0] + 1, step[0])
        if s2_dst_height - (h_space[-1] + crop_sz[0]) > thres_sz[0]:
            h_space = np.append(h_space, s2_dst_height - crop_sz[0])
        w_space = np.arange(0, s2_dst_width - crop_sz[1] + 1, step[1])
        if s2_dst_width - (w_space[-1] + crop_sz[1]) > thres_sz[1]:
            w_space = np.append(w_space, s2_dst_width - crop_sz[1])

        for x in h_space:
            for y in w_space:
                crop_hr_data = s2_dst_data[:, x:x + crop_sz[0], y:y + crop_sz[1]]
                crop_lr_data = l8_dst_data[:,
                               int(x // scale_factor[0]):int(x // scale_factor[0] + crop_sz[0] // scale_factor[0]),
                               int(y // scale_factor[1]):int(y // scale_factor[1] + crop_sz[1] // scale_factor[1])]
                if filter_nodata and (np.any(crop_hr_data == 0) or np.any(crop_lr_data == 0)):
                    continue
                if filter_cloud:
                    crop_l8_cloud = l8_cloud[
                                    int(x // scale_factor[0]):int(x // scale_factor[0] + crop_sz[0] // scale_factor[0]),
                                    int(y // scale_factor[1]):int(y // scale_factor[1] + crop_sz[1] // scale_factor[1])]
                    if np.sum(crop_l8_cloud) / crop_l8_cloud.size > 0.1:
                        continue
                up_crop_lr_data = imresize(crop_lr_data, size=crop_sz, mode='bicubic')

                hr_upper_left_xy = self.xy(s2_dst_transform, x, y, 'ul')
                lr_upper_left_xy = list(self.xy(l8_dst_transform, x // scale_factor[0], y // scale_factor[1], 'ul'))

                if align:
                    try:
                        # sub-pixel alignment
                        corr, t_xy = self._translation(crop_hr_data[:, :1000, :1000], up_crop_lr_data[:, :1000, :1000])
                        if corr < 0.7 or np.abs(t_xy).max() > 3:
                            continue
                    except cv2.error:
                        continue
                    lr_upper_left_xy[0] += t_xy[0] * s2_res[0]
                    lr_upper_left_xy[1] -= t_xy[1] * s2_res[1]

                hr_transform = rasterio.transform.from_origin(*hr_upper_left_xy, *s2_res)
                lr_transform = rasterio.transform.from_origin(*lr_upper_left_xy, *l8_res)
                crop_lr_data = self.reproject('l8', l8_bands, l8_transform, l8_crs,
                                              lr_transform, s2_crs, int(crop_sz[0] // scale_factor[0]),
                                              int(crop_sz[1] // scale_factor[1]))
                crop_hr_data = np.ascontiguousarray(crop_hr_data)
                crop_lr_data = np.ascontiguousarray(crop_lr_data)

                yield {
                    'hr': crop_hr_data,
                    'lr': crop_lr_data,
                    'hr_t': hr_transform,
                    'lr_t': lr_transform,
                    'crs': s2_crs,
                    'corr': corr if align else None
                }

    def __call__(self, bands=(4, 3, 2), save_l8_dir=None, save_s2_dir=None,
                 l8_save_name=None, s2_save_name=None):
        gen = self.patch_generator(bands, crop_sz=None, step=None, thres_sz=None, filter_nodata=False, align=False)
        index = 0
        for patch in gen:
            index += 1
            lr_data = patch['lr']
            hr_data = patch['hr']
            hr_t = patch['hr_t']
            lr_t = patch['lr_t']
            crs = patch['crs']
            if save_l8_dir:
                if l8_save_name is None:
                    l8_save_name = os.path.basename(self.l8_path).replace('_MTL.txt', '_N{:04d}.TIF'.format(index))
                l8_save_path = os.path.join(save_l8_dir, l8_save_name)
                self.write2tiff(lr_data, crs, lr_t, l8_save_path, is_rgb=(len(lr_data) == 3))
            if save_s2_dir:
                if s2_save_name is None:
                    s2_save_name = os.path.basename(self.s2_path)[:60] + '_N{:04d}.TIF'.format(index)
                s2_save_path = os.path.join(save_s2_dir, s2_save_name)
                self.write2tiff(hr_data, crs, hr_t, s2_save_path, is_rgb=(len(hr_data) == 3))