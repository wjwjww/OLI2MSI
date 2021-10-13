"""A multi-thread tool to crop large images to sub-images for faster IO."""
import os
import os.path as osp
import sys
from multiprocessing import Pool
import numpy as np
import cv2
import rasterio
from rasterio.transform import xy, rowcol
from PIL import Image
from utils import ProgressBar  # noqa: E402
from utils import LandsatScene
from utils import s2_open
from utils import imresize
from utils import Extractor
from utils import get_paths_from_images, get_paths_from_l8_dir, get_paths_from_s2_dir, tif_to_png


def main():
    mode = 'sat_pair'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = dict()
    opt['n_thread'] = 8

    # GeoTiff file properties
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'tiled': True,
        'compress': 'deflate',
        'interleave': 'band',
        'blockxsize': 512,
        'blockysize': 512,
    }
    opt['profile'] = profile
    if mode == 'single':
        opt['satellite'] = 'sar'
        opt['channels'] = (1, )
        opt['input_folder'] = "./TRAIN_DATA/"
        opt['save_folder'] = "./SUB_IMGs_TRAIN/"
        opt['crop_sz'] = 480  # the size of each sub-image
        opt['step'] = 240  # step of the sliding crop window
        opt['thres_sz'] = 48  # size threshold
        if opt['satellite'] == 'landsat8':
            extract_landsat8_single(opt)
        elif opt['satellite'] == 'sentinel2':
            extract_sentinel2_single(opt)
        else:
            raise NotImplementedError

    elif mode == 'tif_pair':
        GT_folder = ''
        LR_folder = ''
        save_GT_folder = ''
        save_LR_folder = ''
        scale_ratio = 3
        crop_sz = 480  # the size of each sub-image (GT)
        step = 240  # step of the sliding crop window (GT)
        thres_sz = 48  # size threshold
        ########################################################################
        # check that all the GT and LR images have correct scale ratio
        img_GT_list = get_paths_from_images(GT_folder)
        img_LR_list = get_paths_from_images(LR_folder)
        assert len(img_GT_list) == len(img_LR_list), 'different length of GT_folder and LR_folder.'
        for path_GT, path_LR in zip(img_GT_list, img_LR_list):
            with rasterio.open(path_GT, 'r') as src:
                img_GT = src.read()
            with rasterio.open(path_LR, 'r') as src:
                img_LR = src.read()
            _, w_GT, h_GT = img_GT.size
            _, w_LR, h_LR = img_LR.size
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
            assert w_GT / w_LR == scale_ratio, 'GT width [{:d}] is not {:d}X as LR weight [{:d}] for {:s}.'.format(  # noqa: E501
                w_GT, scale_ratio, w_LR, path_GT)
        # check crop size, step and threshold size
        assert crop_sz % scale_ratio == 0, 'crop size is not {:d}X multiplication.'.format(
            scale_ratio)
        assert step % scale_ratio == 0, 'step is not {:d}X multiplication.'.format(scale_ratio)
        assert thres_sz % scale_ratio == 0, 'thres_sz is not {:d}X multiplication.'.format(
            scale_ratio)
        print('processing ...')
        opt['src_hr_folder'] = GT_folder
        opt['src_lr_folder'] = LR_folder
        opt['dst_hr_folder'] = save_GT_folder
        opt['dst_lr_folder'] = save_LR_folder
        opt['crop_sz'] = crop_sz
        opt['step'] = step
        opt['scale'] = scale_ratio
        opt['thres_sz'] = thres_sz
        extract_tif_pair_data(opt)
        assert len(get_paths_from_images(save_GT_folder)) == len(
            get_paths_from_images(
                save_LR_folder)), 'different length of save_GT_folder and save_LR_folder.'
    elif mode == 'sat_pair':

        # OLI2MSI dataset
        opt['src_hr_folder'] = "/home/ubuntu/data5/WangJW/datasets/paired_SR_satellite_dataset/sentinel2/"
        opt['src_lr_folder'] = "/home/ubuntu/data5/WangJW/datasets/paired_SR_satellite_dataset/landsat8/"
        opt['dst_hr_folder'] = "/home/ubuntu/data5/WangJW/datasets/paired_SR_satellite_dataset/pair_match_hr/"
        opt['dst_lr_folder'] = "/home/ubuntu/data5/WangJW/datasets/paired_SR_satellite_dataset/pair_match_lr/"

        opt['crop_sz'] = 480
        opt['step'] = 480
        opt['thres_sz'] = 48
        opt['bands'] = (4, 3, 2)
        extract_sat_pair_data(opt)
    else:
        raise ValueError('Wrong mode.')


def extract_landsat8_single(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    img_list = get_paths_from_l8_dir(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(landsat8_worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def extract_sentinel2_single(opt):
    input_folder = opt['input_folder']
    save_folder = opt['save_folder']
    if not osp.exists(save_folder):
        os.makedirs(save_folder)
        print('mkdir [{:s}] ...'.format(save_folder))
    else:
        print('Folder [{:s}] already exists. Exit...'.format(save_folder))
        sys.exit(1)
    img_list = get_paths_from_s2_dir(input_folder)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(img_list))

    pool = Pool(opt['n_thread'])
    for path in img_list:
        pool.apply_async(landsat8_worker, args=(path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def extract_tif_pair_data(opt):
    src_hr_dir = opt['src_hr_folder']
    src_lr_dir = opt['src_lr_folder']
    dst_hr_dir = opt['dst_hr_folder']
    dst_lr_dir = opt['dst_lr_folder']
    for save_folder in [dst_hr_dir, dst_lr_dir]:
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
            print('mkdir [{:s}] ...'.format(save_folder))
        else:
            print('Folder [{:s}] already exists. Exit...'.format(save_folder))
            sys.exit(1)
    hr_img_list = get_paths_from_l8_dir(src_hr_dir)
    lr_img_list = get_paths_from_images(src_lr_dir)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(len(hr_img_list))

    pool = Pool(opt['n_thread'])
    for hr_path, lr_path in zip(hr_img_list, lr_img_list):
        pool.apply_async(tif_pair_worker, args=(hr_path, lr_path, opt), callback=update)
    pool.close()
    pool.join()
    print('All subprocesses done.')


def extract_sat_pair_data(opt):
    src_hr_dir = opt['src_hr_folder']
    src_lr_dir = opt['src_lr_folder']
    dst_hr_dir = opt['dst_hr_folder']
    dst_lr_dir = opt['dst_lr_folder']

    hr_paths = get_paths_from_s2_dir(src_hr_dir)
    lr_paths = get_paths_from_l8_dir(src_lr_dir)

    for tem_path in [dst_hr_dir, dst_lr_dir]:
        if not os.path.exists(tem_path):
            os.makedirs(tem_path)
            print('mkdir [{:s}] ...'.format(tem_path))
        else:
            print('Folder [{:s}] already exists. please check it'.format(tem_path))
            sys.exit(1)

    def update(arg):
        pbar.update(arg)

    pbar = ProgressBar(int(len(hr_paths) * len(lr_paths)))
    pool = Pool(opt['n_thread'])
    results = []
    for hr_path in hr_paths:
        for lr_path in lr_paths:
            res = pool.apply_async(sat_pair_worker, args=(hr_path, lr_path, opt), callback=update)
            results.append(res)
    pool.close()
    pool.join()

    for i in range(len(results)):
        hr_path = hr_paths[int(i % len(hr_paths))]
        lr_path = lr_paths[int(i // len(hr_paths))]
        if not results[i].successful():
            print('[**]' + hr_path + '-' + lr_path + ' raised an error')


def tif_pair_worker(hr_path, lr_path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    scale = opt['scale']

    assert crop_sz % scale == 0, 'crop size is not {:d}X multiplication.'.format(scale)
    assert step % scale == 0, 'step is not {:d}X multiplication.'.format(scale)
    assert thres_sz % scale == 0, 'thres_sz is not {:d}X multiplication.'.format(scale)

    hr_src  = rasterio.open(hr_path, 'r')
    hr_data = hr_src.read()
    lr_src = rasterio.open(lr_path, 'r')
    lr_data = lr_src.read()
    hr_mask = np.any(hr_data == 0, axis=0)
    lr_mask = np.any(lr_data == 0, axis=0)

    hr_c, hr_h, hr_w = hr_data.shape
    lr_c, lr_h, lr_w = lr_data.shape
    assert hr_h / lr_h == scale, 'GT height [{:d}] is not {:d}X as LR height [{:d}] for {:s}.'.format(  # noqa: E501
        hr_h, scale, lr_h, hr_path)
    assert hr_w / lr_w == scale, 'GT width [{:d}] is not {:d}X as LR width [{:d}] for {:s}.'.format(  # noqa: E501
        hr_w, scale, lr_w, hr_path)
    assert hr_c == lr_c, 'GT number of channels [{:d}] is not equal to that of LR [{:d}] for {:s}.'.format(
        hr_c, lr_c, hr_path)

    h_space = np.arange(0, hr_h - crop_sz + 1, step)
    if hr_h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, hr_h - crop_sz)
    w_space = np.arange(0, hr_w - crop_sz + 1, step)
    if hr_w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, hr_w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_hr_mask = hr_mask[x:x + crop_sz, y:y + crop_sz]
            crop_lr_mask = lr_mask[x//scale:x//scale + crop_sz//scale, y//scale:y//scale + crop_sz//scale]
            if np.any(crop_hr_mask) or np.any(crop_lr_mask):
                continue
            crop_hr_data = hr_data[:, x:x + crop_sz, y:y + crop_sz]
            crop_lr_data = lr_data[:, x//scale:x//scale + crop_sz//scale, y//scale:y//scale + crop_sz//scale]
            crop_hr_data = np.ascontiguousarray(crop_hr_data)
            crop_lr_data = np.ascontiguousarray(crop_lr_data)

            patch_hr_path = os.path.join(opt['dst_hr_folder'],
                                         os.path.basename(hr_path).replace('.TIF', '_N{:04d}.TIF'.format(index)))
            patch_lr_path = os.path.join(opt['dst_lr_folder'],
                                         os.path.basename(lr_path).replace('.TIF', '_N{:04d}.TIF'.format(index)))
            # save HR crop image
            t = rasterio.transform.from_origin(*hr_src.xy(x, y, 'ul'), *hr_src.res)
            profile = {
                'width': crop_hr_data.shape[2],
                'height': crop_hr_data.shape[1],
                'count': crop_hr_data.shape[0],
                'crs': hr_src.crs,
                'transform': t,
                'photometric': "RGB"
            }
            if hr_c == 3:
                profile['photometric'] = 'RGB'
            profile.update(opt['profile'])
            with rasterio.open(patch_hr_path, 'w', **profile) as dst:
                dst.write(crop_hr_data)

            # save LR crop image
            t = rasterio.transform.from_origin(*lr_src.xy(x//scale, y//scale, 'ul'), *lr_src.res)
            profile = {
                'width': crop_lr_data.shape[2],
                'height': crop_lr_data.shape[1],
                'count': crop_lr_data.shape[0],
                'crs': lr_src.crs,
                'transform': t,
                'photometric': "RGB"
            }
            if lr_c == 3:
                profile['photometric'] = 'RGB'
            profile.update(opt['profile'])
            with rasterio.open(patch_lr_path, 'w', **profile) as dst:
                dst.write(crop_lr_data)
            hr_src.close()
            lr_src.close()

            dst_data = imresize(crop_lr_data, scale_factor=3, mode='bicubic')
            dst_profile = profile.copy()
            dst_profile['height'], dst_profile['width'] = dst_data.shape[-2], dst_data.shape[-1]
            dst_profile['transform'] = dst_profile['transform'] * dst_profile['transform'].scale(1./3., 1./3.)
            dst_profile['photometric'] = 'RGB'
            with rasterio.open(patch_hr_path.replace('.TIF', '_x3.TIF'), 'w', **dst_profile) as dst:
                dst.write(dst_data)
    return 'Processing {:s} ...'.format(os.path.basename(hr_path))


def sat_pair_worker(hr_path, lr_path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']
    bands = opt['bands']

    extractor = Extractor(lr_path, hr_path)
    gen = extractor.patch_generator(bands=bands, crop_sz=(crop_sz, crop_sz), step=(step, step),
                                    thres_sz=(thres_sz, thres_sz))

    s2_part = '_'.join([os.path.basename(hr_path)[:3],
                        os.path.basename(hr_path)[11:19],
                        os.path.basename(hr_path)[38:44]])
    l8_part = '_'.join(['L8', os.path.basename(lr_path)[10:25]])
    file_name = l8_part + '_' + s2_part
    index = 0
    try:
        for patch in gen:
            index += 1
            patch_hr_path = os.path.join(opt['dst_hr_folder'], file_name + '_N{:04d}.TIF'.format(index))
            patch_lr_path = os.path.join(opt['dst_lr_folder'], file_name + '_N{:04d}.TIF'.format(index))
            # save HR crop image
            t = patch['hr_t']
            profile = {
                'width': patch['hr'].shape[2],
                'height': patch['hr'].shape[1],
                'count': patch['hr'].shape[0],
                'crs': patch['crs'],
                'transform': t
            }
            if patch['hr'].shape[0] == 3:
                profile['photometric'] = 'RGB'
            profile.update(opt['profile'])
            with rasterio.open(patch_hr_path, 'w', **profile) as dst:
                dst.write(patch['hr'])

            # save LR crop image
            t = patch['lr_t']
            profile = {
                'width': patch['lr'].shape[2],
                'height': patch['lr'].shape[1],
                'count': patch['lr'].shape[0],
                'crs': patch['crs'],
                'transform': t
            }
            if patch['lr'].shape[0] == 3:
                profile['photometric'] = 'RGB'
            profile.update(opt['profile'])
            with rasterio.open(patch_lr_path, 'w', **profile) as dst:
                dst.write(patch['lr'])

            # # save up-sampled LR crop image
            # dst_data = imresize(patch['lr'], scale_factor=3, mode='bicubic')
            # x_t, y_t = _translation(patch['hr'], dst_data)
            # dst_profile = profile.copy()
            # dst_profile['height'], dst_profile['width'] = dst_data.shape[-2], dst_data.shape[-1]
            # dst_profile['transform'] = dst_profile['transform'] * dst_profile['transform'].scale(1./3., 1./3.)
            # dst_profile['photometric'] = 'RGB'
            # with rasterio.open(patch_hr_path.replace('.TIF', '_x={:.3f}_y={:.3f}.TIF'.format(x_t, y_t)),
            #                    'w', **dst_profile) as dst:
            #     dst.write(dst_data)
            # # save down&up-sampled GT crop image
            # dst_data = imresize(patch['hr'], scale_factor=1./3., mode='bicubic')
            # dst_data = imresize(dst_data, scale_factor=3, mode='bicubic')
            # x_t, y_t = _translation(patch['hr'], dst_data)
            # with rasterio.open(patch_hr_path.replace('.TIF', '_x={:.3f}_y={:.3f}_updown.TIF'.format(x_t, y_t)),
            #                    'w', **dst_profile) as dst:
            #     dst.write(dst_data)
        return 'Processing {:s} ...'.format(os.path.basename(hr_path))
    except AssertionError:
        return 'mismatched data pair'


def landsat8_worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']

    scene = LandsatScene(path)
    img_name = scene.meta.LANDSAT_PRODUCT_ID
    data = np.stack([scene.get_TOA(band) for band in opt['channels']])
    nodata_mask = np.any(data == 0, axis=0)
    cloud_mask = np.any(np.stack(scene.get_mask('cloud')), axis=0)

    c, h, w = data.shape
    # statistic
    out_string = '[*]'
    for i in range(c):
        tem_data = data[i][~nodata_mask]
        out_string += '|ch{}: mean:{:.3f}, std:{:.3f}'.format(i, np.mean(tem_data), np.std(tem_data))
    print(out_string)
    src_profile = scene.profiles['B{}'.format(opt['channels'][0])]

    h_space = np.arange(0, h - crop_sz + 1, step)
    if h - (h_space[-1] + crop_sz) > thres_sz:
        h_space = np.append(h_space, h - crop_sz)
    w_space = np.arange(0, w - crop_sz + 1, step)
    if w - (w_space[-1] + crop_sz) > thres_sz:
        w_space = np.append(w_space, w - crop_sz)

    index = 0
    for x in h_space:
        for y in w_space:
            index += 1
            crop_nodata_mask = nodata_mask[x:x + crop_sz, y:y + crop_sz]
            if np.any(crop_nodata_mask):
                continue
            crop_data = data[:, x:x + crop_sz, y:y + crop_sz]
            crop_cloud_mask = cloud_mask[x:x + crop_sz, y:y + crop_sz]
            crop_data = np.ascontiguousarray(crop_data)
            cloud_ratio = int(100 * np.sum(crop_cloud_mask) / crop_cloud_mask.size)
            if cloud_ratio > 30:
                continue
            patch_path = os.path.join(opt['save_folder'], '{}_C{:0>3d}_N{:04d}.TIF'.format(img_name, cloud_ratio, index))

            t = rasterio.transform.from_origin(*scene.xy(x, y, opt['channels'][0], 'ul'), *src_profile['res'])
            profile = {
                'width': crop_data.shape[2],
                'height': crop_data.shape[1],
                'count': crop_data.shape[0],
                'crs': src_profile['crs'],
                'transform': t,
                'photometric': "RGB"
            }
            profile.update(opt['profile'])
            with rasterio.open(patch_path, 'w', **profile) as dst:
                dst.write(crop_data)
            # geo_data_util.tif_to_png(patch_path)
    return 'Processing {:s} ...'.format(img_name)


def sentinel2_worker(path, opt):
    crop_sz = opt['crop_sz']
    step = opt['step']
    thres_sz = opt['thres_sz']

    with s2_open(path) as scene:
        img_name = scene.product_ID
        granual = scene.granules[0]
        data = np.stack([granual.get_TOA(band) for band in opt['channels']])
        nodata_mask = np.any(data == 0, axis=0)
        cloud_mask = granual.get_cloud_mask()

        c, h, w = data.shape
        # statistic
        out_string = '[*]'
        for i in range(c):
            tem_data = data[i][~nodata_mask]
            out_string += '|ch{}: mean:{:.3f}, std:{:.3f}'.format(i, np.mean(tem_data), np.std(tem_data))
        print(out_string)
        src_profile = granual.profiles['B{}'.format(opt['channels'][0])]

        h_space = np.arange(0, h - crop_sz + 1, step)
        if h - (h_space[-1] + crop_sz) > thres_sz:
            h_space = np.append(h_space, h - crop_sz)
        w_space = np.arange(0, w - crop_sz + 1, step)
        if w - (w_space[-1] + crop_sz) > thres_sz:
            w_space = np.append(w_space, w - crop_sz)

        index = 0
        for x in h_space:
            for y in w_space:
                index += 1
                crop_nodata_mask = nodata_mask[x:x + crop_sz, y:y + crop_sz]
                if np.any(crop_nodata_mask):
                    continue
                crop_data = data[:, x:x + crop_sz, y:y + crop_sz]
                crop_cloud_mask = cloud_mask[x:x + crop_sz, y:y + crop_sz]
                crop_data = np.ascontiguousarray(crop_data)
                cloud_ratio = int(100 * np.sum(crop_cloud_mask) / crop_cloud_mask.size)
                if cloud_ratio > 10:
                    continue
                patch_path = os.path.join(opt['save_folder'], '{}_C{:0>3d}_N{:04d}.TIF'.format(img_name, cloud_ratio, index))

                t = rasterio.transform.from_origin(*granual.xy(x, y, opt['channels'][0], 'ul'), *src_profile['res'])
                profile = {
                    'width': crop_data.shape[2],
                    'height': crop_data.shape[1],
                    'count': crop_data.shape[0],
                    'crs': src_profile['crs'],
                    'transform': t,
                    'photometric': "RGB"
                }
                profile.update(opt['profile'])
                with rasterio.open(patch_path, 'w', **profile) as dst:
                    dst.write(crop_data)
                # geo_data_util.tif_to_png(patch_path)
    return 'Processing {:s} ...'.format(img_name)


def test_worker():
    mode = 'single'  # single (one input folder) | pair (extract corresponding GT and LR pairs)
    opt = dict()

    # GeoTiff file properties
    profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'nodata': None,
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'compress': 'deflate',
        'interleave': 'band'
    }
    opt['profile'] = profile
    if mode == 'single':

        opt['satellite'] = 'landsat8'
        opt['channels'] = (4, 3, 2)
        opt['save_folder'] = r'D:\landsat8_sub_images'
        opt['crop_sz'] = 160  # the size of each sub-image
        opt['step'] = 80  # step of the sliding crop window
        opt['thres_sz'] = 16  # size threshold

        save_folder = opt['save_folder']
        if not osp.exists(save_folder):
            os.makedirs(save_folder)
            print('mkdir [{:s}] ...'.format(save_folder))

        test_l8_path = r'D:\Landsat8_China\LC08_L1TP_120039_20190812_20190820_01_T1\LC08_L1TP_120039_20190812_20190820_01_T1_MTL.txt'
        landsat8_worker(test_l8_path, opt)

        # test_s2_path = r'D:\Sentinel2_China\S2A_MSIL1C_20190714T025551_N0208_R032_T50SMA_20190714T055344.zip'
        # sentinel2_worker(test_s2_path, opt)

    elif mode == 'sat_pair':
        # testset 1
        # hr_path = 'S2A_MSIL1C_20190813T043701_N0208_R033_T46SDF_20190813T074203.zip'
        # lr_path = 'LC08_L1TP_138035_20190725_20190801_01_T1/LC08_L1TP_138035_20190725_20190801_01_T1_MTL.txt'

        # # testset 2
        # hr_path = 'S2A_MSIL1C_20190811T035541_N0208_R004_T48STA_20190811T073553.zip'
        # lr_path = 'LC08_L1TP_131038_20190825_20190903_01_T1/LC08_L1TP_131038_20190825_20190903_01_T1_MTL.txt'
        # hr_path = "S2B_MSIL1C_20190816T035549_N0208_R004_T47RQM_20190816T082533.zip"
        # lr_path = "LC08_L1TP_131040_20190825_20190903_01_T1/LC08_L1TP_131040_20190825_20190903_01_T1_MTL.txt"

        # hr_path = "S2B_MSIL1C_20190816T035549_N0208_R004_T48RTV_20190816T082533.zip"
        # lr_path = 'LC08_L1TP_131038_20190825_20190903_01_T1/LC08_L1TP_131038_20190825_20190903_01_T1_MTL.txt'
        # hr_path = "S2B_MSIL1C_20190816T035549_N0208_R004_T47RPL_20190816T082533.zip"
        # lr_path = "LC08_L1TP_131041_20190825_20190903_01_T1/LC08_L1TP_131041_20190825_20190903_01_T1_MTL.txt"

        # testset 3
        # hr_path = "S2B_MSIL1C_20190818T025549_N0208_R032_T50RKR_20190818T063225.zip"
        # lr_path = "LC08_L1TP_122041_20190725_20190801_01_T1/LC08_L1TP_122041_20190725_20190801_01_T1_MTL.txt"

        # testset 4
        # hr_path = "S2B_MSIL1C_20190831T030549_N0208_R075_T50SLJ_20190831T072516.zip"
        # lr_path = "LC08_L1TP_124033_20190723_20190801_01_T1/LC08_L1TP_124033_20190723_20190801_01_T1_MTL.txt"

        hr_path = "S2B_MSIL1C_20190923T031539_N0208_R118_T48RYQ_20190923T073039.zip"
        lr_path = "LC08_L1TP_126042_20190923_20190926_01_T1/LC08_L1TP_126042_20190923_20190926_01_T1_MTL.txt"
        hr_dir = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/sentinel2"
        lr_dir = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/landsat8"

        GT_path = os.path.join(hr_dir, hr_path)
        LR_path = os.path.join(lr_dir, lr_path)
        opt['dst_hr_folder'] = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/landsat8_sub"
        opt['dst_lr_folder'] = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/sentinel2_sub"
        opt['crop_sz'] = 480
        opt['step'] = 480
        opt['thres_sz'] = 48
        opt['bands'] = (4, 3, 2)
        log = sat_pair_worker(GT_path, LR_path, opt)
        print(log)


def test_results():
    test_path = r'D:\Landsat8_China\LC08_L1TP_120039_20190812_20190820_01_T1\LC08_L1TP_120039_20190812_20190820_01_T1_MTL.txt'
    sub_img_path = r'D:\landsat8_sub_images\LC08_L1TP_120039_20190812_20190820_01_T1_C000_N0581.TIF'
    scene = LandsatScene(test_path)
    with rasterio.open(sub_img_path, 'r') as src:
        sub_data = src.read()
    tif_to_png(sub_img_path)

    # test_s2_path = r'D:\Sentinel2_China\S2A_MSIL1C_20190714T025551_N0208_R032_T50SMA_20190714T055344.zip'
    # sub_img_path = r'D:\landsat8_sub_images\S2A_MSIL1C_20190714T025551_N0208_R032_T50SMA_20190714T055344_C000_N0008.TIF'
    # with s2_open(test_s2_path) as scene:
    #     granual = scene.granules[0]
    #     with rasterio.open(sub_img_path, 'r') as src:
    #         sub_data = src.read()


def _translation(arr1, arr2):
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
    cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria, None, 5)
    return warp_matrix[:, 2], cc


if __name__ == '__main__':
    # test_worker()
    # test_results()
    main()

