import sys
import os.path as osp
import os
import cv2
import numpy as np
from utils import get_image_paths, read_img


img_dir = "/home/wjw/data/sr_datasets/paird_geo_data/test_hr_plus/"
save_dir = "/home/wjw/data/sr_datasets/paird_geo_data/test_hr_plus_png/"
os.makedirs(save_dir, exist_ok=True)
img_paths, _ = get_image_paths('img', img_dir)
print('total imgs: {}'.format(len(img_paths)))
for i, path in enumerate(img_paths):
    img_LQ = read_img(None, path, None).transpose(1, 2, 0)
    img = np.round(np.clip(img_LQ, 0, 0.3) / 0.3 * 255).clip(0, 255).astype('u1')[:, :, ::-1]
    img_name = os.path.basename(path).replace('TIF', 'png')
    cv2.imwrite(os.path.join(save_dir, img_name), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
    print("processing... {}/{}".format(i + 1, len(img_paths)))
