from glob import glob
import os
import random
import shutil
import math
from PIL import Image
import torch
from torchvision import utils
import numpy as np
import utils


def main():
    select_num = 100
    source_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/checked_sentinel2_sub/"
    source_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/checked_landsat8_sub/"
    dest_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/test_hr/"
    dest_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/test_lr/"

    all_images = glob(os.path.join(source_root1, '*.TIF'))
    selected_iamges = random.sample(all_images, select_num)
    for path1 in selected_iamges:
        img_name = os.path.basename(path1)
        path2 = os.path.join(source_root2, img_name)
        if os.path.isfile(path2):
            shutil.move(path1, dest_root1)
            shutil.move(path2, dest_root2)
        else:
            raise NotImplementedError('corresponding file was not found')
    print('successfully selected {} images'.format(select_num))


def sanp_shot():
    img_dir = '/home/ubuntu/data5/WangJW/datasets/DIV2K_StyleGAN_256x256/test_unseen_LR'
    save_dir = '/home/ubuntu/data5/WangJW/datasets/DIV2K_StyleGAN_256x256'
    save_img_name = 'unseen_LR'
    img_paths, _ = get_image_paths('img', img_dir)   # modify this function by yourself
    img_list = []
    for path in img_paths:
        # get LQ image
        img_LQ = read_img(None, path, None, 'center')   # modify this function by yourself
        H, W, C = img_LQ.shape

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_LQ.shape[2] == 3:
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float().unsqueeze(0)
        img_list.append(img_LQ)
    save_img_path = os.path.join(save_dir, '{}.png'.format(save_img_name))
    image = torch.cat(img_list)
    utils.save_image(image, save_img_path, nrow=math.ceil(math.pow(image.shape[0], 0.5)), normalize=True, range=(-1, 1))


def geo_sanp_shot():
    img_dir = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/test_hr/"
    save_dir = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/"
    save_img_name = 'test_HR'
    img_paths, _ = utils.get_image_paths('img', img_dir)
    img_list = []
    for path in img_paths:
        # get LQ image
        # img_LQ = imresize(geo_data_util.read_img(None, path, None), scale_factor=3, mode='bicubic')
        img_LQ = utils.read_img(None, path, None)
        img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float().unsqueeze(0)
        img_list.append(img_LQ)
    save_img_path = os.path.join(save_dir, '{}.png'.format(save_img_name))
    image = torch.cat(img_list)
    utils.save_image(image, save_img_path, nrow=math.ceil(math.pow(image.shape[0], 0.5)), normalize=True, range=(0.03, 0.28))


def select_and_view():
    select_num = 200
    source_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/sentinel2_sub/"
    source_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/landsat8_sub/"
    dest_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_hr"
    dest_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_lr"
    os.makedirs(dest_root1, exist_ok=True)
    os.makedirs(dest_root2, exist_ok=True)
    all_images = glob(os.path.join(source_root1, '*.TIF'))
    for num, idx in enumerate(range(0, len(all_images), select_num)):
        sub_list = all_images[idx:idx + select_num]
        tem_hr_dir = os.path.join(dest_root1, f'{num}')
        tem_lr_dir = os.path.join(dest_root2, f'{num}')
        os.makedirs(tem_hr_dir)
        os.makedirs(tem_lr_dir)
        for path1 in sub_list:
            img_name = os.path.basename(path1)
            path2 = os.path.join(source_root2, img_name)
            if os.path.isfile(path2):
                shutil.copy(path1, tem_hr_dir)
                shutil.copy(path2, tem_lr_dir)
            else:
                raise NotImplementedError('corresponding file was not found')

        img_paths, _ = utils.get_image_paths('img', tem_hr_dir)
        img_list = []
        for path in img_paths:
            # get LQ image
            # img_LQ = imresize(geo_data_util.read_img(None, path, None), scale_factor=3, mode='bicubic')
            img_LQ = utils.read_img(None, path, None)
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float().unsqueeze(0)
            img_list.append(img_LQ)
        save_img_path = os.path.join(dest_root1, '{}.png'.format(num))
        image = torch.cat(img_list)
        utils.save_image(image, save_img_path, nrow=20, normalize=True,
                         range=(0.03, 0.28))

        img_paths, _ = utils.get_image_paths('img', tem_lr_dir)
        img_list = []
        for path in img_paths:
            # get LQ image
            img_LQ = utils.imresize(utils.read_img(None, path, None), scale_factor=3, mode='bicubic')
            # img_LQ = geo_data_util.read_img(None, path, None)
            img_LQ = torch.from_numpy(np.ascontiguousarray(img_LQ)).float().unsqueeze(0)
            img_list.append(img_LQ)
        save_img_path = os.path.join(dest_root1, '{}_BicUp.png'.format(num))
        image = torch.cat(img_list)
        utils.save_image(image, save_img_path, nrow=20, normalize=True,
                         range=(0.03, 0.28))


def delete_invalid_imgs():
    num = 29
    idx = [1,8,38,39,41,90,123,124,125,141]
    source_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/checked_sentinel2_sub/"
    source_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/checked_landsat8_sub/"
    dest_root1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_hr/{}".format(num)
    dest_root2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_lr/{}".format(num)
    os.makedirs(source_root1, exist_ok=True)
    os.makedirs(source_root2, exist_ok=True)

    hr_img_paths, _ = utils.get_image_paths('img', dest_root1)
    lr_img_paths, _ = utils.get_image_paths('img', dest_root2)
    for i in idx:
        os.remove(hr_img_paths[i - 1])
        os.remove(lr_img_paths[i - 1])

    hr_img_paths, _ = utils.get_image_paths('img', dest_root1)
    lr_img_paths, _ = utils.get_image_paths('img', dest_root2)
    num_copied = 0
    for _, path1 in enumerate(hr_img_paths):
        img_name = os.path.basename(path1)
        path2 = os.path.join(dest_root2, img_name)
        if os.path.isfile(path2):
            shutil.copy(path1, source_root1)
            shutil.copy(path2, source_root2)
            num_copied += 1
        else:
            raise NotImplementedError('corresponding file was not found')
    print('successfully copied {} images'.format(num_copied))


def img_ratio():
    total_num = 30
    for i in range(total_num):
        img_path1 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_hr/{}.png".format(i)
        img_path2 = "/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_hr/{}_BicUp.png".format(i)
        img1 = np.asarray(Image.open(img_path1)).astype('f4')
        img2 = np.asarray(Image.open(img_path2)).astype('f4')
        dif = np.abs(img2 - img1).astype('u1')
        dif_img = Image.fromarray(dif)
        dif_img.save("/home/ubuntu/data5/WangJW/datasets/pair_satellite_dataset/tem_check_hr/{}_dif.png".format(i))


if __name__ == '__main__':
    # main()
    geo_sanp_shot()
    # select_and_view()
    # delete_invalid_imgs()
    # img_ratio()
