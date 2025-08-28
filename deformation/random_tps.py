import time

import cv2
import numpy as np
import os

import decord
import torch
import random


def tps_transform_pure(img, grid_img, std=1.0, grid_size=32):
    """
    :param img: original image before warp
    :return: image warped by random tps transform
    """
    tps = cv2.createThinPlateSplineShapeTransformer()
    tps.setRegularizationParameter(0.1)
    shape = img.shape
    # create sshape uniformly distributed in the image
    num_y = shape[0] // grid_size
    num_x = shape[1] // grid_size
    x = np.linspace(0, shape[1], num_x)
    y = np.linspace(0, shape[0], num_y)
    xx, yy = np.meshgrid(x, y)
    sshape = np.zeros((num_x * num_y, 2))
    sshape[:, 0] = xx.flatten()
    sshape[:, 1] = yy.flatten()
    sshape = sshape.astype(np.float32)
    # create tshape based on sshape, using normal distribution with mean and std
    random_vector = np.random.normal(0, std, sshape.shape).astype(np.float32)
    tshape = random_vector + sshape
    sshape = sshape.reshape(1, -1, 2)
    tshape = tshape.reshape(1, -1, 2)
    matches = list()
    for i in range(sshape.shape[1]):
        matches.append(cv2.DMatch(i, i, 0))
    tps.estimateTransformation(tshape, sshape, matches)

    out_img = tps.warpImage(img, flags=cv2.INTER_LINEAR)  # XinT: do not use nearest for raw image
    if grid_img:
        out_grid_img = tps.warpImage(grid_img, flags=cv2.INTER_LINEAR)  # XinT: do not use nearest for raw image
    else:
        out_grid_img = None

    return out_img, out_grid_img  # , random_vector


def generate_video_TPS(video, grid_size=32, check=True, rank_num=4):
    # todo: what if the input is a tensor?
    # # suppose the video here is a tensor of size(1, 3, 32, 256, 256) (N, channel, clip_len, H_size, W_size)
    # video = video.cpu().detach().float().squeeze().numpy()  # convert to ndarray of size (3, 32, 256, 256)
    # video_gray = video[0, :, :, :]  # gray scale original sequence images

    # the input from decord is actually ndarray of size (3, 32, H_size, W_size)
    if len(video.shape) == 4:
        video_gray = video[0, :, :, :]  # gray scale original sequence images
    else:
        video_gray = video

    if rank_num == 4:  # training and validating use
        std_min, std_max = 0.99, 3.00  # todo: is the range reasonable?
    else:
        std_min, std_max = 0.33, 0.33 * rank_num

    degraded_videos = []
    for std in np.arange(std_min, std_max, std_min):
        tps_video = np.zeros((video_gray.shape[0], 224, 224), dtype='uint8')
        # read input sequence and apply random tps transform
        t_s = time.perf_counter()
        for z in range(video_gray.shape[0]):
            img = video_gray[z, :, :]
            if z == 0:  # do not change the first reference image
                tps_video[z, :, :] = img[10:10 + 224, 10:10 + 224]
            else:  # do tps deformation
                tps_img, _ = tps_transform_pure(img, None, std=std, grid_size=grid_size)
                tps_video[z, :, :] = tps_img[10:10+224, 10:10+224]  # todo: change to central roi?
        t_e = time.perf_counter()
        # print('Time consuming for single serial tps transform: {:.2f}'.format(t_e - t_s))
        degraded_videos.append(tps_video)
        if check:  # just for ranking check
            write_deformed_image_for_check(std, grid_size, tps_video)

    return degraded_videos  # todo: what to return? list, ndarray, tensor?


def write_deformed_image_for_check(std, grid_size, tps_video):
    """
     temporal function for deformation check
    """
    out_dir = './data_for_check/tps_gs{}_std{:.2f}/'.format(grid_size, std)
    os.makedirs(out_dir, exist_ok=True)

    for z in range(tps_video.shape[0]):
        cv2.imwrite(out_dir + str(z).zfill(3) + '.png', tps_video[z, :, :])


def generate_ranking_video_tps(video, mean=123.675, std=58.395, rank_num=4, check=False):
    """
        Mean value and std value from FusionDataset
        self.mean = torch.FloatTensor([123.675, 116.28, 103.53])
        self.std = torch.FloatTensor([58.395, 57.12, 57.375])
        Only need first channel, may need to be changed in the future
    """
    original_video = video['resize'].squeeze(dim=0)[0]  # simply get channel 0
    original_video = original_video.cpu().detach().numpy()
    original_video = (original_video * std + mean).astype('uint8')  # retrieve uint8 data for tps

    # todo: random data augmentation using flip
    original_video = np.flip(original_video, axis=random.randint(0, 2))

    degraded_videos = generate_video_TPS(original_video, grid_size=32,
                                         check=check, rank_num=rank_num)  # do random tps deformation
    ranked_videos = torch.FloatTensor(np.array([original_video[:, 10:10 + 224, 10:10 + 224]]
                                               + degraded_videos))  # convert to tensor
    r_v = ranked_videos.unsqueeze(dim=1)
    r_v = r_v.repeat(1, 3, 1, 1, 1)  # (4, 3, 32, 224, 224) tensor
    r_v = ((r_v - mean) / std)  # normalize?
    return r_v




