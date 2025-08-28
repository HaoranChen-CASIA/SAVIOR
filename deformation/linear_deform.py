import numpy as np
import cv2

from deformation.random_tps import *
import random
import torch


def linear_transform_pure(img, mode, std, angle=361):
    rows, cols = img.shape

    # define a random deformation metric based on mode
    if mode == 'affine' or 'translation':
        p1 = np.float32([[0, 0], [cols // 2, 0], [0, rows // 2]])
        # create a random vector using normal distribution with zero-mean and std
        if mode == 'translation':
            trans_vec = np.random.normal(0, std, size=(1, 2)).astype(np.float32)
            dist = np.linalg.norm(trans_vec)
            random_vector = np.array([trans_vec, trans_vec, trans_vec]).squeeze(1)
        else:
            random_vector = np.random.normal(0, std, size=p1.shape).astype(np.float32)
        p2 = p1 + random_vector
        deform_mat = cv2.getAffineTransform(p1, p2)
    if mode == 'rotation':
        center = (cols // 2, rows // 2)
        if angle == 361:
            angle = np.random.randint(low=-15, high=15)
        deform_mat = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1)

    # warp original image
    warped_img = cv2.warpAffine(img, deform_mat, dsize=(cols, rows))

    return warped_img


def generate_video_linear_deformed(video, mode='affine', rank_num=4, check=False):
    # the input from decord is actually ndarray of size (3, 32, H_size, W_size)
    if len(video.shape) == 4:
        video_gray = video[0, :, :, :]  # gray scale original sequence images
    else:
        video_gray = video

    if rank_num == 4:  # training and validating mode
        std_min, std_max = 0.99, 3.00  # todo: is the range reasonable?
    else:
        std_min, std_max = 0.33,  0.33 * rank_num  # todo: try to test CC computation

    degraded_videos = []
    for std in np.arange(std_min, std_max, std_min):
        deformed_video = np.zeros((video_gray.shape[0], 224, 224), dtype='uint8')
        # read input sequence and apply random tps transform
        # t_s = time.perf_counter()
        for z in range(video_gray.shape[0]):
            img = video_gray[z, :, :]
            if z == 0:  # do not change the first reference image
                deformed_video[z, :, :] = img[10:10 + 224, 10:10 + 224]
            else:  # do linear deformation
                warped_img = linear_transform_pure(img, mode, std=std)
                deformed_video[z, :, :] = warped_img[10:10 + 224, 10:10 + 224]  # todo: change to central roi?
        # t_e = time.perf_counter()
        # print('Time consuming for single serial tps transform: {:.2f}'.format(t_e - t_s))
        degraded_videos.append(deformed_video)
        if check:  # just for ranking check
            write_deformed_image_for_check(std, mode, deformed_video)

    return degraded_videos  # todo: what to return? list, ndarray, tensor?


def write_deformed_image_for_check(std, mode, deformed_video):
    """
     temporal function for deformation check
    """
    out_dir = './data_for_check/linear_{}_std{:.2f}/'.format(mode, std)
    os.makedirs(out_dir, exist_ok=True)

    for z in range(deformed_video.shape[0]):
        cv2.imwrite(out_dir + str(z).zfill(3) + '.png', deformed_video[z, :, :])


def generate_ranking_video_linear(video, mean=123.675, std=58.395, mode='affine', rank_num=4, check=False):
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

    # do random linear deformation
    degraded_videos = generate_video_linear_deformed(original_video,
                                                     mode=mode, check=check, rank_num=rank_num)
    ranked_videos = torch.FloatTensor(np.array([original_video[:, 10:10 + 224, 10:10 + 224]]
                                               + degraded_videos))  # convert to tensor
    r_v = ranked_videos.unsqueeze(dim=1)
    r_v = r_v.repeat(1, 3, 1, 1, 1)  # (4, 3, 32, 224, 224) tensor
    r_v = ((r_v - mean) / std)  # normalize?
    return r_v

