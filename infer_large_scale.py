import os
import time

from PIL import Image
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"  # avoid traffic jam

from skimage.util import view_as_blocks, view_as_windows

from inference_whole_block_v2 import *

from concurrent.futures import ThreadPoolExecutor


class Assess_Large_Scale_Data:
    def __init__(self, vol_size=224, loss='Rankloss', eval=True, mlp_pth='./em_weights/cnn_model.pth'):
        # set sub-volume size
        self.vol_size = vol_size
        # initialize evaluator (StableVQA) and scorer (Q-Align)
        if eval:
            print('Eval mode on!\n')
            self.evaluator, self.scorer, self.opt = initialize_models(loss=loss, mlp_pth=mlp_pth)

    # TODO: 数据如何输入，如何转换为MP4或其他共模型推理使用的视频，数据体量？
    def convert_and_save_sp_and_cv_videos(self, raw_image_path, video_save_path, area_size=None):
        """
            假定输入为H*W*224的薄片型图像数据，分辨率为拆分为H/224 * W/224个体块
            随后转换为对应的切片平面(sp)与侧切面(cv)视频并分别存储，文件夹名称设置为area0_0这种形式？
        """
        def load_images_to_ndarray(folder_path, area_size=None):
            """
            Load images from the given folder and store them in a ndarray.
            """
            # Get sorted list of image files
            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')
                                  or f.endswith('.tif') or f.endswith('.jpg')])  # 只读取图像文件

            # Read the first image to determine its size
            first_image = Image.open(os.path.join(folder_path, image_files[0]))
            raw_img_size = first_image.size  # (width, height)
            if area_size is None:
                image_size = raw_img_size
            else:
                image_size = (area_size, area_size)

            # Initialize ndarray to hold all images (depth, height, width)
            num_images = len(image_files)
            image_array = np.zeros((num_images, image_size[1], image_size[0]), dtype=np.uint8)

            # Load images into the ndarray
            for i, file_name in enumerate(image_files):
                img = Image.open(os.path.join(folder_path, file_name))
                if area_size is None:
                    image_array[i] = np.array(img)
                else:
                    image_array[i] = np.array(img)[raw_img_size[1]//2 - area_size//2:raw_img_size[1]//2 + area_size//2,
                                     raw_img_size[0]//2 - area_size//2:raw_img_size[0]//2 + area_size//2]

            return image_array

        def calculate_stride(image_size, block_size):
            """
            自动计算 stride，确保尽量减少重叠区域。
            """
            if image_size <= block_size:
                return block_size  # 图像小于块大小，无需重叠

            # 理想块数 (向上取整)
            num_blocks = math.ceil(image_size / block_size)
            stride = (image_size - block_size) // (num_blocks - 1)
            return stride

        def divide_image_with_min_overlap(image, block_size):
            """
            自动划分图像为最小重叠块。
            """
            strides = tuple(calculate_stride(image.shape[i], block_size[i]) for i in range(len(block_size)))
            blocks = view_as_windows(image, block_size, strides)
            return blocks, strides

        # Specify folder containing the images
        folder_path = raw_image_path

        # Load images into a ndarray
        print('Loading raw images into a 3D volume\n')
        image_ndarray = load_images_to_ndarray(folder_path, area_size=area_size)  # returned in (depth, H, W)shape
        print('Volume shape: {}'.format(image_ndarray.shape))

        # Define block size
        block_size = (self.vol_size, self.vol_size, self.vol_size)

        # Divide the ndarray into blocks
        if image_ndarray.shape[0] % self.vol_size == 0 and image_ndarray.shape[1] % self.vol_size == 0 and image_ndarray.shape[2] % self.vol_size == 0:
            print('Divide with no-overlap')
            blocks = view_as_blocks(image_ndarray, block_size)
        else:
            print('Divide with minimum overlap')
            blocks, stride = divide_image_with_min_overlap(image_ndarray, block_size)

        num_d, num_h, num_w = blocks.shape[0], blocks.shape[1], blocks.shape[2]

        # TODO: convert block images into sp_xy.mp4, cv_xz.mp4 and cv_yz.mp4
        print('Saving block videos\n')
        for d_idx in tqdm(range(num_d)):  # depth in z-axis
            for h_idx in tqdm(range(num_h)):  # height
                for w_idx in tqdm(range(num_w)):  # width
                    block = blocks[d_idx, h_idx, w_idx].squeeze()
                    # for direction in ['sp_xy', 'cv_xz', 'cv_yz']:
                    for direction in ['sp_xy', 'cv_yz']:  # TODO：see the comments inside
                        self.convert_block_into_videos(block, d_idx, h_idx, w_idx, video_save_path, direction)

        print('All sub-volumes saved into sp and cv videos!')

    def convert_and_save_sp_and_cv_videos_SSH(self, raw_image_path, video_save_path,
                                              depth, area_size, center, h_max):
        """
            针对时松海数据进行特化，用于超大规模数据读取与分块
        """
        def load_images_to_ndarray(folder_path, area_size, depth):
            """
            Load images from the given folder and store them in a ndarray.
            """
            # Get sorted list of image files
            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')
                                  or f.endswith('.tif') or f.endswith('.jpg')])  # 只读取图像文件

            image_files = image_files[depth * 224:(depth + 1) * 224]  # 只读取指定深度的图像内容

            image_size = (area_size[0], area_size[1])

            # Initialize ndarray to hold all images (depth, height, width)
            num_images = len(image_files)
            image_array = np.zeros((num_images, image_size[0], image_size[1]), dtype=np.uint8)

            def load_image(i, file_name):
                """ 读取图像并提取ROI """
                img = cv2.imread(os.path.join(folder_path, file_name), 0)  # 读取为灰度图
                roi = img[center[0] - area_size[0] // 2:center[0] + area_size[0] // 2,
                          center[1] - area_size[1] // 2:center[1] + area_size[1] // 2]
                return i, roi

            # 多线程读取数据
            with ThreadPoolExecutor(max_workers=8) as executor:  # 8 线程
                results = list(
                    tqdm(executor.map(lambda args: load_image(*args), enumerate(image_files)), total=num_images))

            # 按照索引填充数组
            for i, roi in results:
                image_array[i] = roi

            return image_array

        def calculate_stride(image_size, block_size):
            """
            自动计算 stride，确保尽量减少重叠区域。
            """
            if image_size <= block_size:
                return block_size  # 图像小于块大小，无需重叠

            # 理想块数 (向上取整)
            num_blocks = math.ceil(image_size / block_size)
            stride = (image_size - block_size) // (num_blocks - 1)
            return stride

        def divide_image_with_min_overlap(image, block_size):
            """
            自动划分图像为最小重叠块。
            """
            strides = tuple(calculate_stride(image.shape[i], block_size[i]) for i in range(len(block_size)))
            blocks = view_as_windows(image, block_size, strides)
            return blocks, strides

        # Specify folder containing the images
        folder_path = raw_image_path

        # Load images into a ndarray
        print('Loading raw images into a 3D volume\n')
        t_s = time.perf_counter()
        image_ndarray = load_images_to_ndarray(folder_path, area_size=area_size, depth=depth)  # returned in (depth, H, W)shape
        t_e = time.perf_counter()
        print('Volume shape: {}'.format(image_ndarray.shape))
        print('Loading time {:.2f}s'.format(t_e - t_s))

        # Define block size
        block_size = (self.vol_size, self.vol_size, self.vol_size)

        # Divide the ndarray into blocks
        if image_ndarray.shape[0] % self.vol_size == 0 and image_ndarray.shape[1] % self.vol_size == 0 and image_ndarray.shape[2] % self.vol_size == 0:
            print('Divide with no-overlap')
            blocks = view_as_blocks(image_ndarray, block_size)
        else:
            print('Divide with minimum overlap')
            blocks, stride = divide_image_with_min_overlap(image_ndarray, block_size)

        num_d, num_h, num_w = blocks.shape[0], blocks.shape[1], blocks.shape[2]
        if h_max:  # TODO： 0305晚临时应急，主要解决视频写到h39 w73发生的意外中断
            num_h = h_max

        # TODO: convert block images into sp_xy.mp4, cv_xz.mp4 and cv_yz.mp4
        print('Saving block videos\n')
        for d_idx in range(num_d):  # depth in z-axis
            for h_idx in tqdm(range(num_h)):  # height
                for w_idx in range(num_w):  # width
                    block = blocks[d_idx, h_idx, w_idx].squeeze()
                    for direction in ['sp_xy', 'cv_yz']:  # TODO：see the comments inside
                        self.convert_block_into_videos(block, d_idx, h_idx, w_idx, video_save_path, direction)

        print('All sub-volumes saved into sp and cv videos!')

    def convert_and_save_sp_and_cv_videos_SSH_comp(self, raw_image_path, video_save_path,
                                                   depth, area_size, center,
                                                   h_last, ws, we):
        """
            0306单独用于补全时松海数据生成失败部分的代码
        """
        def load_images_to_ndarray(folder_path, area_size, depth):
            """
            Load images from the given folder and store them in a ndarray.
            """
            # Get sorted list of image files
            image_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.png')
                                  or f.endswith('.tif') or f.endswith('.jpg')])  # 只读取图像文件

            image_files = image_files[depth * 224:(depth + 1) * 224]  # 只读取指定深度的图像内容

            image_size = (area_size[0], area_size[1])

            # Initialize ndarray to hold all images (depth, height, width)
            num_images = len(image_files)
            image_array = np.zeros((num_images, image_size[0], image_size[1]), dtype=np.uint8)

            def load_image(i, file_name):
                """ 读取图像并提取ROI """
                img = cv2.imread(os.path.join(folder_path, file_name), 0)  # 读取为灰度图
                roi = img[center[0] - area_size[0] // 2:center[0] + area_size[0] // 2,
                          center[1] - area_size[1] // 2:center[1] + area_size[1] // 2]
                return i, roi

            # 多线程读取数据
            with ThreadPoolExecutor(max_workers=8) as executor:  # 8 线程
                results = list(
                    tqdm(executor.map(lambda args: load_image(*args), enumerate(image_files)), total=num_images))

            # 按照索引填充数组
            for i, roi in results:
                image_array[i] = roi

            return image_array

        def calculate_stride(image_size, block_size):
            """
            自动计算 stride，确保尽量减少重叠区域。
            """
            if image_size <= block_size:
                return block_size  # 图像小于块大小，无需重叠

            # 理想块数 (向上取整)
            num_blocks = math.ceil(image_size / block_size)
            stride = (image_size - block_size) // (num_blocks - 1)
            return stride

        def divide_image_with_min_overlap(image, block_size):
            """
            自动划分图像为最小重叠块。
            """
            strides = tuple(calculate_stride(image.shape[i], block_size[i]) for i in range(len(block_size)))
            blocks = view_as_windows(image, block_size, strides)
            return blocks, strides

        # Specify folder containing the images
        folder_path = raw_image_path

        # Load images into a ndarray
        print('Loading raw images into a 3D volume\n')
        t_s = time.perf_counter()
        image_ndarray = load_images_to_ndarray(folder_path, area_size=area_size, depth=depth)  # returned in (depth, H, W)shape
        t_e = time.perf_counter()
        print('Volume shape: {}'.format(image_ndarray.shape))
        print('Loading time {:.2f}s'.format(t_e - t_s))

        # Define block size
        block_size = (self.vol_size, self.vol_size, self.vol_size)

        # Divide the ndarray into blocks
        if image_ndarray.shape[0] % self.vol_size == 0 and image_ndarray.shape[1] % self.vol_size == 0 and image_ndarray.shape[2] % self.vol_size == 0:
            print('Divide with no-overlap')
            blocks = view_as_blocks(image_ndarray, block_size)
        else:
            print('Divide with minimum overlap')
            blocks, stride = divide_image_with_min_overlap(image_ndarray, block_size)

        num_d, num_h, num_w = blocks.shape[0], blocks.shape[1], blocks.shape[2]

        # TODO: convert block images into sp_xy.mp4, cv_xz.mp4 and cv_yz.mp4
        print('Saving block videos for last row {}\n'.format(h_last))
        for w_idx in tqdm(range(ws, we)):  # width
            block = blocks[0, h_last, w_idx].squeeze()
            for direction in ['sp_xy', 'cv_yz']:  # TODO：see the comments inside
                self.convert_block_into_videos(block, 0, h_last, w_idx, video_save_path, direction)

        print('Lost sub-volumes saved into sp and cv videos!')

    def convert_block_into_videos(self, block, d_idx, h_idx, w_idx, save_path, direction):
        """
            将获取到的block保存为对应的MP4文件，便于后续计算SAVIOR指标数值
            单个block shape为224**3
            block 维度顺序为depth，height，width
        """
        os.makedirs(save_path + 'b_d{}_h{}_w{}/'.format(d_idx, h_idx, w_idx), exist_ok=True)

        frame_height, frame_width = self.vol_size, self.vol_size
        fps = 5
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed (e.g., 'XVID', 'MJPG')

        # for direction in ['sp_xy', 'cv_xz', 'cv_yz']:
        # Save section plane video

        out = cv2.VideoWriter(save_path + 'b_d{}_h{}_w{}/{}.mp4'.format(d_idx, h_idx, w_idx, direction),
                              fourcc, fps, (frame_height, frame_width), isColor=False)
        if direction == 'sp_xy':
            for frame_id in range(self.vol_size):
                frame = block[frame_id, :, :]
                out.write(frame)
        # TODO: FAFB area 5数据在这里有点问题，暂时跳过x_z方向生成
        # if direction == 'cv_xz':
        #     for frame_id in range(self.vol_size):
        #         frame = block[:, frame_id, :]
        #         # if frame_id == 115:
        #         #     print('Watch out!!')
        #         out.write(frame)
        if direction == 'cv_yz':
            for frame_id in range(self.vol_size):
                frame = block[:, :, frame_id]
                out.write(frame)
        out.release()

    def infer_videos_from_single_volume(self, root_dir, cv_direction='cv_yz', post_fix='.mp4', raw=False, lamda=0.428):
        """
            Scoring function changed from infernce_whole_block_v2.py
            Only consider single cross-view this time?
        """
        # print('Root dir is: ' + root_dir)
        sp_video_path = root_dir + 'sp_xy' + post_fix
        score_sp = single_video_score(sp_video_path, self.evaluator, self.opt, show_score=False)

        cv_video_path = root_dir + cv_direction + post_fix
        cv_video_list = [load_video(cv_video_path)]
        score_cv = self.scorer(cv_video_list).tolist()[0]

        print('Raw Scores sp, cv are: {:.3f}, {:.3f}'.format(score_sp, score_cv))

        if raw:
            return score_sp, score_cv, 0
        else:
            re_sp, re_cv, final = change_score(score_sp, score_cv)
            return re_sp, re_cv, final


if __name__ == '__main__':
    ALSD = Assess_Large_Scale_Data(vol_size=224, loss='Rankloss', eval=False)

    """Minnie 65 Areas try"""
    # for area in range(1, 3):
    #     raw_root_dir = f'/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_images/' \
    #                    f'Minnie65/mip2_250820/area{area}/'
    #     video_save_path = f'/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                       f'Minnie65/mip2_250820/area{area}/'
    #     ALSD.convert_and_save_sp_and_cv_videos(raw_root_dir, video_save_path)

    """Compare FlyWire(v14.1) with FAFB v14 clahe, similar area"""
    # TODO: 3.91 path
    # img_root_dir = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_images/SAVIOR_250825/'
    # video_root_dir = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/SAVIOR_250825/'
    # TODO: 3.1 path
    img_root_dir = '/mnt/Ext001/chenhr/mira_datacenter/dataset/em_images/SAVIOR_250825/'
    video_root_dir = '/mnt/Ext001/chenhr/mira_datacenter/dataset/em_videos/SAVIOR_250825/'

    for data_name in ['FlyWire/32_32_40nm/area1/', 'FAFBv14/32_32_40nm/area1/']:
        image_dir = img_root_dir + data_name + 'sp_xy/'
        video_save_dir = video_root_dir + data_name
        ALSD.convert_and_save_sp_and_cv_videos(image_dir, video_save_dir)

    """FAFB test"""
    # raw_root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FAFB_v14/SAVIOR_data/area5/'
    # video_save_path = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_data/SAVIOR_data/area5/'

    """ SSH data test"""
    # raw_root_dir = '/mnt/Ext001/chenhr/dataset/em_images/SSH_down8/7_12_partial/'
    # video_save_path = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_down8/7_12_partial/'
    #
    # area_name = '7_30'
    # raw_root_dir = '/mnt/Ext001/chenhr/dataset/em_images/SSH_down8/' + area_name + '/'
    # video_save_path = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_down8/' + area_name + '/'
    #
    # ALSD.convert_and_save_sp_and_cv_videos(raw_root_dir, video_save_path)

    # ALSD = Assess_Large_Scale_Data(vol_size=224, loss='Rankloss', eval=True)
    # # FAFB area 5, 5*5 = 25 blocks test, old version without depth
    # raw_root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FAFB_v14/SAVIOR_data/area5/'
    # video_save_path = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_data/SAVIOR_data/area5/'

    # from visualize_plot import *
    # # 边界那种小区域的分数会出现负分，直接设置为0
    # final_heatmap = np.array([[0.93, 0.76, 0.82, 0.75, 0.80], [0.92, 0.80, 0.81, 0.82, 0.77],
    #                           [0.94, 0.83, 0.82, 0.80, 0.75], [0.87, 0.73, 0.72, 0.41, 0],
    #                           [0.35, 0.52, 0, 0, 0]])
    # draw_heatmaps_with_scores(final_heatmap, 0, video_save_path, True, True)

    # import time
    # t_s = time.perf_counter()
    # final_heatmap = np.zeros((5, 5))
    # for h in range(5):
    #     for w in range(5):
    #         print('Scoring block h{} w{}'.format(h, w))
    #         block_video_dir = video_save_path + 'b_h{}_w{}/'.format(h, w)
    #         re_sp, re_cv, final = ALSD.infer_videos_from_single_volume(block_video_dir, 'cv_yz', lamda=0.572)
    #         final_heatmap[h, w] = final
    # t_e = time.perf_counter()
    # print('Inference time for 25 blocks: {:.3f}s'.format(t_e - t_s))
    # from visualize_plot import *
    # draw_heatmaps_with_scores(final_heatmap, 0, video_save_path, True, True)

    # video_save_path = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_down8/7_30/'
    # d, h, w = 11, 5, 5
    #
    # with open(os.path.join(video_save_path, 'SAVIOR_Scores.txt'), 'w+') as f:
    #     out_line = 'd, h, w, re_sp, re_cv, final\n'
    #     f.write(out_line)
    #
    # import time
    # t_s = time.perf_counter()
    # for d in range(10):
    #     for h in range(5):
    #         for w in range(5):
    #             print('Scoring block d{} h{} w{}'.format(d, h, w))
    #             block_video_dir = video_save_path + 'b_d{}_h{}_w{}/'.format(d, h, w)
    #             re_sp, re_cv, final = ALSD.infer_videos_from_single_volume(block_video_dir, 'cv_yz')
    #             out_line = '{}, {}, {}, {:.3f}, {:.3f}, {:.3f}\n'.format(d, h, w, re_sp, re_cv, final)
    #             with open(os.path.join(video_save_path, 'SAVIOR_Scores.txt'), 'a') as f:
    #                 f.write(out_line)
    #
    # t_e = time.perf_counter()
    # print('Inference time for {} blocks: {:.3f}s'.format(d*h*w, t_e - t_s))

    # # 20241223 SSH down 8 in section-plane then down 4 in all axis
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_down8/ds4_allaxis/7_12_p_videos/'
    # ALSD.infer_videos_from_single_volume(root_dir, cv_direction='cv_xz')
    # ALSD.infer_videos_from_single_volume(root_dir, cv_direction='cv_yz')

    # """ 2025 01 13 使用SSH down8 原始与ARFlow重新配准sub-blocks 构造SAVIOR新训练集 """
    # ALSD = Assess_Large_Scale_Data(vol_size=224, loss='Rankloss', eval=False)
    #
    # SSH_raw_dir = '/mnt/Ext001/chenhr/dataset/em_images/SSH_down8/short_seqs/'
    # SSH_realign_dir = '/mnt/Ext001/chenhr/dataset/em_images/SSH_down8/ARFlow_realigned/'
    # valid_area_lst = ['7_12', '7_30', '16_29']
    #
    # SSH_videos_dir = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_retrain/'
    #
    # # save into videos
    # print('Converting raw videos')
    # for valid_area in valid_area_lst:
    #     print('Aligning short_seqs from ' + valid_area)
    #     input_dir = SSH_raw_dir + valid_area + '/'
    #     raw_output_dir = SSH_videos_dir + '/raw/' + valid_area + '/'
    #     for seq_id in range(0, 11):  # short sequence numbers
    #         seq_dir = input_dir + 'short_seq{}/'.format(seq_id)
    #         seq_video_outdir = raw_output_dir + 'sq{}/'.format(seq_id)
    #         ALSD.convert_and_save_sp_and_cv_videos(seq_dir, seq_video_outdir, area_size=224*4)
    #
    # print('Converting re-aligned videos')
    # for valid_area in valid_area_lst:
    #     print('Aligning short_seqs from ' + valid_area)
    #     input_dir = SSH_realign_dir + valid_area + '/'
    #     re_output_dir = SSH_videos_dir + '/realigned/' + valid_area + '/'
    #     for seq_id in range(0, 11):  # short sequence numbers
    #         seq_dir = input_dir + 'short_seq{}/'.format(seq_id)
    #         seq_video_outdir = re_output_dir + 'sq{}/'.format(seq_id)
    #         ALSD.convert_and_save_sp_and_cv_videos(seq_dir, seq_video_outdir, area_size=224*4)





