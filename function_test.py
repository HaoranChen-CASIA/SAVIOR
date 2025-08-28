import os

from evaluate_sv_ranked import *
import warnings
import os
from deformation.stab_to_unstab import *

from ComparisonMethods.vqa_s_methods import *


def ranked_xy_evaluation(mode):
    print('\nEvaluator mode: ' + mode)
    warnings.filterwarnings('ignore')

    if 'sf+of' in mode:
        # Step 1: Initialize evaluator
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # avoid traffic jam
        device = "cuda" if torch.cuda.is_available() else "cpu"
        with open('./options/vRQA_Ablation.yml', "r") as f:
            opt = yaml.safe_load(f)
        print(opt)
        if mode == 'CCloss_sf+of':
            weights_path = './em_weights/vRQA2024_singleF_CCloss_val_sf+of.pth'
        else:  # ranked loss
            weights_path = './em_weights/vRQA2024_singleF_ranked_val_sf+of.pth'
        opt['model']['args']['feat_type'] = 'sf+of'
        opt['test_load_path'] = weights_path
        # load model
        evaluator = load_model(opt, device)

    # Step 2: load and score
    # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/ranked_videos/', '.mp4'
    root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/ranked_videos/combined_tps1/', '.mp4'
    # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/ranked_videos/combined_tps1/', '.mp4'
    print(root_dir)

    labels = []
    std_min = 0.33
    deform_amp = np.arange(0, std_min * 13, std_min)

    for subblock in [f'subblock{i}' for i in range(0, 10, 1)]:  # for FlyEM
    # for subblock in [f'subblock{i}' for i in range(0, 1, 1)]:  # for Lucchi++
        print('\nEvaluating ' + subblock)
        # subblock_path = root_dir + subblock + '/'
        subblock_path = root_dir + subblock + '/section_plane/'
        print('Loading and Scoring Videos...')
        score_list = []
        for video_name in [f'rank{i}' for i in range(0, 13, 1)]:
            video_path = subblock_path + video_name + post_fix
            if 'sf+of' in mode:
                score = single_video_score(video_path, evaluator, opt, show_score=False)
            elif mode == 'ITF' or mode == 'S_Score':
                score = read_in_and_score_videos(video_path, mode=mode)
            else:
                score = 0
                print('Illegal Mode!!')
            score_list.append(score)
        plt.plot(np.arange(13), score_list)
        labels.append(subblock)
        compute_CC(score_list, deform_amp)

    plt.title(mode + ' Scores list')
    plt.xlabel('Video clip index (ranks)')
    plt.ylabel(mode + ' Scores')
    plt.legend(labels=labels)
    plt.grid()
    plt.show()


def generate_and_save_subblocks(root_dir, format_fix='.mp4', mode='combined', rank_num=4, seq_length=32, block_num=200):
    print('Generating deformed&ranked volumes using {}'.format(root_dir))
    in_situ_video_dir = root_dir + 'in-situ_videos/'
    video_files = sorted([video for video in os.listdir(in_situ_video_dir) if video.endswith(format_fix)])
    for idx in tqdm(range(min(len(video_files), block_num))):
    # for idx in range(13, 25):
    #     print('\n Generating ranking sequence {}'.format(idx))
        input_video_name = in_situ_video_dir + str(idx).zfill(4) + format_fix
        save_path = root_dir + 'ranked_videos/{}/subblock{}/'.format(mode, idx)
        os.makedirs(save_path, exist_ok=True)
        # os.makedirs(save_path, exist_ok=True)
        generate_and_save_deformed_rank_videos(input_video_name, save_path,
                                               mode=mode, rank_num=rank_num, seq_length=seq_length)

    print('done')


def generate_and_save_cross_view(root_dir, format_fix='.mp4', mode='combined', rank_num=4, seq_length=32):
    in_situ_video_dir = root_dir + 'in-situ_videos/'
    video_files = sorted([video for video in os.listdir(in_situ_video_dir) if video.endswith(format_fix)])

    # for idx in range(len(video_files)):
    for idx in range(64, len(video_files)):
        print('\n Generating ranking sequence {}'.format(idx))
        input_video_name = in_situ_video_dir + str(idx).zfill(4) + format_fix
        save_path = root_dir + 'QAlign_data/{}/'.format(mode)
        os.makedirs(save_path, exist_ok=True)
        deformed_and_save_cross_view(input_video_name, idx, save_path, mode, rank_num, seq_length)

    print('done')


def deformed_and_save_cross_view(video_name, sb_id, save_path, mode='random tps', rank_num=13, seq_length=32):
    """
        input parameters:
    """
    # print('Deform in-situ video ', video_name)
    video_reader = decord.VideoReader(video_name)
    frame_dict = {idx: video_reader[idx] for idx in range(256)}
    imgs = [frame_dict[idx] for idx in range(256)]
    video_sampled = torch.stack(imgs, 0)
    video_sampled = video_sampled.permute(3, 0, 1, 2)

    video_sampled = video_sampled.unsqueeze(dim=0)
    video = {}
    video['resize'] = video_sampled

    original_video = video['resize'].squeeze(dim=0)[0]  # simply get channel 0
    original_video = original_video.cpu().detach().numpy()

    if mode == 'random_tps':
        ranked_videos = generate_video_TPS(original_video, grid_size=32,
                                           check=False, rank_num=rank_num)
    elif mode == 'random_affine':
        ranked_videos = generate_video_linear_deformed(original_video,
                                                       mode=mode, check=False, rank_num=rank_num)
    else:  # combined
        ranked_videos = generate_video_combined(original_video, rank_num=rank_num)

    ranked_videos.insert(0, original_video[:, 10:10+224, 10:10+224])

    corss_view_path = save_path + str(sb_id).zfill(4) + '/'  # 按subblock的来源分别存储
    os.makedirs(corss_view_path, exist_ok=True)

    save_cross_view_videos_for_training(ranked_videos, corss_view_path, 'xz', rank_num, seq_length)
    save_cross_view_videos_for_training(ranked_videos, corss_view_path, 'yz', rank_num, seq_length)


def read_and_extract_cross_view_image(rank, position):
    root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/ranked_videos/combined_tps1/subblock0/section_plane/'
    video_name = root_dir + 'rank{}.mp4'.format(rank)

    video_reader = decord.VideoReader(video_name)
    frame_dict = {idx: video_reader[idx] for idx in range(224)}
    imgs = [frame_dict[idx] for idx in range(224)]
    video_sampled = torch.stack(imgs, 0)
    video_sampled = video_sampled.permute(3, 0, 1, 2)

    video_sampled = video_sampled.unsqueeze(dim=0)
    video = {}
    video['resize'] = video_sampled

    original_video = video['resize'].squeeze(dim=0)[0]  # simply get channel 0
    original_video = original_video.cpu().detach().numpy()

    os.makedirs('./example_cv/', exist_ok=True)
    cv2.imwrite('./example_cv/rank{}.jpg'.format(rank), original_video[:, :, position])

if __name__ == '__main__':
    # print('Function Test Start')
    # flow_estimator = load_flow_estimator()
    #
    # dataset_path = '/home/chenhr/mnt/dataset/em_videos/'  # absolute path to common datasets
    # format_fix = '.mp4'
    #
    # # adaptively choose the device
    # device = "cuda" if torch.cuda.is_available() else "cpu"
    #
    # v_ust = single_video_loader('zbf_s250_264_affine_roi')
    # v_st = single_video_loader('zbf_s250_264_fixed1_constraint2_roi')
    #
    # v_syn = motion_syn(flow_estimator, v_st, v_ust)
    #
    # # write v_syn into disk for check
    # print('Saving results')
    # os.makedirs('./deformation/syn_sequence/', exist_ok=True)
    # for i in range(len(v_syn)):
    #     cv2.imwrite('./deformation/syn_sequence/syn_' + str(i).zfill(3) + '.png', v_syn[i])
    #
    # print('done')


    # Generate and Save in-situ videos
    print('Saving in-situ videos\n')
    # # root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FlyEM/FlyEM_32nm_raw_0626/'
    # # root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FlyEM/AAAI_supplementary/new_area2/'
    root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FAFB_v14/SAVIOR_data/FAFB_break/'
    #
    # # video_out_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/' + 'in-situ_videos/'
    # # video_out_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM_AAAI_supp/new_area2/' + 'in-situ_videos/'
    video_out_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_AAAI_supp/4080/' + 'in-situ_videos/'
    # #
    # #
    os.makedirs(video_out_dir, exist_ok=True)
    selected_area = [0, 0, 256, 256]  # save full sequence
    # for i in range(10, 100):
    # for i in range(0, 25):
    for i in range(0, 100):
        image_folder = root_dir + 'subblock{}/'.format(i)
        save_name = video_out_dir + str(i).zfill(4)
        generate_video_selected_area(image_folder, save_name, selected_area)

    print('Saved')

    # # Generate deformed & ranked videos
    # # Generate deformed ranked videos using FlyEM dataset
    # # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/'
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM_AAAI_supp/new_area1/'
    # # 形变可以在'random affine', 'random tps', 'combined_tps1'三个预设模式下选取
    # generate_and_save_subblocks(root_dir, mode='random tps', rank_num=13, seq_length=224)
    #
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM_AAAI_supp/new_area2/'
    # generate_and_save_subblocks(root_dir, mode='random tps', rank_num=13, seq_length=224)

    root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_AAAI_supp/4080/'
    generate_and_save_subblocks(root_dir, mode='random affine', rank_num=13, seq_length=224)

    # # Generate deformed ranked videos using Lucchi++ dataset
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/'
    # generate_and_save_subblocks(root_dir, format_fix='.avi', mode='combined_tps1', rank_num=13, seq_length=224)

    # for mode in ['rankloss_sf+of', 'CCloss_sf+of']:
    #     ranked_xy_evaluation(mode=mode)
    #
    # print('done')

    # print('Function_Test: Generating Training Data for Q-Align Branch')
    # # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/', '.avi'
    # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/', '.mp4'
    #
    # generate_and_save_cross_view(root_dir, post_fix, 'random_tps', 4, 32)
    #
    # print('Done')

    # # save cross_view examples of different ranks
    # for rank in [0, 3, 6, 9]:
    #     read_and_extract_cross_view_image(rank, 100)











