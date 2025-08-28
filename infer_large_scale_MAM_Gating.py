import os
# hf_cache_dir = '/opt/data/3_107/chenhr_remount/.cache'  # 3.91
hf_cache_dir = '/mnt/Ext001/chenhr/mira_datacenter/.cache/huggingface/hub/'  # 3.1
os.environ['HF_HOME'] = hf_cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = hf_cache_dir

from train_Gating_and_eval import Gating
from inference_whole_block_v2 import *
from visualize_plot import *
import time


def initialize_models_MAM_Gating(device_sfQ='cuda:1', device_ofSAM='cuda:0',
                                 gating_weight='Gating_CCloss_v0725rank13.pth'):
    print('\n Evaluator mode full branches')

    print('\n Loading Pretrained Experts')
    # Step 1: initialize sf section plane scorer
    warnings.filterwarnings('ignore')
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt_sf = yaml.safe_load(f)
    # print(opt)
    print('Branch sf Ranking loss\n')
    # weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    opt_sf['model']['args']['feat_type'] = 'sf'
    opt_sf['test_load_path'] = weights_path
    # load model
    evaluator_sf = load_model(opt_sf, device=device_sfQ)

    # Step 2: initialize Q-Align as cross view evaluator
    print('Branch CV Ranking loss\n')
    scorer = finetuned_qalign_mlp(device=device_sfQ, mlp_pth='./em_weights/cnn_model.pth')

    # Step 3: initialize SAM_of_affinity branch
    # with open('./options/of_SAMaffinity_3dot91.yml', "r") as f:
    #     opt = yaml.safe_load(f)
    with open('./options/of_SAMaffinity_3dot1.yml', "r") as f:
        opt_of = yaml.safe_load(f)
    # print(opt)
    print('Branch SAM_of_Affinity Ranking loss\n')
    # weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    opt_of['model']['args']['feat_type'] = 'of_with_SAM_affinity'
    opt_of['test_load_path'] = weights_path
    # load model
    evaluator_of = load_model(opt_of, device=device_ofSAM)

    print('All experts initialized Successfully!')

    GatingModule = Gating(input_dim=24576 + 512 + 32000, num_experts=3)
    GatingModule.load_state_dict(torch.load(f'/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/'
                                            f'GatingWeight_CCloss/{gating_weight}'))  # 3.1
    # GatingModule.load_state_dict(torch.load(f'/opt/data/3_107/chenhr_remount/python_codes/VRQA/'
    #                                         f'GatingWeight_CCloss//{gating_weight}'))  # 3.91
    GatingModule.eval()
    print(f'Gating-Weighted Module, CC loss 13 ranks, loaded from {gating_weight}!')

    return evaluator_sf, opt_sf, evaluator_of, opt_of, scorer, GatingModule


def change_score_sf_of(sp_score_raw, sp_min, sp_max):
    re_sp = (sp_score_raw - sp_min) / (sp_max - sp_min)
    return re_sp


def change_score_cv(cv_score_raw):
    cv_min, cv_max = -63.468, 111.018  # from area 1 and 2
    tmp_cv = cv_score_raw - cv_min + 1
    tmp_max = cv_max - cv_min + 1
    log_cv = np.log(tmp_cv)  # 先log
    log_max = np.log(tmp_max)
    re_cv = log_cv / log_max
    return re_cv


class Assess_Large_Scale_Data_MAM_Gating:
    def __init__(self, vol_size=224, eval=True, device_sfQ='cuda:1', device_ofSAM='cuda:0'):
        # set sub-volume size
        self.vol_size = vol_size
        self.device_sfQ = device_sfQ
        self.device_ofSAM = device_ofSAM
        # initialize all evaluators, sf, MAM_of, QAlign
        if eval:
            print('Eval mode on!\n')
            self.evaluator_sf, self.opt_sf, self.evaluator_of, \
            self.opt_of, self.scorer, self.GatingModule = initialize_models_MAM_Gating()

    def infer_videos_from_single_volume(self, root_dir, cv_direction='cv_yz', post_fix='.mp4'):
        """
            Scoring function changed from infernce_whole_block_v2.py
            Only consider single cross-view this time?
        """
        # print('Root dir is: ' + root_dir)
        sp_video_path = root_dir + 'sp_xy' + post_fix
        score_sf, F_sf = single_video_score(sp_video_path, self.evaluator_sf, self.opt_sf, show_score=False,
                                            return_feature=True, device=self.device_sfQ)
        score_of, F_of = single_video_score(sp_video_path, self.evaluator_of, self.opt_of, show_score=False,
                                            return_feature=True, device=self.device_ofSAM)
        F_sf = torch.cat(F_sf, 1)  # 4个clip依次在dim1进行级联，tensor(4,25088-512)
        F_of = torch.cat(F_of, 1)  # 4个clip依次在dim1进行级联，tensor(4,512)

        cv_video_path = root_dir + cv_direction + post_fix
        cv_video_list = [load_video(cv_video_path)]
        score_cv, F_cv = self.scorer(cv_video_list, return_feature=True)
        # Fcv为一个tensor(1,32000)
        score_cv = score_cv.tolist()[0]

        # 根据特征计算分支权重后组合
        X = torch.cat([F_sf.mean(dim=0, keepdim=True),
                       F_of.mean(dim=0, keepdim=True).to(self.device_sfQ), F_cv], dim=-1).to("cpu")

        weights = self.GatingModule(X)

        re_sf = change_score_sf_of(score_sf, sp_min=-42.14, sp_max=80.245)  # data from area 1 and 2
        re_of = change_score_sf_of(score_of, sp_min=-324.43, sp_max=36.644)  # data collected from new area 2
        re_cv = change_score_cv(score_cv)

        final_score = re_sf * weights[0, 0].item() + re_of * weights[0, 1].item() + re_cv * weights[0, 2].item()

        return max(final_score, 0)


def compare_FlyWire_and_FAFBv14():
    ALSD = Assess_Large_Scale_Data_MAM_Gating(vol_size=224, eval=True)

    video_root_dir = '/mnt/Ext001/chenhr/mira_datacenter/dataset/em_videos/SAVIOR_250825/'

    for data_name in ['FlyWire/32_32_40nm/area1/', 'FAFBv14/32_32_40nm/area1/']:
        print('Evaluating Data {}'.format(data_name))
        video_save_path = video_root_dir + data_name
        t_s = time.perf_counter()
        final_heatmap = np.zeros((5, 5))

        for h in range(5):
            for w in range(5):
                print('Scoring block h{} w{}'.format(h, w))
                block_video_dir = video_save_path + 'b_d0_h{}_w{}/'.format(h, w)
                final = ALSD.infer_videos_from_single_volume(block_video_dir, 'cv_yz')
                final_heatmap[h, w] = final
        t_e = time.perf_counter()
        print('Inference time for 25 blocks: {:.3f}s\n'.format(t_e - t_s))
        draw_heatmaps_with_scores(final_heatmap, 0, video_save_path, True, True)


if __name__ == '__main__':
    # compare_FlyWire_and_FAFBv14()

    ALSD = Assess_Large_Scale_Data_MAM_Gating(vol_size=224, eval=True)
    # # FAFB area 5, 5*5 = 25 blocks test, old version without depth
    # # TODO: 路径修改为3.91对应路径
    # # raw_root_dir = '/mnt/Ext001/chenhr/dataset/em_images/FAFB_v14/SAVIOR_data/area5/'
    # # video_save_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/FAFB_data/SAVIOR_data/area5/'
    #
    # TODO: Try for Minnie65 data, extend experiments for qualitative study
    # video_save_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/Minnie65/mip2_250820/area1/'
    video_save_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/Minnie65/mip2_250820/area2/'
    #
    # import time
    t_s = time.perf_counter()
    final_heatmap = np.zeros((5, 5))
    # # TODO: for Hemibrain part
    # # for h in range(5):
    # #     for w in range(5):
    # #         print('Scoring block h{} w{}'.format(h, w))
    # #         block_video_dir = video_save_path + 'b_h{}_w{}/'.format(h, w)
    # #         final = ALSD.infer_videos_from_single_volume(block_video_dir, 'cv_yz')
    # #         final_heatmap[h, w] = final
    # # t_e = time.perf_counter()
    #
    # # TODO: for Minnie part
    for h in range(5):
        for w in range(5):
            print('Scoring block h{} w{}'.format(h, w))
            block_video_dir = video_save_path + 'b_d0_h{}_w{}/'.format(h, w)
            final = ALSD.infer_videos_from_single_volume(block_video_dir, 'cv_yz')
            final_heatmap[h, w] = final
    t_e = time.perf_counter()
    print('Inference time for 25 blocks: {:.3f}s'.format(t_e - t_s))
    draw_heatmaps_with_scores(final_heatmap, 0, video_save_path, True, True)





