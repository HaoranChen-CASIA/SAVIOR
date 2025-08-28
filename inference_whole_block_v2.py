"""
    Combine sf+of(ranked) with QAlign&MLP(ranked ft)
"""
# import os
# # os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # avoid traffic jam

import numpy as np

from inference_whole_block import *
from PIL import Image
from typing import List

import time

import torch.nn as nn
from q_align.model.builder import load_pretrained_model
from q_align.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
from q_align.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria


class Simple_MLP(nn.Module):
    # 减少需要存储的参数量，单纯只训练CNN
    def __init__(self, device="cuda"):
        super().__init__()
        self.feat_shape = 32000
        self.mlp_regressor = self.quality_regression(self.feat_shape, 128, 1).to(device)

    def quality_regression(self, in_channels, middle_channels, out_channels):
        regression_block = nn.Sequential(
            nn.Linear(in_channels, middle_channels),
            nn.Linear(middle_channels, out_channels),
        )

        return regression_block

    def forward(self, average_features):
        score = self.mlp_regressor(average_features)
        return score


class finetuned_qalign_mlp(nn.Module):
    def __init__(self, pretrained="q-future/one-align", device="cuda", mlp_pth='./em_weights/cnn_model.pth'):
        super().__init__()
        tokenizer, model, image_processor, _ = load_pretrained_model(pretrained, None, "mplug_owl2", device=device,
                                                                     device_map=0)
        """
        cited from Q_Align official codes
        device_map (`str` or `Dict[str, Union[int, str, torch.device]]` or `int` or `torch.device`, *optional*):
                A map that specifies where each submodule should go. It doesn't need to be refined to each
                parameter/buffer name, once a given module name is inside, every submodule of it will be sent to the
                same device. If we only pass the device (*e.g.*, `"cpu"`, `"cuda:1"`, `"mps"`, or a GPU ordinal rank
                like `1`) on which the model will be allocated, the device map will map the entire model to this
                device. Passing `device_map = 0` means put the whole model on GPU 0.
        """
        prompt = "USER: How would you rate the quality of this video?\n<|image|>\nASSISTANT: The quality of the video is"

        self.preferential_ids_ = [id_[1] for id_ in
                                  tokenizer(["excellent", "good", "fair", "poor", "bad"])["input_ids"]]
        self.weight_tensor = torch.Tensor([1, 0.75, 0.5, 0.25, 0.]).half().to(model.device)

        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(
            0).to(model.device)
        print('Q-Align Feature Extractor ready!\n')

        self.mlp = Simple_MLP(device=device)
        print('Load MLP weights from {}\n'.format(mlp_pth))
        self.mlp.load_state_dict(torch.load(mlp_pth))
        self.mlp.eval()
        print('MLP Model weights loaded!\n')

    def expand2square(self, pil_img, background_color):
        width, height = pil_img.size
        if width == height:
            return pil_img
        elif width > height:
            result = Image.new(pil_img.mode, (width, width), background_color)
            result.paste(pil_img, (0, (width - height) // 2))
            return result
        else:
            result = Image.new(pil_img.mode, (height, height), background_color)
            result.paste(pil_img, ((height - width) // 2, 0))
            return result

    def forward(self, video: List[List[Image.Image]], return_feature=False):
        self.eval()
        video = [
            [self.expand2square(frame, tuple(int(x * 255) for x in self.image_processor.image_mean)) for frame in
             vid]
            for vid in video]
        with torch.no_grad():
            video_tensors = [
                self.image_processor.preprocess(vid, return_tensors="pt")["pixel_values"].half().to(
                    self.model.device)
                for vid in video]
            hidden_states = self.model(self.input_ids.repeat(len(video_tensors), 1),
                                       images=video_tensors)["logits"]
            average_features = torch.mean(hidden_states, dim=0)
            average_features = torch.mean(average_features, dim=0, keepdim=True).type(torch.float32)
            scores = self.mlp(average_features)

        if return_feature:
            return scores.squeeze(dim=1), average_features
        else:
            return scores.squeeze(dim=1)


def initialize_models(loss='Rankloss', mlp_pth='./em_weights/cnn_model.pth'):
    # Step 1: initialize sf+of section plane scorer
    warnings.filterwarnings('ignore')
    # os.environ['CUDA_VISIBLE_DEVICES'] = device_id  # avoid traffic jam
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cuda:" + device_id if torch.cuda.is_available() else "cpu"

    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    if loss == 'Rankloss':
        print('Branch 1 Ranking loss\n')
        weights_path = './em_weights/vRQA2024_singleF_ranked_val_sf+of.pth'
    else:
        print('Branch 1 CC loss\n')
        weights_path = './em_weights/vRQA2024_singleF_CCloss_val_sf+of.pth'
    opt['model']['args']['feat_type'] = 'sf+of'
    opt['test_load_path'] = weights_path
    # load model
    evaluator = load_model(opt, device)

    # Step 2: initialize Q-Align as cross view evaluator
    scorer = finetuned_qalign_mlp(device=device, mlp_pth=mlp_pth)

    print('All branches initialized Successfully!\n')

    return evaluator, scorer, opt


def ranked_video_evaluation_v2(root_dir, post_fix, block_num, loss='Rankloss'):
    print('\n Evaluator mode full branches')
    # Step 1: initialize sf+of section plane scorer
    warnings.filterwarnings('ignore')
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # avoid traffic jam
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    if loss == 'Rankloss':
        print('Branch 1 Ranking loss\n')
        weights_path = './em_weights/vRQA2024_singleF_ranked_val_sf+of.pth'
    else:
        print('Branch 1 CC loss\n')
        weights_path = './em_weights/vRQA2024_singleF_CCloss_val_sf+of.pth'
    opt['model']['args']['feat_type'] = 'sf+of'
    opt['test_load_path'] = weights_path
    # load model
    evaluator = load_model(opt, device)

    # Step 2: initialize Q-Align as cross view evaluator
    scorer = finetuned_qalign_mlp(device=device)

    print('All branches initialized Successfully!')

    # Step 3
    # # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/ranked_videos/combined_tps1/', '.mp4'
    # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/ranked_videos/combined_tps1/', '.mp4'

    print(root_dir)
    labels = []
    std_min = 0.33
    deform_amp = np.arange(0, std_min * 13, std_min)

    # TODO：2024.0808统计三种CC的均值和标准差
    plcc_lst, srcc_lst, krcc_lst = [], [], []

    # TODO: 2024.12.11记录所有raw score方便之后进行归一化
    sp_raw_all, cv_raw_all = [], []

    # for subblock in [f'subblock{i}' for i in range(0, 10, 1)]:  # for FlyEM

    for subblock in [f'subblock{i}' for i in range(0, block_num, 1)]:  # For any other new datasets
        print('\nEvaluating ' + subblock)
        sp_path = root_dir + subblock + '/section_plane/'
        cv_path = root_dir + subblock + '/cross_view/'
        print('Loading and Scoring Videos...')
        sp_score_list = []
        cv_score_list = []
        tsp_lst, tcv_lst = [], []
        for video_name in [f'rank{i}' for i in range(0, 13, 1)]:
            # compute branch 1 sf+of score
            sp_video_path = sp_path + video_name + post_fix
            ts_v = time.perf_counter()
            score_sp = single_video_score(sp_video_path, evaluator, opt, show_score=False)
            te_v = time.perf_counter()
            # print('Infer time for sp branch: {:.3f}s'.format(te_v - ts_v))
            tsp_lst.append(te_v - ts_v)
            sp_score_list.append(score_sp)

            # compute branch 2 Q-Align score
            cv_video_path = cv_path + video_name + post_fix
            ts_c = time.perf_counter()
            cv_video_list = [load_video(cv_video_path)]
            score_cv = scorer(cv_video_list).tolist()[0]
            te_c = time.perf_counter()
            # print('Infer time for cv branch: {:.3f}s'.format(te_c - ts_c))
            tcv_lst.append(te_c - ts_c)
            cv_score_list.append(score_cv)

        # 排除第一个时间最长的，计算后面的
        print('SP branch time: mean {:.3f}, std {:.3f}'.format(np.mean(tsp_lst[1:]), np.std(tsp_lst[1:])))
        print('CV branch time: mean {:.3f}, std {:.3f}'.format(np.mean(tcv_lst[1:]), np.std(tcv_lst[1:])))

        sp_raw_all.append(sp_score_list)
        cv_raw_all.append(cv_score_list)

        # Step 3: combine 2 scores
        sp_score_list = rescale_branch_score(sp_score_list, 'min-max')  # Normalize branch sp score list
        cv_score_list = rescale_branch_score(cv_score_list, 'log&min-max')  # Normalize branch cv score list
        final_scores = [0.5*(sp_score_list[i] + cv_score_list[i]) for i in range(13)]

        plt.plot(np.arange(13), final_scores)
        labels.append(subblock)
        print('\n Branch 1 CC:')
        compute_CC(sp_score_list, deform_amp)
        print('\n Branch 2 CC:')
        compute_CC(cv_score_list, deform_amp)
        print('\n Final Score CC:')
        plcc, srcc, krcc = compute_CC(final_scores, deform_amp)
        plcc_lst.append(plcc)
        srcc_lst.append(srcc)
        krcc_lst.append(krcc)

    plt.title('SAVIOR Scores')
    plt.xlabel('Video clip index (ranks)')
    plt.ylabel('Full-Branch Scores')
    plt.legend(labels=labels)
    plt.grid()
    plt.show()

    print('\n Statistical Results on these sub-blocks:')
    print('plcc, srocc, krocc: {:.3f}±{:.3f}， {:.3f}±{:.3f}， {:.3f}±{:.3f}'.format(
        np.mean(plcc_lst), np.std(plcc_lst), np.mean(srcc_lst), np.std(srcc_lst), np.mean(krcc_lst), np.std(krcc_lst)
    ))

    print('Raw sp scores min, max, mean, std: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        np.min(sp_raw_all), np.max(sp_raw_all), np.mean(sp_raw_all), np.std(sp_raw_all)))
    print('Raw cv scores min, max, mean, std: {:.3f}, {:.3f}, {:.3f}, {:.3f}'.format(
        np.min(cv_raw_all), np.max(cv_raw_all), np.mean(cv_raw_all), np.std(cv_raw_all)))

    # 按行存储
    with open(root_dir + "results.txt", "w") as f:
        # 将每个列表分别作为一行写入文件
        f.write("PLCC: " + " ".join(map(str, plcc_lst)) + "\n")
        f.write("SROCC: " + " ".join(map(str, srcc_lst)) + "\n")
        f.write("KROCC: " + " ".join(map(str, krcc_lst)) + "\n")

    return sp_raw_all, cv_raw_all


def infer_videos_from_single_block(root_dir, evaluator, scorer, opt, post_fix='.mp4'):
    print('Root dir is: ' + root_dir)

    sp_video_path = root_dir + 'sp_xy' + post_fix
    score_sp = single_video_score(sp_video_path, evaluator, opt, show_score=False)

    cv_video_path = root_dir + 'cv_xz' + post_fix
    cv_video_list = [load_video(cv_video_path)]
    score_cv_xz = scorer(cv_video_list).tolist()[0]

    cv_video_path = root_dir + 'cv_yz' + post_fix
    cv_video_list = [load_video(cv_video_path)]
    score_cv_yz = scorer(cv_video_list).tolist()[0]

    print('Raw Scores xy, xz, yz are: {:.3f}, {:.3f}, {:.3f}'.format(score_sp, score_cv_xz, score_cv_yz))

    change_score(score_sp, 0.5*(score_cv_xz + score_cv_yz))  # 先试一下看看？


def change_score(sp_score_raw, cv_score_raw, lamda=0.5):
    """
        normalize scores
        using min-max normalize for sp_score, and log&min-max for cv_score
    """
    sp_min, sp_max = -395.326, 54.632  # from area 1 and 2
    cv_min, cv_max = -63.468, 111.018  # from area 1 and 2

    re_sp = (sp_score_raw - sp_min) / (sp_max - sp_min)

    tmp_cv = cv_score_raw - cv_min + 1
    tmp_max = cv_max - cv_min + 1
    log_cv = np.log(tmp_cv)  # 先log
    log_max = np.log(tmp_max)
    re_cv = log_cv / log_max

    print('min-max normalized SP score: {:.3f}'.format(re_sp))
    print('log&min-max normalized CV score: {:.3f}'.format(re_cv))
    print('Final score: {:.3f}\n'.format(lamda*re_sp + (1-lamda)*re_cv))

    return re_sp, re_cv, max(0, lamda*re_sp + (1-lamda)*re_cv)


if __name__ == '__main__':
    # root_dir, post_fix, block_num = '/mnt/Ext001/chenhr/dataset/em_videos/' \
    #                                 'FlyEM/ranked_videos/combined_tps1/', '.mp4', 10
    # # # root_dir, post_fix, block_num = '/mnt/Ext001/chenhr/dataset/em_videos/' \
    # # #                                 'Lucchi++/ranked_videos/combined_tps1/', '.mp4', 1
    # ranked_video_evaluation_v2(root_dir, post_fix, block_num, loss='Rankloss')
    #
    # 2024.09.23, 补充新的独立数据进行实验
    # new area 1
    root_dir, post_fix, block_num = '/mnt/Ext001/chenhr/dataset/em_videos/' \
                                    'FlyEM_AAAI_supp/new_area1/ranked_videos/combined_tps1/', '.mp4', 10

    ranked_video_evaluation_v2(root_dir, post_fix, block_num, loss='Rankloss')
    #
    # # new area 2
    # root_dir, post_fix, block_num = '/mnt/Ext001/chenhr/dataset/em_videos/' \
    #                                 'FlyEM_AAAI_supp/new_area2/ranked_videos/combined_tps1/', '.mp4', 25
    #
    # ranked_video_evaluation_v2(root_dir, post_fix, block_num, loss='Rankloss')

    # 20241212
    # evaluator, scorer, opt = initialize_models(loss='Rankloss')
    #
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_data/area2/'
    # infer_videos_from_single_block(root_dir, evaluator, scorer, opt)
    #
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_data/area3/'
    # infer_videos_from_single_block(root_dir, evaluator, scorer, opt)
    #
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/FAFB_data/area4/'
    # infer_videos_from_single_block(root_dir, evaluator, scorer, opt)

    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/Microns_data/raw/area1/'
    # infer_videos_from_single_block(root_dir, evaluator, scorer, opt)
    #
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/Microns_data/normalized/area1/'
    # infer_videos_from_single_block(root_dir, evaluator, scorer, opt)


