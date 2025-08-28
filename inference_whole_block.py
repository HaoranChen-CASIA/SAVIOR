"""
    组合已训练好的整个sf+of模型，以及预训练的Q-Align模型
    2条不同方向的支路，完整评测整个体块
"""

# 初期测试
from function_test import *
from q_align import QAlignVideoScorer

from PIL import Image


def load_video(video_file):
    from decord import VideoReader
    vr = VideoReader(video_file)

    # Get video frame rate
    fps = vr.get_avg_fps()

    # Calculate frame indices for 1fps
    frame_indices = [int(fps * i) for i in range(int(len(vr) / fps))]
    frames = vr.get_batch(frame_indices).numpy()
    return [Image.fromarray(frames[i]) for i in range(int(len(vr) / fps))]


def rescale_branch_score(pr, n_mode='zero-score'):
    if n_mode == 'zero-score':
        print("Zero-Score normalize, mean", np.mean(pr), "std", np.std(pr))
        pr = (pr - np.mean(pr)) / np.std(pr)  # zero-score归一化，严格来说不是0~1；将数据变换为均值为 0，标准差为 1 的分布
    elif n_mode == 'min-max':
        print('Min-Max normalize')
        pr = (pr - np.min(pr)) / (np.max(pr) - np.min(pr))  # min-max归一化，归一到[0,1]里
    elif n_mode == 'log&min-max':
        print('Log & min-max normalize')
        pr = pr - np.min(pr) + 1
        pr = np.log(pr)  # 先log
        pr = (pr - np.min(pr)) / (np.max(pr) - np.min(pr))  # 再min-max归一化，归一到[0,1]里
    else:
        print('Unselected Mode!!')

    return pr


def ranked_video_evaluation(loss='Rankloss'):
    print('\n Evaluator mode full branches')
    # Step 1: initialize sf+of section plane scorer
    warnings.filterwarnings('ignore')
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam
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
    scorer = QAlignVideoScorer()

    print('All branches initialized Successfully!')

    # Step 3
    root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/FlyEM/ranked_videos/combined_tps1/', '.mp4'
    # root_dir, post_fix = '/mnt/Ext001/chenhr/dataset/em_videos/Lucchi++/ranked_videos/combined_tps1/', '.mp4'
    print(root_dir)

    labels = []
    std_min = 0.33
    deform_amp = np.arange(0, std_min * 13, std_min)

    for subblock in [f'subblock{i}' for i in range(0, 5, 1)]:  # for FlyEM
    # for subblock in [f'subblock{i}' for i in range(0, 1, 1)]:  # For Lucchi++
        print('\nEvaluating ' + subblock)
        sp_path = root_dir + subblock + '/section_plane/'
        cv_path = root_dir + subblock + '/cross_view/'
        print('Loading and Scoring Videos...')
        sp_score_list = []
        cv_score_list = []
        for video_name in [f'rank{i}' for i in range(0, 13, 1)]:
            # compute branch 1 sf+of score
            sp_video_path = sp_path + video_name + post_fix
            score_sp = single_video_score(sp_video_path, evaluator, opt, show_score=False)
            sp_score_list.append(score_sp)
            # compute branch 2 Q-Align score
            cv_video_path = cv_path + video_name + post_fix
            cv_video_list = [load_video(cv_video_path)]
            score_cv = scorer(cv_video_list).tolist()[0]
            cv_score_list.append(score_cv)
        # Step 3: combine 2 scores
        sp_score_list = rescale_branch_score(sp_score_list, 'min-max')  # Normalize branch 1 score list
        cv_score_list = rescale_branch_score(cv_score_list, 'min-max')  # Normalize branch 1 score list
        final_scores = [0.5 * (sp_score_list[i] + cv_score_list[i]) for i in range(13)]

        plt.plot(np.arange(13), final_scores)
        labels.append(subblock)
        print('\n branch 1 CC:')
        compute_CC(sp_score_list, deform_amp)
        print('\n branch 2 CC:')
        compute_CC(cv_score_list, deform_amp)
        print('\n final score CC:')
        compute_CC(final_scores, deform_amp)

    plt.title('Full Branch Scores')
    plt.xlabel('Video clip index (ranks)')
    plt.ylabel('Full-Branch Scores')
    plt.legend(labels=labels)
    plt.grid()
    plt.show()


if __name__ == '__main__':
    ranked_video_evaluation(loss='Rankloss')

    # todo: CCloss下是形变越严重分数越高，这里和Q-Align组合时需要重新考虑一下
    # todo: CCloss下直接取负数值如何？
    # ranked_video_evaluation(loss='CCloss')

