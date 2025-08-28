import matplotlib.pyplot as plt

from evaluate_single_video import *
from new_train_ranked import *
from deformation.linear_deform import *


def compute_CC(scores_n, deform_amp, show=True):
    # calculate PLCC and p-value
    plcc, _ = pearsonr(scores_n, deform_amp)
    # calculate SRCC and p-value
    srcc, _ = spearmanr(scores_n, deform_amp)
    # calculate KRCC and p-value
    krcc, _ = kendallr(scores_n, deform_amp)
    if show:
        # print results
        print("PLCC: {:.3f}".format(plcc))
        print("SRCC: {:.3f}".format(srcc))
        print("KRCC: : {:.3f}".format(krcc))

    return plcc, srcc, krcc


def compute_rank2(scores_n_list):
    "对NTIRE2024中rank2指标的复现，计算non-homologous ranking的准确性"


def plot_and_compute_CC(raw_scores, mode, rank_num):
    # step 1: normalize raw scores
    mean, std = raw_scores.mean(), raw_scores.std()
    scores_n = (raw_scores - mean) / std

    std_min = 0.33
    deform_amp = np.arange(0, std_min * rank_num, std_min)
    compute_CC(scores_n, deform_amp)

    plt.plot(deform_amp, scores_n, 'ro-')
    plt.title('ranking of random {}'.format(mode))
    plt.xlabel('deformation amplitude')
    plt.ylabel('normalized score')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('ignore')

    print('Begin sv-ranked evaluate process. \n')

    # Read In options
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    # adaptively choose the device
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam

    # adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    evaluator = load_model(opt, device)

    # read in video
    # video_path = './VSdataset/01859.mp4'
    dataset_path = '/home/chenhr/mnt/dataset/em_videos/'  # absolute path to common datasets
    format_fix = '.mp4'

    # todo: choose deformation mode here, form 'affine' to 'random tps'; rank_num=4 for default
    mode, rank_num = 'affine', 13

    for video_name in ['Lucchi_dxy6_z6', 'FlyEM_Neurite', 'FlyEM_OCG1']:
        print(video_name)
        video_path = dataset_path + video_name + format_fix

        video_reader = decord.VideoReader(video_path)
        frame_dict = {idx: video_reader[idx] for idx in range(10, 42, 1)}
        imgs = [frame_dict[idx] for idx in range(10, 42, 1)]
        video_sampled = torch.stack(imgs, 0)
        video_sampled = video_sampled.permute(3, 0, 1, 2)

        video_sampled = video_sampled.unsqueeze(dim=0)

        # todo: create ranked dataset as input
        video = {}
        video['resize'] = video_sampled
        print('Generating ranking data')

        # select deform method
        if mode == 'random tps':
            # todo: tps test
            # mode, rank_num = 'random tps', 13
            print('deform mode {}, rank num {}'.format(mode, rank_num))
            t_s = time.perf_counter()
            r_v = generate_ranking_video_tps(video, mean=0, std=1, rank_num=rank_num)
            print('deformation time {:.2f}s'.format(time.perf_counter() - t_s))
        else:  # linear deform
            # todo: linear test
            # mode, rank_num = 'affine', 4
            print('deform mode {}, rank num {}'.format(mode, rank_num))
            r_v = generate_ranking_video_linear(video, mean=0, std=1,
                                                mode=mode, rank_num=rank_num)

        video['resize'] = r_v.to(device)

        print('Predicting')
        raw_score = evaluator(video)
        raw_score = raw_score.squeeze(dim=1).cpu().numpy()
        print('scores from rank 0 to {}'.format(rank_num - 1))
        print(raw_score)

        plot_and_compute_CC(raw_score, mode, rank_num)

        # # predict and rescale score
        # single_video_score(video_path, evaluator, opt)

    print('done')

