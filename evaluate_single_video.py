"""
Extract single video inference from new_test.py
"""

from new_test import *
import decord

mean_stds = {
    "Stable_val_pr": (-4.8906377e-09, 1.0),
    "Stable_val_gt": (66.57547008547009, 17.36868385658723),
    "Stable-test_pr": (1.6463687e-08, 1.0),
    "Stable-test_gt": (66.46410912190963, 18.332435202228417)
}


def load_model(opt, device):
    # defining model and loading checkpoint

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)

    state_dict = torch.load(opt["test_load_path"], map_location=device)["state_dict"]

    model.load_state_dict(state_dict, strict=True)

    return model


def single_video_score(video_path, evaluator, opt, show_score=False, return_feature=False, device='cuda'):
    video_reader = decord.VideoReader(video_path)

    # Sample video clips
    vsamples = video_clip_sampler(video_reader, opt, device)

    if return_feature:
        score, F_sp = predict_score(vsamples, evaluator, model='val', show_score=show_score, return_feature=return_feature)
    else:
        score = predict_score(vsamples, evaluator, model='val', show_score=show_score)

    if show_score:
        print(f"The quality score of the video is {score:.5f}.")

    if return_feature:
        return score, F_sp
    else:
        return score


def predict_score(vsamples, evaluator, model='val', show_score=True, return_feature=False):
    if return_feature:
        result, F_sp = evaluator(vsamples, return_Feat=return_feature)
    else:
        result = evaluator(vsamples)
    # score = rescale_with_gt(result.mean().item(), model)
    if show_score:
        print('Score for each v_clip: ', result)
        print('Score Mean and STD before rescale: {:.2f}, {:.3f}'.format(result.mean().item(), result.std().item()))
    score = result.mean().item()
    if return_feature:
        return score, F_sp
    return score


def sigmoid_rescale(score, model="FasterVQA"):  # does this function match with Stable-VQA?
    mean, std = mean_stds[model]
    # TODO: check values here!!
    # mean, std = 0.14759505, 0.03613452
    x = (score - mean) / std
    print(f"Inferring with model [{model}]:")
    score = 1 / (1 + np.exp(-x))
    return score


def rescale_with_gt(score, model='val'):
    pr_mean, pr_std = mean_stds['Stable_' + model + '_pr']
    gt_mean, gt_std = mean_stds['Stable_'+model+'_gt']
    pr = ((score - pr_mean) / pr_std) * gt_std + gt_mean
    return pr


def video_clip_sampler(video_reader, opt, device='cuda'):
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    ## Data Definition
    vsamples = {}
    t_data_opt = opt["data"]["test"]["args"]  # not used here
    s_data_opt = opt["data"]["test"]["args"]["sample_types"]
    for sample_type, sample_args in s_data_opt.items():
        ## Sample Temporally
        # sampler = datasets.SampleFrames(clip_len=sample_args["clip_len"], num_clips=sample_args["num_clips"])
        # TODO: beware of 'frame_interval' and other arguments
        sampler = datasets.SampleFrames(clip_len=sample_args["clip_len"], frame_interval=sample_args["frame_interval"],
                                        num_clips=sample_args["num_clips"])

        num_clips = sample_args.get("num_clips", 1)
        frames = sampler(len(video_reader))
        # print("Sampled frames are", frames)
        frame_dict = {idx: video_reader[idx] for idx in np.unique(frames)}
        imgs = [frame_dict[idx] for idx in frames]
        video = torch.stack(imgs, 0)
        video = video.permute(3, 0, 1, 2)

        ## Sample Spatially
        sampled_video = datasets.get_spatial_fragments(video, **sample_args)

        # TODO: check values here!!
        # normalize video clips
        mean, std = normalize_vclips(sampled_video)
        # mean, std = torch.FloatTensor([123.675, 116.28, 103.53]), torch.FloatTensor([58.395, 57.12, 57.375])  # ???
        sampled_video = ((sampled_video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)

        sampled_video = sampled_video.reshape(sampled_video.shape[0], num_clips, -1,
                                              *sampled_video.shape[2:]).transpose(0, 1)
        vsamples[sample_type] = sampled_video.to(device)
        # print(sampled_video.shape)
    return vsamples


def normalize_vclips(sampled_video):
    c_means, c_stds = [], []
    for channel in range(3):
        channel_mean = sampled_video[channel].mean().item()
        channel_std = sampled_video[channel].std().item()
        c_means.append(channel_mean)
        c_stds.append(channel_std)
    return torch.FloatTensor(c_means), torch.FloatTensor(c_stds)


def for_CBH_data(evaluator, version='pred'):
    print('Evaluating Version ' + version)
    dataset_path = '/home/chenhr/mnt/dataset/em_videos/CBH_datas/{}/'.format(version)  # absolute path

    format_fix = '.mp4'

    video_files = sorted([video for video in os.listdir(dataset_path) if video.endswith(format_fix)])

    for video_name in video_files:
        print(video_name)
        video_path = dataset_path + video_name
        # predict and rescale score
        single_video_score(video_path, evaluator, opt)

    print('done')


if __name__ == '__main__':
    print('Begin single video evaluate process. \n')

    # Read In options
    # with open('./options/stable.yml', "r") as f:
    #     opt = yaml.safe_load(f)
    # print(opt)
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    print(opt['test_load_path'])

    # adaptively choose the device
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # avoid traffic jam

    # adaptively choose the device

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load model
    evaluator = load_model(opt, device)

    # # read in video
    # # video_path = './VSdataset/01859.mp4'
    # dataset_path = '/home/chenhr/mnt/dataset/em_videos/'  # absolute path to common datasets
    # format_fix = '.mp4'
    #
    # for video_name in ['Lucchi_dxy6_z6', 'FlyEM_Neurite', 'FlyEM_OCG1',
    #                    'zbf_s250_264_affine_roi', 'zbf_s250_264_elastic_roi', 'zbf_s250_264_fixed1_constraint2_roi']:
    # # for video_name in ['fromCBH_pred_roi', 'fromCBH_pred_2_roi']:
    #     print(video_name)
    #     video_path = dataset_path + video_name + format_fix
    #     # predict and rescale score
    #     single_video_score(video_path, evaluator, opt)
    #
    # print('done')

    # for version in ['pred', 'pred_2']:
    #     for_CBH_data(evaluator, version=version)



