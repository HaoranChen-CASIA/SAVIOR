import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '3'  # avoid traffic jam
# hf_cache_dir = '/opt/data/3_107/chenhr_remount/.cache'  # 3.91
hf_cache_dir = '/mnt/Ext001/chenhr/mira_datacenter/.cache/huggingface/hub/'  # 3.1
os.environ['HF_HOME'] = hf_cache_dir
os.environ['HUGGINGFACE_HUB_CACHE'] = hf_cache_dir

from inference_whole_block_v2 import *
import torch.nn.functional as F
import torch.optim as optim

import random

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()


def plcc_loss(y_pred, y):
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()


def generate_train_val_test_list_CWM(subblock_num=10, shuffle=False):
    """
        基于最初设计的合成实验数据（FlyEM， 10blocks，3种形变）构建数据集
    """
    distortions = ['combined_tps1', 'random_affine', 'random_tps']

    # 存储所有子文件夹路径
    folder_list = []

    for dist in distortions:
        for sb_id in range(subblock_num):
            folder_path = f"/{dist}/subblock{sb_id}/"
            folder_list.append(folder_path)

    if shuffle:
        random.shuffle(folder_list)

    # NO NEED TO CREATE ADDITIONAL TEST SET
    train_name_list = folder_list

    return train_name_list


class Gating(nn.Module):
    """
        根据输入特征直接决定分支权重，无需计算余弦相似度，参考
        https://medium.com/@prateeksikdar/understanding-mixture-of-experts-building-a-moe-model-with-pytorch-dd373d9db81c
    """
    def __init__(self, input_dim,
                 num_experts, dropout_rate=0.1):
        super(Gating, self).__init__()

        # Layers
        self.layer1 = nn.Linear(input_dim, 128)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.layer2 = nn.Linear(128, 256)
        self.leaky_relu1 = nn.LeakyReLU()
        self.dropout2 = nn.Dropout(dropout_rate)

        self.layer3 = nn.Linear(256, 128)
        self.leaky_relu2 = nn.LeakyReLU()
        self.dropout3 = nn.Dropout(dropout_rate)

        self.layer4 = nn.Linear(128, num_experts)

    def forward(self, x):
        x = torch.relu(self.layer1(x))
        x = self.dropout1(x)

        x = self.layer2(x)
        x = self.leaky_relu1(x)
        x = self.dropout2(x)

        x = self.layer3(x)
        x = self.leaky_relu2(x)
        x = self.dropout3(x)

        return torch.softmax(self.layer4(x), dim=1)


def serial_ranking_loss_4rank(y_pred, device, margin=10):
    y_pred = y_pred.squeeze(dim=1)
    # ranking score (0,0,0,1,1,2)
    y_pred_1 = torch.cat((y_pred[0].reshape(1), y_pred[0].reshape(1), y_pred[0].reshape(1),
                          y_pred[1].reshape(1), y_pred[1].reshape(1), y_pred[2].reshape(1)), dim=0)
    # ranking score (1,2,3,2,3,3)
    y_pred_2 = torch.cat((y_pred[1:],
                          y_pred[2].reshape(1), y_pred[3].reshape(1), y_pred[3].reshape(1)), dim=0)
    l12 = torch.ones([6]).to(device)  # ranked indicator
    ranking_loss = pairwise_ranking_hinge_loss(y_pred_1, y_pred_2, l12, margin=margin)
    return ranking_loss


def train_process_Hinge(margin=10, model_name='GatingList', rank_num=4, num_epochs=1):
    print('Training Gating-Weight Module\n')

    # 初始化模型、损失函数和优化器
    model = Gating(input_dim=24576+512+32000, num_experts=3)
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 暂时先用默认优化器

    # 读取训练与测试视频名称列表
    model_name = model_name + 'rank{}'.format(rank_num)
    margin = margin
    data_root_dir = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/FlyEM/ranked_videos/'
    train_name_list = generate_train_val_test_list_CWM(subblock_num=10, shuffle=True)

    # 训练模型
    train_Gating_model_3branch_HingeLoss(model, optimizer, data_root_dir, train_name_list,
                               rank_num=rank_num, num_epochs=num_epochs,
                               model_name=model_name, margin=margin)


def train_process_CCloss(model_name='GatingList', rank_num=4, num_epochs=1):
    print('Training Gating-Weight Module using Lplcc+0.3Lrank\n')

    # 初始化模型、损失函数和优化器
    model = Gating(input_dim=24576+512+32000, num_experts=3)
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)  # 暂时先用默认优化器

    # 读取训练与测试视频名称列表
    model_name = model_name + 'rank{}'.format(rank_num)
    print('Model name is {}, total training epoch {}'.format(model_name, num_epochs))
    # margin = margin
    data_root_dir = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/FlyEM/ranked_videos/'
    train_name_list = generate_train_val_test_list_CWM(subblock_num=10, shuffle=True)

    # 训练模型
    train_Gating_model_3branch_CCloss(model, optimizer, data_root_dir, train_name_list,
                                      rank_num=rank_num, num_epochs=num_epochs, model_name=model_name)


def train_Gating_model_3branch_HingeLoss(model, optimizer, data_root_dir, train_name_list,
                                         rank_num=4, num_epochs=1, model_name=None, margin=10):
    print('\n Loading Pretrained Experts')
    # Step 1: initialize sf section plane scorer
    warnings.filterwarnings('ignore')
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    print('Branch sf Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    opt['model']['args']['feat_type'] = 'sf'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_sf = load_model(opt, device='cuda:1')

    # Step 2: initialize Q-Align as cross view evaluator
    scorer = finetuned_qalign_mlp(device='cuda:1')

    # Step 3: initialize SAM_of_affinity branch
    with open('./options/of_SAMaffinity_3dot91.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)
    print('Branch SAM_of_Affinity Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    opt['model']['args']['feat_type'] = 'of_with_SAM_affinity'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_of = load_model(opt, device='cuda:0')

    print('All experts initialized Successfully!')

    for epoch in range(num_epochs):
        model.train()
        loss_in_epoch = []
        for i, name in enumerate(tqdm(train_name_list, desc="Training"), 0):
            # 将梯度缓存清零
            optimizer.zero_grad()
            sf_score_list = []
            of_score_list = []
            cv_score_list = []
            weights_list = []

            for rank in sorted(random.sample(range(13), rank_num)):
                sp_video_path = data_root_dir + name + '/section_plane/rank{}.mp4'.format(rank)
                score_sf, F_sf = single_video_score(sp_video_path, evaluator_sf, opt, show_score=False,
                                                    return_feature=True, device='cuda:1')
                score_of, F_of = single_video_score(sp_video_path, evaluator_of, opt, show_score=False,
                                                    return_feature=True, device='cuda:0')
                F_sf = torch.cat(F_sf, 1)  # 4个clip依次在dim1进行级联，tensor(4,25088-512)
                sf_score_list.append(score_sf)
                F_of = torch.cat(F_of, 1)  # 4个clip依次在dim1进行级联，tensor(4,512)
                of_score_list.append(score_of)

                cv_video_path = data_root_dir + name + '/cross_view/rank{}.mp4'.format(rank)
                cv_video_list = [load_video(cv_video_path)]
                score_cv, F_cv = scorer(cv_video_list, return_feature=True)
                # Fcv为一个tensor(1,32000)
                score_cv = score_cv.tolist()[0]
                cv_score_list.append(score_cv)

                X = torch.cat([F_sf.mean(dim=0, keepdim=True), F_of.mean(dim=0, keepdim=True).to('cuda:1'), F_cv], dim=-1).to("cpu")

                weights = model(X)
                weights_list.append(weights)

            # Step 3: normalize 2 scores
            sf_scores = rescale_branch_score(sf_score_list, 'min-max')  # Normalize branch sp score list
            of_scores = rescale_branch_score(of_score_list, 'min-max')  # Normalize branch sp score list
            cv_scores = rescale_branch_score(cv_score_list, 'log&min-max')  # Normalize branch cv score list
            final_scores = [sf_scores[r]*weights_list[r][0, 0] + of_scores[r]*weights_list[r][0, 1] +
                            cv_scores[r]*weights_list[r][0, 2] for r in range(rank_num)]
            print(weights_list)

            scores = torch.stack(final_scores).view(-1, 1)
            # TODO：下面这部分loss考虑替换成Lplcc+λlrank？这部分替换一下看看
            loss = serial_ranking_loss_4rank(scores, device="cpu", margin=margin)

            loss_in_epoch.append(loss.item())  # 记录当前loss
            # 反向传播和优化
            loss.backward()
            optimizer.step()
        model.eval()

    # todo: draw the loss curve in this epoch
    plot_ranking_loss_in_epoch(loss_in_epoch, label='Gating Training Loss')

    # 保存模型权重
    os.makedirs('./GatingWeight/', exist_ok=True)
    with open('./GatingWeight/{}_loss.pkl'.format(model_name), 'wb') as f:
        pickle.dump(loss_in_epoch, f)
    if model_name is None:
        torch.save(model.state_dict(), './GatingWeight/vanilla.pth')
        print('Model weights saved to vanilla.pth')
    else:
        torch.save(model.state_dict(), './GatingWeight/{}_margin{}.pth'.format(model_name, margin))
        print('Model weights saved to {}_margin{}.pth'.format(model_name, margin))


def train_Gating_model_3branch_CCloss(model, optimizer, data_root_dir, train_name_list,
                                      rank_num=4, full_rank=13, num_epochs=1, model_name=None):
    device_sfQ = 'cuda:1'
    device_ofSAM = 'cuda:0'

    print('\n Loading Pretrained Experts')
    # Step 1: initialize sf section plane scorer
    warnings.filterwarnings('ignore')
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)

    print('Branch sf Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    opt['model']['args']['feat_type'] = 'sf'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_sf = load_model(opt, device=device_sfQ)

    # Step 2: initialize Q-Align as cross view evaluator
    scorer = finetuned_qalign_mlp(device=device_sfQ)

    # Step 3: initialize SAM_of_affinity branch
    with open('./options/of_SAMaffinity_3dot91.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    print('Branch SAM_of_Affinity Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    opt['model']['args']['feat_type'] = 'of_with_SAM_affinity'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_of = load_model(opt, device=device_ofSAM)

    print('All experts initialized Successfully!')
    print('Training Gating Module using Lplcc+λLrank')

    # 保存模型权重与所有损失变化信息
    saving_dir = './GatingWeight_CCloss/'
    os.makedirs(saving_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        loss_in_epoch = []
        for i, name in enumerate(tqdm(train_name_list, desc="Training"), 0):
            # 将梯度缓存清零
            optimizer.zero_grad()
            sf_score_list = []
            of_score_list = []
            cv_score_list = []
            weights_list = []

            rank_labels = []

            for rank in sorted(random.sample(range(13), rank_num)):
                rank_labels.append([1. - rank/full_rank])
                sp_video_path = data_root_dir + name + '/section_plane/rank{}.mp4'.format(rank)
                score_sf, F_sf = single_video_score(sp_video_path, evaluator_sf, opt, show_score=False,
                                                    return_feature=True, device=device_sfQ)
                score_of, F_of = single_video_score(sp_video_path, evaluator_of, opt, show_score=False,
                                                    return_feature=True, device=device_ofSAM)
                F_sf = torch.cat(F_sf, 1)  # 4个clip依次在dim1进行级联，tensor(4,25088-512)
                sf_score_list.append(score_sf)
                F_of = torch.cat(F_of, 1)  # 4个clip依次在dim1进行级联，tensor(4,512)
                of_score_list.append(score_of)

                cv_video_path = data_root_dir + name + '/cross_view/rank{}.mp4'.format(rank)
                cv_video_list = [load_video(cv_video_path)]
                score_cv, F_cv = scorer(cv_video_list, return_feature=True)
                # Fcv为一个tensor(1,32000)
                score_cv = score_cv.tolist()[0]
                cv_score_list.append(score_cv)

                X = torch.cat([F_sf.mean(dim=0, keepdim=True),
                               F_of.mean(dim=0, keepdim=True).to(device_sfQ), F_cv], dim=-1).to("cpu")

                weights = model(X)
                weights_list.append(weights)

            # Step 3: normalize 2 scores
            sf_scores = rescale_branch_score(sf_score_list, 'min-max')  # Normalize branch sp score list
            of_scores = rescale_branch_score(of_score_list, 'min-max')  # Normalize branch sp score list
            cv_scores = rescale_branch_score(cv_score_list, 'log&min-max')  # Normalize branch cv score list
            final_scores = [sf_scores[r]*weights_list[r][0, 0] + of_scores[r]*weights_list[r][0, 1] +
                            cv_scores[r]*weights_list[r][0, 2] for r in range(rank_num)]
            print(weights_list)

            scores = torch.stack(final_scores).view(-1, 1)
            # TODO：下面这部分loss考虑替换成Lplcc+λlrank？这部分替换一下看看
            # TODO: 这部分的label分数可能还得调整一下，变更为1/13倒序？
            y_labels = torch.tensor(rank_labels).to(dtype=torch.float32)  # rank越低应该分数越高
            loss = plcc_loss(scores, y_labels) + 0.3*rank_loss(scores, y_labels)

            loss_in_epoch.append(loss.item())  # 记录当前loss
            # 反向传播和优化
            loss.backward()
            optimizer.step()

        model.eval()

        # todo: draw the loss curve in this epoch
        # plot_ranking_loss_in_epoch(loss_in_epoch, label=f'Gating Training Loss epoch{epoch}')

        # 保存所有损失变化信息
        with open(saving_dir + '{}_loss_epoch{}.pkl'.format(model_name, epoch), 'wb') as f:
            pickle.dump(loss_in_epoch, f)

    if model_name is None:
        torch.save(model.state_dict(), saving_dir + 'vanilla.pth')
        print('Model weights saved to vanilla.pth')
    else:
        torch.save(model.state_dict(), saving_dir + '{}.pth'.format(model_name))
        print('Model weights saved to {}.pth'.format(model_name))


def ranked_video_evaluation_GatingList_v2(root_dir, post_fix, block_num,
                                          gating_weight='Gating_CClossrank13.pth', show_plot=True, show_individual=False):
    print('\n Evaluator mode full branches')
    device_sfQ = 'cuda:1'
    device_ofSAM = 'cuda:0'

    print('\n Loading Pretrained Experts')
    # Step 1: initialize sf section plane scorer
    warnings.filterwarnings('ignore')
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    print('Branch sf Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    # weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    opt['model']['args']['feat_type'] = 'sf'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_sf = load_model(opt, device=device_sfQ)

    # Step 2: initialize Q-Align as cross view evaluator
    print('Branch CV Ranking loss\n')
    scorer = finetuned_qalign_mlp(device=device_sfQ)

    # Step 3: initialize SAM_of_affinity branch
    # with open('./options/of_SAMaffinity_3dot91.yml', "r") as f:
    #     opt = yaml.safe_load(f)
    with open('./options/of_SAMaffinity_3dot1.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    print('Branch SAM_of_Affinity Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    # weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    opt['model']['args']['feat_type'] = 'of_with_SAM_affinity'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_of = load_model(opt, device=device_ofSAM)

    print('All experts initialized Successfully!')
    print('Infer with Gating Module trained by Lplcc+λLrank')

    GatingModule = Gating(input_dim=24576+512+32000, num_experts=3)
    # GatingModule.load_state_dict(torch.load(f'/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/'
    #                                         f'GatingWeight_CCloss/{gating_weight}'))  # 3.1
    GatingModule.load_state_dict(torch.load(f'/opt/data/3_107/chenhr_remount/python_codes/VRQA/'
                                            f'GatingWeight_CCloss//{gating_weight}'))  # 3.91
    GatingModule.eval()
    print(f'Gating-Weighted Module, CC loss 13 ranks, loaded from {gating_weight}!')

    print(root_dir)
    labels = []
    std_min = 0.33
    deform_amp = np.arange(0, std_min * 13, std_min)

    # TODO：2024.0808统计三种CC的均值和标准差
    plcc_lst, srcc_lst, krcc_lst = [], [], []

    with torch.no_grad():
        for subblock in [f'subblock{i}' for i in range(0, block_num, 1)]:  # For any other new datasets
            print('\nEvaluating ' + subblock)
            sp_path = root_dir + subblock + '/section_plane/'
            cv_path = root_dir + subblock + '/cross_view/'
            print('Loading and Scoring Videos...')
            sf_score_list = []
            of_score_list = []
            cv_score_list = []
            weights_list = []

            for video_name in [f'rank{i}' for i in range(0, 13, 1)]:
                # compute branch 1 sf, then of score
                sp_video_path = sp_path + video_name + post_fix
                score_sf, F_sf = single_video_score(sp_video_path, evaluator_sf, opt, show_score=False,
                                                    return_feature=True, device=device_sfQ)
                score_of, F_of = single_video_score(sp_video_path, evaluator_of, opt, show_score=False,
                                                    return_feature=True, device=device_ofSAM)
                F_sf = torch.cat(F_sf, 1)  # 4个clip依次在dim1进行级联，tensor(4,25088-512)
                sf_score_list.append(score_sf)
                F_of = torch.cat(F_of, 1)  # 4个clip依次在dim1进行级联，tensor(4,512)
                of_score_list.append(score_of)

                # compute branch 2 Q-Align score
                cv_video_path = cv_path + video_name + post_fix
                cv_video_list = [load_video(cv_video_path)]
                score_cv, F_cv = scorer(cv_video_list, return_feature=True)
                # Fcv为一个tensor(1,32000)
                score_cv = score_cv.tolist()[0]
                cv_score_list.append(score_cv)

                # 根据特征计算分支权重后组合
                X = torch.cat([F_sf.mean(dim=0, keepdim=True),
                               F_of.mean(dim=0, keepdim=True).to(device_sfQ), F_cv], dim=-1).to("cpu")

                weights = GatingModule(X)
                weights_list.append(weights)

            # Step 3: normalize 2 scores
            sf_scores = rescale_branch_score(sf_score_list, 'min-max')  # Normalize branch sp score list
            of_scores = rescale_branch_score(of_score_list, 'min-max')  # Normalize branch sp score list
            cv_scores = rescale_branch_score(cv_score_list, 'log&min-max')  # Normalize branch cv score list
            final_scores = [sf_scores[r] * weights_list[r][0, 0].item() + of_scores[r] * weights_list[r][0, 1].item() +
                            cv_scores[r] * weights_list[r][0, 2].item() for r in range(13)]

            plt.plot(np.arange(13), final_scores)
            labels.append(subblock)

            if show_individual:
                print('\n Branch Sf CC:')
                compute_CC(sf_scores, deform_amp)
                print('\n Branch Of&SAM_Affinity CC:')
                compute_CC(of_scores, deform_amp)
                print('\n Branch CV CC:')
                compute_CC(cv_scores, deform_amp)
                print('\n Final Score CC:')

            plcc, srcc, krcc = compute_CC(final_scores, deform_amp)
            plcc_lst.append(plcc)
            srcc_lst.append(srcc)
            krcc_lst.append(krcc)

    print('\n Statistical Results on these sub-blocks:')
    print('plcc, srocc, krocc: {:.3f}±{:.3f}， {:.3f}±{:.3f}， {:.3f}±{:.3f}'.format(
        np.mean(plcc_lst), np.std(plcc_lst), np.mean(srcc_lst), np.std(srcc_lst), np.mean(krcc_lst), np.std(krcc_lst)
    ))


    if show_plot:
        plt.title('SAVIOR Scores')
        plt.xlabel('Video clip index (ranks)')
        plt.ylabel('Full-Branch Scores')
        plt.legend(labels=labels)
        plt.grid()
        plt.show()


def ranked_video_evaluation_Average(root_dir, post_fix, block_num, show_plot=True):
    """
        0818对比实验，比较Gating vs Average（各1/3)
    """
    print('\n Evaluator mode full branches')
    device_sfQ = 'cuda:1'
    device_ofSAM = 'cuda:0'

    print('\n Loading Pretrained Experts')
    # Step 1: initialize sf section plane scorer
    warnings.filterwarnings('ignore')
    with open('./options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    print('Branch sf Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    # weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/vRQA2024_singleF_ranked_val_sf.pth'
    opt['model']['args']['feat_type'] = 'sf'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_sf = load_model(opt, device=device_sfQ)

    # Step 2: initialize Q-Align as cross view evaluator
    print('Branch CV Ranking loss\n')
    scorer = finetuned_qalign_mlp(device=device_sfQ)

    # Step 3: initialize SAM_of_affinity branch
    # with open('./options/of_SAMaffinity_3dot91.yml', "r") as f:
    #     opt = yaml.safe_load(f)
    with open('./options/of_SAMaffinity_3dot1.yml', "r") as f:
        opt = yaml.safe_load(f)
    # print(opt)
    print('Branch SAM_of_Affinity Ranking loss\n')
    weights_path = '/opt/data/3_107/chenhr_remount/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    # weights_path = '/mnt/Ext001/chenhr/mira_datacenter/VRQA_weights/SAMmaskaffinity_v1_ranked.pth'
    opt['model']['args']['feat_type'] = 'of_with_SAM_affinity'
    opt['test_load_path'] = weights_path
    # load model
    evaluator_of = load_model(opt, device=device_ofSAM)

    print('All experts initialized Successfully!')
    print('Infer with Average Weights')

    print(root_dir)
    labels = []
    std_min = 0.33
    deform_amp = np.arange(0, std_min * 13, std_min)

    # TODO：2024.0808统计三种CC的均值和标准差
    plcc_lst, srcc_lst, krcc_lst = [], [], []

    with torch.no_grad():
        for subblock in [f'subblock{i}' for i in range(0, block_num, 1)]:  # For any other new datasets
            print('\nEvaluating ' + subblock)
            sp_path = root_dir + subblock + '/section_plane/'
            cv_path = root_dir + subblock + '/cross_view/'
            print('Loading and Scoring Videos...')
            sf_score_list = []
            of_score_list = []
            cv_score_list = []

            for video_name in [f'rank{i}' for i in range(0, 13, 1)]:
                # compute branch 1 sf, then of score
                sp_video_path = sp_path + video_name + post_fix
                score_sf, _ = single_video_score(sp_video_path, evaluator_sf, opt, show_score=False,
                                                    return_feature=True, device=device_sfQ)
                score_of, _ = single_video_score(sp_video_path, evaluator_of, opt, show_score=False,
                                                    return_feature=True, device=device_ofSAM)
                sf_score_list.append(score_sf)
                of_score_list.append(score_of)

                # compute branch 2 Q-Align score
                cv_video_path = cv_path + video_name + post_fix
                cv_video_list = [load_video(cv_video_path)]
                score_cv, _ = scorer(cv_video_list, return_feature=True)
                score_cv = score_cv.tolist()[0]
                cv_score_list.append(score_cv)

            # Step 3: normalize 2 scores
            sf_scores = rescale_branch_score(sf_score_list, 'min-max')  # Normalize branch sp score list
            of_scores = rescale_branch_score(of_score_list, 'min-max')  # Normalize branch sp score list
            cv_scores = rescale_branch_score(cv_score_list, 'log&min-max')  # Normalize branch cv score list
            final_scores = [sf_scores[r] * 1/3 + of_scores[r] * 1/3 +
                            cv_scores[r] * 1/3 for r in range(13)]

            plt.plot(np.arange(13), final_scores)
            labels.append(subblock)
            print('\n Branch Sf CC:')
            compute_CC(sf_scores, deform_amp)
            print('\n Branch Of&SAM_Affinity CC:')
            compute_CC(of_scores, deform_amp)
            print('\n Branch CV CC:')
            compute_CC(cv_scores, deform_amp)
            print('\n Final Score CC:')
            plcc, srcc, krcc = compute_CC(final_scores, deform_amp)
            plcc_lst.append(plcc)
            srcc_lst.append(srcc)
            krcc_lst.append(krcc)

    print('\n Statistical Results on these sub-blocks:')
    print('plcc, srocc, krocc: {:.3f}±{:.3f}， {:.3f}±{:.3f}， {:.3f}±{:.3f}'.format(
        np.mean(plcc_lst), np.std(plcc_lst), np.mean(srcc_lst), np.std(srcc_lst), np.mean(krcc_lst), np.std(krcc_lst)
    ))

    if show_plot:
        plt.title('SAVIOR Scores')
        plt.xlabel('Video clip index (ranks)')
        plt.ylabel('Full-Branch Scores')
        plt.legend(labels=labels)
        plt.grid()
        plt.show()


if __name__ == "__main__":
    # train_process_Hinge(margin=10, model_name='Gating_ListV2', rank_num=4)

    # os.environ['CUDA_VISIBLE_DEVICES'] = "1,2"  # 3.91设置
    # train_process_CCloss(model_name='Gating_CCloss_v0725', rank_num=13, num_epochs=3)


    weight_name = 'Gating_CCloss_v0725rank13.pth'

    # root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM_AAAI_supp/new_area1/ranked_videos/combined_tps1/', '.mp4', 25
    # root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM_AAAI_supp/new_area2/ranked_videos/combined_tps1/', '.mp4', 25
    root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
                                    'Lucchi++/ranked_videos/combined_tps1/', '.mp4', 1
    #
    # ranked_video_evaluation_GatingList(root_dir, post_fix, block_num, loss='Rankloss')

    # Ablation Data
    # root_dir, post_fix, block_num = '/mnt/Ext001/chenhr/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM/ranked_videos/combined_tps1/', '.mp4', 10
    # root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM/ranked_videos/combined_tps1/', '.mp4', 10


    # Extend experiments
    # New Areas related
    # root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM_AAAI_supp/new_area1/ranked_videos/random affine/', '.mp4', 25

    # root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
    #                                 'FlyEM_AAAI_supp/new_area2/ranked_videos/random affine/', '.mp4', 25

    ranked_video_evaluation_GatingList_v2(root_dir, post_fix, block_num,
                                          gating_weight=weight_name, show_individual=False)
    # ranked_video_evaluation_Average(root_dir, post_fix, block_num)

    # Old Area related
    for distortion in ['random_affine', 'random_tps']:
        root_dir, post_fix, block_num = '/opt/data/3_107/chenhr_remount/mira_datacenter/dataset/em_videos/' \
                                        'FlyEM/ranked_videos/{}/'.format(distortion), '.mp4', 10
        ranked_video_evaluation_GatingList_v2(root_dir, post_fix, block_num,
                                              gating_weight=weight_name, show_individual=False)


