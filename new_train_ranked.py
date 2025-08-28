from new_train import *
from torch.nn import MarginRankingLoss
from deformation.random_tps import *
from deformation.linear_deform import *

from visualize_plot import plot_ranking_loss_in_epoch


def pairwise_ranking_hinge_loss(y_pred1, y_pred2, l12, margin=10):
    """
    pairwise ranking hinge loss from rankIQA
    using pytorch implementation
    """
    loss = MarginRankingLoss(margin=margin, reduction='mean')
    return loss(y_pred1, y_pred2, l12)


def serial_ranking_loss(y_pred, device):
    y_pred = y_pred.squeeze(dim=1)
    # ranking score (0,0,0,1,1,2)
    y_pred_1 = torch.cat((y_pred[0].reshape(1), y_pred[0].reshape(1), y_pred[0].reshape(1),
                          y_pred[1].reshape(1), y_pred[1].reshape(1), y_pred[2].reshape(1)), dim=0)
    # ranking score (1,2,3,2,3,3)
    y_pred_2 = torch.cat((y_pred[1:],
                          y_pred[2].reshape(1), y_pred[3].reshape(1), y_pred[3].reshape(1)), dim=0)
    l12 = torch.ones([6]).to(device)  # ranked indicator
    ranking_loss = pairwise_ranking_hinge_loss(y_pred_1, y_pred_2, l12)
    return ranking_loss


def finetune_epoch(ft_loader, model, model_ema, optimizer, scheduler, device, epoch=-1,
                   need_feat=False):
    # suppose batch_size = 1
    model.train()
    loss_in_epoch = []
    for i, datas in enumerate(tqdm(ft_loader, desc=f"Training in epoch {epoch}")):
        data = datas
        optimizer.zero_grad()
        video = {}

        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)

        # todo: create ranked dataset as input
        r_v = generate_ranking_video_tps(video)
        video['resize'] = r_v.to(device)

        if need_feat:
            scores, feats = model(video, inference=False,
                                  return_pooled_feats=True,
                                  reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            if (y_pred.dim() != 2):
                y_pred = y_pred.mean((-3, -2, -1))
        else:
            scores = model(video, inference=False,
                           reduce_scores=False)
            if len(scores) > 1:
                y_pred = reduce(lambda x, y: x + y, scores)
            else:
                y_pred = scores[0]
            if (y_pred.dim() != 2):
                y_pred = y_pred.mean((-3, -2, -1))

        # TODO: compute pairwise ranking hinge loss
        loss = serial_ranking_loss(y_pred, device)

        wandb.log({"train/pairwise_ranking_hinge_loss": loss.item(), })
        # print('training loss: {:.3f}'.format(loss.item()))
        loss_in_epoch.append(loss.item())

        loss.backward()
        optimizer.step()
        scheduler.step()

        # ft_loader.dataset.refresh_hypers()

        # todo: update lr? not sure?
        if model_ema is not None:
            model_params = dict(model.named_parameters())
            model_ema_params = dict(model_ema.named_parameters())
            for k in model_params.keys():
                model_ema_params[k].data.mul_(0.999).add_(
                    model_params[k].data, alpha=1 - 0.999
                )
    model.eval()

    # todo: draw the loss curve in this epoch
    plot_ranking_loss_in_epoch(loss_in_epoch, label='Training Loss')


# todo: rewrite code in this function
def inference_set(inf_loader, model, device, best_, save_model=True, suffix='s',
                  save_name="divide", deform_mode='tps'):
    # best_s, best_p, best_k, best_r = best_
    best_ranked = best_
    results = []
    loss_in_epoch = []

    for i, datas in enumerate(tqdm(inf_loader, desc="Validating")):

        data = datas
        result = dict()
        video = {}
        for key in sample_types:
            if key in data:
                video[key] = data[key].to(device)
                ## Reshape into clips
                b, c, t, h, w = video[key].shape
                video[key] = video[key].reshape(b, c, data["num_clips"][key], t // data["num_clips"][key], h,
                                                w).permute(0, 2, 1, 3, 4, 5).reshape(b * data["num_clips"][key], c,
                                                                                     t // data["num_clips"][key], h, w)

        with torch.no_grad():
            # todo: create ranked dataset as input
            if deform_mode == 'tps':
                r_v = generate_ranking_video_tps(video)
            else:  # linear deformation
                r_v = generate_ranking_video_linear(video, mode=deform_mode)
            video['resize'] = r_v.to(device)
            pred_scores = model(video)
            # compute ranking loss
            ranking_loss = serial_ranking_loss(pred_scores, device)

            result["pr_scores"] = pred_scores.cpu().numpy()
            result["ranking_loss"] = ranking_loss.item()
            loss_in_epoch.append(ranking_loss.item())

            del video, r_v
            results.append(result)

    val_ranked_loss = np.sum(loss_in_epoch)  # todo: update soon

    del result  # , video, video_up
    torch.cuda.empty_cache()

    if val_ranked_loss < best_ranked and save_model:
        state_dict = model.state_dict()
        torch.save(
            {
                "state_dict": state_dict,
                "validation_results": best_,
            },
            f"em_weights/{save_name}_ranked_{suffix}.pth",
        )

    print('best ranking loss changed from {:.2f} to {:.2f}'.format(best_ranked, min(best_ranked, val_ranked_loss)))
    best_ranked = min(best_ranked, val_ranked_loss)

    # todo: draw the loss curve in this epoch
    plot_ranking_loss_in_epoch(loss_in_epoch, label='Validation Loss')

    # todo: visualize validation results
    from visualize_plot import draw_fig2_rankiqa
    draw_fig2_rankiqa(results, deform_mode=deform_mode)

    # save recordings for further plot
    import pickle
    with open('./em_weights/val_results/val_dict_{}_{}.pkl'.format(deform_mode, suffix), 'wb') as f:
        pickle.dump(results, f)

    return best_ranked


def main():
    # Read In options
    with open('options/vRQA_Ablation.yml', "r") as f:
        opt = yaml.safe_load(f)
    print(opt)

    ## adaptively choose the device
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # avoid traffic jam
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # os.environ["WANDB_API_KEY"] = 'b612e7272a305e0ad1d20151608f2f19ff7da07e'
    os.environ["WANDB_MODE"] = 'offline'

    ## defining model and loading checkpoint

    # bests_ = []

    model = getattr(models, opt["model"]["type"])(**opt["model"]["args"]).to(device)
    # model = torch.nn.parallel.DataParallel(model)
    param_num = 0
    for param in model.parameters():
        param_num += int(np.prod(param.shape))
    print('Trainable params: %.2f million' % (param_num / 1e6))

    if opt.get("split_seed", -1) > 0:
        num_splits = 10
    else:
        num_splits = 1

    # read in dataset
    for split in range(num_splits):
        # TODO: for FlyEM videos, we only have 32 frames in one video, so set total_frames=32
        val_datasets = {}
        for key in opt["data"]:
            if key.startswith("val"):
                val_datasets[key] = getattr(datasets,
                                            opt["data"][key]["type"])(opt["data"][key]["args"], total_frames=32)

        val_loaders = {}
        for key, val_dataset in val_datasets.items():
            val_loaders[key] = torch.utils.data.DataLoader(
                val_dataset, batch_size=1, num_workers=opt["num_workers"], pin_memory=True,
            )

        train_datasets = {}
        for key in opt["data"]:
            if key.startswith("train"):
                train_dataset = getattr(datasets, opt["data"][key]["type"])(opt["data"][key]["args"], total_frames=32)
                train_datasets[key] = train_dataset

        train_loaders = {}
        for key, train_dataset in train_datasets.items():
            train_loaders[key] = torch.utils.data.DataLoader(
                train_dataset, batch_size=opt["batch_size"], num_workers=opt["num_workers"], shuffle=True,
            )

        run = wandb.init(
            project=opt["wandb"]["project_name"],
            name=opt["name"] + f'_{split}' if num_splits > 1 else opt["name"],
            reinit=True,
        )

        if "load_path_aux" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)["state_dict"]
            aux_state_dict = torch.load(opt["load_path_aux"], map_location=device)["state_dict"]

            from collections import OrderedDict

            fusion_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if "head" in k:
                    continue
                if k.startswith("vqa_head"):
                    ki = k.replace("vqa", "fragments")
                else:
                    ki = k
                fusion_state_dict[ki] = v

            for k, v in aux_state_dict.items():
                if "head" in k:
                    continue
                if k.startswith("frag"):
                    continue
                if k.startswith("vqa_head"):
                    ki = k.replace("vqa", "resize")
                else:
                    ki = k
                fusion_state_dict[ki] = v
            state_dict = fusion_state_dict
            print(model.load_state_dict(state_dict))

        elif "load_path" in opt:
            state_dict = torch.load(opt["load_path"], map_location=device)

            state_dict = state_dict["model"]
            model.load_pretrained(state_dict)

        # print(model)

        if opt["ema"]:
            from copy import deepcopy
            model_ema = deepcopy(model)
        else:
            model_ema = None

        # profile_inference(val_dataset, model, device)

        # finetune the model

        # Warm Up
        # Not used in our version? but 'best' updated
        print('Warming Up\n')
        param_groups = []

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                param_groups += [
                    {"params": value.parameters(), "lr": opt["optimizer"]["lr"] * opt["optimizer"]["backbone_lr_mult"]}]
            else:
                param_groups += [{"params": value.parameters(), "lr": opt["optimizer"]["lr"]}]

        optimizer = torch.optim.AdamW(lr=opt["optimizer"]["lr"], params=param_groups,
                                      weight_decay=opt["optimizer"]["wd"],
                                      )
        warmup_iter = 0
        for train_loader in train_loaders.values():
            warmup_iter += int(opt["warmup_epochs"] * len(train_loader))
        max_iter = int((opt["num_epochs"] + opt["l_num_epochs"]) * len(train_loader))
        lr_lambda = (
            lambda cur_iter: cur_iter / warmup_iter
            if cur_iter <= warmup_iter
            else 0.5 * (1 + math.cos(math.pi * (cur_iter - warmup_iter) / max_iter))
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda,
        )

        bests = {}
        bests_n = {}
        for key in val_loaders:  # todo: we only have ranking loss, changing the validation code
            bests[key] = 10000
            bests_n[key] = 10000  # changed since 2024.01.14

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = False

        for epoch in range(opt["l_num_epochs"]):
            print(f"Linear Epoch {epoch}:")
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch,
                    opt.get("need_upsampled", False), opt.get("need_feat", False), opt.get("need_fused", False),
                )
            for key in val_loaders:
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix=key + '_n',
                    )
                else:
                    bests_n[key] = bests[key]

        for key, value in dict(model.named_children()).items():
            if "backbone" in key:
                for param in value.parameters():
                    param.requires_grad = True

        # Start Training
        print('Start Training and Validating\n')
        for epoch in range(opt["num_epochs"]):
            print(f"Finetune Epoch {epoch}:")

            # Training Process
            for key, train_loader in train_loaders.items():
                finetune_epoch(
                    train_loader, model, model_ema, optimizer, scheduler, device, epoch,
                    opt.get("need_feat", False))

            # Validating Process
            for key in val_loaders:
                if model_ema is not None:
                    bests_n[key] = inference_set(
                        val_loaders[key],
                        model,
                        device, bests_n[key], save_model=opt["save_model"], save_name=opt["name"],
                        suffix=key + '_' + opt["model"]["args"]["feat_type"]
                    )
                else:
                    bests_n[key] = bests[key]

        run.finish()


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    main()

