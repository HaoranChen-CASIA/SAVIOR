import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation


def plot_scatter_figure(gt_labels, pr_labels, fit=False):
    font1 = {'weight': 'bold',
             'style': 'normal',
             'size': 12,
             }

    fig, ax = plt.subplots()
    ax.scatter(pr_labels, gt_labels, s=1, label='Stable')

    if fit:
        def fourth_order_polynomial(x, a, b, c, d, e):
            return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

        popt, pcov = curve_fit(fourth_order_polynomial, np.array(pr_labels), np.array(gt_labels))
        points = np.linspace(0, 100, 1000)
        plt.plot(points, fourth_order_polynomial(points, *popt), 'r-', label='Fitted Curve')

    plt.xlabel('Pred score')
    plt.ylabel('GT mos')
    plt.xlim((0, 100))
    plt.ylim((0, 100))
    plt.show()


def plot_ranking_loss_in_epoch(loss_in_epoch, label='Training Loss'):
    font1 = {'weight': 'bold',
             'style': 'normal',
             'size': 12,
             }

    step = np.linspace(1, len(loss_in_epoch), len(loss_in_epoch))
    # TODO: 新增根据train or validate选择loss曲线颜色，便于视觉区分
    if label == 'Training Loss':
        plt.plot(step, loss_in_epoch, 'b-', label=label)
    else:  # validation or test
        plt.plot(step, loss_in_epoch, 'r-', label=label)
    plt.xlabel('Iteration in epoch')
    plt.ylabel('Pairwise ranking hinge loss')
    plt.legend()
    plt.grid()
    plt.show()


def draw_fig2_rankiqa(val_results, deform_mode='tps'):
    print('\nCurrent dataset: {}'.format(deform_mode))
    scores = np.array([val_i['pr_scores'] for val_i in val_results]).squeeze(axis=2)

    # step 1 : normalize
    mean, std = scores.mean(), scores.std()
    print('Dataset mean & std: {:.3f}, {:.3f}'.format(mean, std))
    scores_normalized = (scores - mean) / std

    # step 2: plot histogram
    colors = ['red', 'tan', 'lime', 'blue']
    # plt.hist(scores_normalized, bins=np.arange(-3.2, 1.4, 0.2), color=colors, label=['0', '1', '2', '3'])
    for i in range(scores.shape[1]):
        plt.hist(scores_normalized[:, i], bins=10, label='rank'+str(i)
                 , alpha=0.7, edgecolor='black')
    plt.legend()
    plt.title('Random ' + deform_mode)
    plt.xlabel('Normalized Score')
    plt.grid()
    plt.savefig('ranked_histo_{}.tif'.format(deform_mode))

    plt.show()


def draw_heatmaps_with_scores(data, depth_id, save_dir, show=True, save=False, title=''):
    """
        suppose input data is a 2d ndarray
        default score range is [0, 1] as they are normalized
    """
    # 创建热力图
    plt.figure(figsize=(data.shape[0] + 1, data.shape[1] + 1))
    # heatmap = plt.imshow(data, cmap='Oranges', interpolation='nearest', vmin=0, vmax=1)
    heatmap = plt.imshow(data, cmap="coolwarm", interpolation='nearest', vmin=0, vmax=1)

    # 添加颜色条并设置值域范围
    cbar = plt.colorbar(heatmap)
    cbar.set_label('Value Range [0, 1]', rotation=270, labelpad=20)  # 设置颜色条标签

    # 显示数值
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='black')

    # 设置轴标签
    plt.xlabel('Column')
    plt.ylabel('Row')

    # 显示图像
    plt.title('{}, Depth {}'.format(title, depth_id))
    if save:
        plt.savefig(save_dir + "depth{}.png".format(depth_id), dpi=300, bbox_inches='tight')
    if show:
        plt.show()


def plot_3d_heatmap(data):
    """
    绘制适用于 (z, x, y) 形状的 3D 热力图，其中:
    - z: 层数 (深度)
    - x: 宽度
    - y: 高度
    """
    # 获取数据维度
    z_dim, x_dim, y_dim = data.shape  # 确保适用于 (z, x, y) 形状

    # 颜色映射
    cmap = plt.get_cmap("coolwarm")  # 热力图颜色
    # norm = plt.Normalize(data.min(), data.max())  # 归一化数值
    norm = plt.Normalize(0, 1)  # 归一化数值

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 设定物理比例 (假设真实尺寸 12:39:80)
    depth, height, width = data.shape
    # scaling = np.array([width, height, depth])  # 物理尺寸顺序
    scaling = np.array([height, width, depth])  # 物理尺寸顺序

    # 归一化缩放
    scaling = scaling / scaling.max()

    # 设置缩放比例，使其在物理上更合理
    ax.set_box_aspect(scaling)  # `set_box_aspect` 让长方体比例更符合实际

    # 遍历所有 (z, x, y) 坐标
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                value = data[z, x, y]
                color = cmap(norm(value))  # 根据数值获取颜色

                # 立方体的 8 个顶点坐标 (根据 x, y, z 位置计算)
                vertices = [
                    (x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z),
                    (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)
                ]

                # 定义立方体的 6 个面
                faces = [
                    [vertices[j] for j in [0, 1, 5, 4]],  # 前
                    [vertices[j] for j in [1, 2, 6, 5]],  # 右
                    [vertices[j] for j in [2, 3, 7, 6]],  # 后
                    [vertices[j] for j in [3, 0, 4, 7]],  # 左
                    [vertices[j] for j in [4, 5, 6, 7]],  # 顶
                    [vertices[j] for j in [0, 1, 2, 3]],  # 底
                ]

                # 绘制立方体
                poly = Poly3DCollection(faces, alpha=0.6, facecolor=color, edgecolor="k")
                ax.add_collection3d(poly)

                # # 在立方体中心标注数值
                # ax.text(x + 0.5, y + 0.5, z + 0.5, str(value),
                #         ha="center", va="center", fontsize=8, color="black")

    # 设定坐标轴范围
    ax.set_xlim([0, x_dim])
    ax.set_ylim([0, y_dim])
    ax.set_zlim([0, z_dim])

    # 轴标签
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f"3D Heatmap ({z_dim}×{x_dim}×{y_dim})")

    # ** 添加颜色条 **
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)  # 颜色映射对象
    sm.set_array([])  # 必须设置，否则 colorbar 无法创建
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Value Intensity", fontsize=12)  # 颜色条标题

    plt.show()


def plot_3d_heatmap_anime(data, save_gif=True, gif_name="3d_heatmap", fps=1):
    """
    绘制适用于 (z, x, y) 形状的 3D 热力图，并生成沿 y 方向递减消失的动画。
    """
    print('Generating gif for fps{}\n'.format(fps))
    # 获取数据维度
    z_dim, x_dim, y_dim = data.shape

    # 颜色映射
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(0, 1)  # 归一化数值

    # 创建 3D 图
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    # # 设定物理比例
    # scaling = np.array([y_dim, x_dim, z_dim])  # 物理尺寸顺序
    # scaling = scaling / scaling.max()
    # ax.set_box_aspect(scaling)  # 设置长方体比例

    # 设定物理比例 (假设真实尺寸 12:39:80)
    depth, height, width = data.shape
    scaling = np.array([height, width, depth])  # 物理尺寸顺序

    # 归一化缩放
    scaling = scaling / scaling.max()

    # 设置缩放比例，使其在物理上更合理
    ax.set_box_aspect(scaling)  # `set_box_aspect` 让长方体比例更符合实际

    # 存储立方体对象
    cubes = []

    # 遍历所有 (z, x, y) 坐标
    for z in range(z_dim):
        for x in range(x_dim):
            for y in range(y_dim):
                value = data[z, x, y]
                color = cmap(norm(value))  # 根据数值获取颜色

                # 立方体的 8 个顶点坐标
                vertices = [
                    (x, y, z), (x + 1, y, z), (x + 1, y + 1, z), (x, y + 1, z),
                    (x, y, z + 1), (x + 1, y, z + 1), (x + 1, y + 1, z + 1), (x, y + 1, z + 1)
                ]

                # 定义立方体的 6 个面
                faces = [
                    [vertices[j] for j in [0, 1, 5, 4]],  # 前
                    [vertices[j] for j in [1, 2, 6, 5]],  # 右
                    [vertices[j] for j in [2, 3, 7, 6]],  # 后
                    [vertices[j] for j in [3, 0, 4, 7]],  # 左
                    [vertices[j] for j in [4, 5, 6, 7]],  # 顶
                    [vertices[j] for j in [0, 1, 2, 3]],  # 底
                ]

                # 绘制立方体
                poly = Poly3DCollection(faces, alpha=1.0, facecolor=color, edgecolor="k")
                ax.add_collection3d(poly)
                cubes.append((poly, y))  # 存储立方体及其 y 坐标

    # 设定坐标轴范围
    ax.set_xlim([0, x_dim])
    ax.set_ylim([0, y_dim])
    ax.set_zlim([0, z_dim])

    # 轴标签
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title(f"3D Heatmap ({z_dim}×{x_dim}×{y_dim})")

    # 添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, aspect=20, pad=0.1)
    cbar.set_label("Value Intensity", fontsize=12)

    # # 动画更新函数
    # def update(frame):
    #     for poly, y_pos in cubes:
    #         if y_pos >= frame:  # 如果立方体的 y 坐标大于当前帧，逐渐减小透明度
    #             alpha = max(0, 1 - (y_pos - frame) / y_dim)  # 透明度递减
    #             poly.set_alpha(alpha)
    #     return cubes

    # # 动画更新函数
    # def update(frame):
    #     for poly, y_pos in cubes:
    #         # y 越小越透明
    #         alpha = max(0, (y_pos - frame) / y_dim)  # 透明度随 y 减小而减小
    #         poly.set_alpha(alpha)
    #         poly.set_edgecolor((0, 0, 0, alpha))  # 设置边框透明度（RGBA 格式）
    #     return cubes

    # 动画更新函数
    def update(frame):
        print(f"当前帧: {frame}")  # 打印当前帧数
        for poly, y_pos in cubes:
            if y_pos < frame:  # y_pos < frame 的部分透明度为 0
                alpha = 0
            else:  # y_pos >= frame 的部分透明度为 1
                alpha = 1
            poly.set_alpha(alpha)  # 设置面透明度
            poly.set_edgecolor((0, 0, 0, alpha))  # 设置边框透明度（RGBA 格式）
        return cubes

    # 创建动画
    ani = FuncAnimation(fig, update, frames=y_dim + 1, interval=1000, blit=False)

    # 保存为 GIF
    if save_gif:
        ani.save(gif_name + '_fps{}.gif'.format(fps), writer="pillow", fps=fps)
        print(f"动画已保存为 {gif_name}, fps={fps}")

    # plt.show()

def statistic_repo_multi_datas(datas, data_legend, threshold=0, xy_lim=[]):
    i = 0
    for data in datas:
        # 统计直方图并计算百分比
        values, counts = np.unique(data, return_counts=True)
        percentiles = np.cumsum(counts) / np.sum(counts) * 100

        # 绘制曲线图
        plt.plot(values, percentiles, label=data_legend[i])
        i += 1

    # 添加图例
    plt.legend()

    # 添加参考线（如 75th 百分位）
    if threshold > 0:
        plt.axvline(x=threshold, color='r', linestyle='--', label='CPC={}'.format(threshold))

    # 设置横坐标范围（如 0 到 0.3）
    if xy_lim == []:
        print('Print all range')
    else:
        plt.xlim(0, xy_lim[0])  # 将横坐标限制在特定显示范围
        plt.ylim(-0.1, xy_lim[1])

    plt.xlabel('Scores')
    plt.ylabel('Percentile')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.title('Distribution Percentile')
    plt.show()


def draw_box_plot(datas, title):
    # 绘制箱型图
    plt.boxplot(datas, vert=True, patch_artist=True,
                labels=['new', 'old'])

    # 添加标题和标签
    plt.title(title)
    plt.xlabel("Groups")
    plt.ylabel("Values")

    # 显示图像
    plt.show()


if __name__ == '__main__':
    import pickle

    # # read in score data
    # deform_mode = 'tps'
    # with open('./em_weights/validation_dict_{}.pkl'.format(deform_mode), 'rb') as f:
    #     val_results = pickle.load(f)
    #
    # # val_scores = np.array([val_i['pr_scores'] for val_i in val_results]).squeeze(axis=2)
    #
    # # draw figure
    # draw_fig2_rankiqa(val_results, deform_mode)
    #
    # print('done')

    # print('ablation mode')
    #
    # for feat_mode in ['sf', 'of', 'bf', 'sf+of', 'sf+bf', 'of+bf']:
    #     with open('./em_weights/val_results/val_dict_tps_val_{}.pkl'.format(feat_mode), 'rb') as f:
    #         val_results = pickle.load(f)
    #     draw_fig2_rankiqa(val_results, feat_mode)

    # """
    #     Plot depth-wise heatmap for SSH_down8 data, 20241219
    # """
    # area = '19_15'
    # root_dir = '/mnt/Ext001/chenhr/dataset/em_videos/SSH_down8/'
    #
    # data_dir = root_dir + area + '/'
    # heatmap_save_dir = data_dir + 'Heatmaps/'
    # os.makedirs(heatmap_save_dir, exist_ok=True)
    #
    # data = np.loadtxt(data_dir + 'SAVIOR_Scores.txt', delimiter=',', skiprows=1)
    # d, h, w = 10, 5, 5
    #
    # final_scores = data[:, -1]
    # reshaped_final = final_scores.reshape(d, h, w)
    #
    # for depth in range(d):
    #     draw_heatmaps_with_scores(reshaped_final[depth, :], depth, heatmap_save_dir, False, True)

    # # TODO:250701 visualize SAM_OF_divided results
    # date = '0709'
    # with open(f'./em_weights/train_{date}/train_loss.pkl', 'rb') as f:
    #     loss_in_epoch = pickle.load(f)
    #
    # plot_ranking_loss_in_epoch(loss_in_epoch, label='Training Loss')
    #
    # with open(f'./em_weights/train_{date}/val_loss.pkl', 'rb') as f:
    #     loss_in_epoch = pickle.load(f)
    #
    # plot_ranking_loss_in_epoch(loss_in_epoch, label='Validation Loss')
    #
    # with open(f'./em_weights/val_results/J{date}_tps.pkl', 'rb') as f:
    #     val_results = pickle.load(f)
    #
    # draw_fig2_rankiqa(val_results, deform_mode='tps')


    # with open('./GatingWeight_CCloss/Gating_CCloss_v0719rank13_loss.pkl', 'rb') as f:
    #     loss_in_epoch = pickle.load(f)
    #
    # plot_ranking_loss_in_epoch(loss_in_epoch, label='Training Loss')

    for epoch in range(0, 3):
        with open(f'./GatingWeight_CCloss/Gating_CCloss_v0725rank13_loss_epoch{epoch}.pkl', 'rb') as f:
            loss_in_epoch = pickle.load(f)

        plot_ranking_loss_in_epoch(loss_in_epoch, label=f'Training Loss ep{epoch}')




