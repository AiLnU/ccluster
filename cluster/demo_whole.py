from pathlib import Path
import argparse
import random
import numpy as np
import matplotlib.cm as cm
import torch
import torch.nn as nn
from torch.autograd import Variable
from load_data import NewCustomDataset_cluster, NewCustomDataset_all
import os
import torch.multiprocessing
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans

from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, read_image_modified)

from models.superpoint import SuperPoint
from models.superglue import SuperGlue
# from models.supergluekcenter import SuperGlue #Notesp alt 选择合适的聚类中心
from models.matchingForTraining import MatchingForTraining

torch.set_grad_enabled(True)
torch.multiprocessing.set_sharing_strategy('file_system')

Len = 1
parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', '-m', default='log/weight/all'+str(Len)+'.pth',  # best_4_test_5_no_filter.pth
                    help='Give a model to test')
parser.add_argument('--descriptor_dim', type=int, default=Len,
    help='number of descriptor length'
         ' (Must be positive)')
parser.add_argument('--fig_save_path', type=str, default='fig_cluster/'+str(Len)+'a',  # best_4_test_5_no_filter.pth
                    help='path to save fig')
parser.add_argument(
    '--batch_size', type=int, default=10,  # 太大就超内存了
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='/home/lsp/matchpoint/target20_datas_diff/train_datasets/',
    # /home/lsp/matchpoint/data_prepare_G/10000_data_5s_train/
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--use_other', type=bool, default=False,
    help='Whether to use RCS, size and other information')
parser.add_argument(
    '--part_covis', type=bool, default=False,
    help='Whether to use RCS, size and other information')
parser.add_argument(
    '--plotfig', type=bool, default=True,
    help='Whether to plot fig')
parser.add_argument(
    '--learning_rate', type=int, default=0.0001,
    help='Learning rate')
parser.add_argument(
    '--viz', action='store_true',
    help='Visualize the matches and dump the plots')
parser.add_argument(
    '--eval', action='store_true',
    help='Perform the evaluation'
         ' (requires ground truth pose and intrinsics)')
parser.add_argument(
    '--superglue', choices={'indoor', 'outdoor'}, default='indoor',
    help='SuperGlue weights')
parser.add_argument(
    '--max_keypoints', type=int, default=1024,
    help='Maximum number of keypoints detected by Superpoint'
         ' (\'-1\' keeps all keypoints)')
parser.add_argument(
    '--keypoint_threshold', type=float, default=0.005,
    help='SuperPoint keypoint detector confidence threshold')
parser.add_argument(
    '--nms_radius', type=int, default=4,
    help='SuperPoint Non Maximum Suppression (NMS) radius'
         ' (Must be positive)')
parser.add_argument(
    '--sinkhorn_iterations', type=int, default=20,
    help='Number of Sinkhorn iterations performed by SuperGlue')
parser.add_argument(
    '--match_threshold', type=float, default=0.2,
    help='SuperGlue match threshold')
parser.add_argument(
    '--resize', type=int, nargs='+', default=[640, 480],
    help='Resize the input image before running inference. If two numbers, '
         'resize to the exact dimensions, if one number, resize the max '
         'dimension, if -1, do not resize')
parser.add_argument(
    '--resize_float', action='store_true',
    help='Resize the image after casting uint8 to float')

parser.add_argument(
    '--cache', action='store_true',
    help='Skip the pair if output .npz files are already found')
parser.add_argument(
    '--show_keypoints', action='store_true',
    help='Plot the keypoints in addition to the matches')
parser.add_argument(
    '--fast_viz', action='store_true',
    help='Use faster image visualization based on OpenCV instead of Matplotlib')
parser.add_argument(
    '--viz_extension', type=str, default='png', choices=['png', 'pdf'],
    help='Visualization file extension. Use pdf for highest-quality.')

parser.add_argument(
    '--opencv_display', action='store_true',
    help='Visualize via OpenCV before saving output images')
parser.add_argument(
    '--eval_pairs_list', type=str, default='assets/scannet_sample_pairs_with_gt.txt',
    help='Path to the list of image pairs for evaluation')
parser.add_argument(
    '--shuffle', action='store_true',
    help='Shuffle ordering of pairs before processing')
parser.add_argument(
    '--max_length', type=int, default=-1,
    help='Maximum number of pairs to evaluate')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optional,'
         'visualizations are written')
parser.add_argument(
    '--epoch', type=int, default=1000,
    help='Number of epoches')

random.seed(2)


# torch.cuda.set_device(1)

Colors = ['aliceblue', 'antiquewhite', 'aqua', 'aquamarine', 'azure',
'beige', 'bisque', 'black', 'blanchedalmond', 'blue',
'blueviolet', 'brown', 'burlywood', 'cadetblue', 'chartreuse',
'chocolate', 'coral', 'cornflowerblue', 'cornsilk', 'crimson']

Edgecolors=['red', 'black']

legend_handles = []

a=1


def test(model1, testloader1):
    model1.eval()
    scr_mean, avg_purity_mean, acc_mean = np.array([]), np.array([]), np.array([])
    iik = 0 #　保存图像的编号
    for i, preds in enumerate(testloader1):
        for pred in preds:
            for k in pred:
                # if k != 'file_name' and k != 'image0' and k != 'image1':
                if type(pred[k]) == torch.Tensor:
                    pred[k] = Variable(pred[k].cuda())
                else:
                    pred[k] = Variable(torch.stack(pred[k]).cuda())

        data = superglue(preds)
        pred_labels = data['indices']
        batchsize = pred_labels.shape[0]
        constraint_group = data['groups']
        pred_labels = pred_labels.cpu()

        # section 画图
        kpts_all = data['kpts'].cpu()
        for ii in range(batchsize):
            cc = 0
            plt.figure(figsize=(12, 8), dpi=100)
            legend_handles = []
            for group in constraint_group:
                if len(group) < 2:
                    continue

                group_i = group[1][torch.where(group[0] == ii)]
                constrained_labels = pred_labels[ii][group_i]
                kpts = kpts_all[ii][group_i]

                #　统计聚类对错，为了画图，错的则边沿为红色
                unique_vals, counts = torch.unique(constrained_labels, return_counts=True)
                # 找到最高频率值（众数）
                max_count = counts.max()
                most_frequent_vals = unique_vals[counts == max_count]
                target_val = most_frequent_vals[0]
                mask = (constrained_labels == target_val)

                for k in range(kpts.shape[0]):
                    scatter = plt.scatter(
                        kpts[k, 0],
                        kpts[k, 1],
                        s=100,  # 点的大小
                        c=Colors[cc],
                        alpha=0.7,
                        edgecolors=Edgecolors[mask[k]],
                        # label=cc
                    )
                # legend_handles.append(scatter)

                for k in range(kpts.shape[0]):
                    plt.annotate(
                        str(constrained_labels[k].item()),
                        (kpts[k, 0], kpts[k, 1]),
                        xytext=(5, 5),  # 标签偏移量
                        textcoords='offset points',
                        fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.7)
                    )
                cc += 1

            plt.title('Coordinate Points with Indices and Categories', fontsize=14)
            plt.xlabel('X Coordinate', fontsize=12)
            plt.ylabel('Y Coordinate', fontsize=12)
            # 添加网格
            plt.grid(True, linestyle='--', alpha=0.6)
            plt.savefig(opt.fig_save_path + f'/demo_{iik}.png', pad_inches=0)
            iik += 1  # 保存图像的编号


        # section  计算指标
        accum_array = np.array([])
        i = 0.0
        satisfied_groups = 0
        total_purity = 0
        count_sum, count_all = 0, 0
        for pred_label in pred_labels:
            for group in constraint_group:
                if len(group) < 2:
                    continue

                group_i = group[1][torch.where(group[0] == i)]
                constrained_labels = pred_label[group_i]
                _indices = torch.where(constrained_labels != -1)[0]
                constrained_labels = constrained_labels[_indices]
                unique_labels, count = torch.unique(constrained_labels, return_counts=True)
                accum_array = np.append(accum_array, len(unique_labels))
                if len(unique_labels) == 1:
                    satisfied_groups += 1

                if count.numel() > 0:  # 或者 len(count) > 0
                    max_count = count.max().item()
                else:
                    max_count = 0
                # max_count = count.max().item()
                if len(unique_labels) > 0:
                    total_purity += max_count / len(unique_labels)
                count_sum = count_sum + max_count
                count_all = count_all + constrained_labels.shape[0]
            i = i + 1

        scr = satisfied_groups / len(constraint_group) / len(pred_labels)
        avg_purity = total_purity / len(constraint_group) / len(pred_labels)
        acc = count_sum / count_all
        scr_mean = np.append(scr_mean, scr)
        avg_purity_mean = np.append(avg_purity_mean, avg_purity)
        acc_mean = np.append(acc_mean, acc)
        print('unique_labels:'+str(accum_array.mean())+'  satisfied:'+str(scr)+'  purity:'+str(avg_purity)+'  acc:'+str(acc))

    return scr_mean.mean(), avg_purity_mean.mean(), acc_mean.mean()

    # for k, v in pred.items():
    #     pred[k] = v[0]
    # pred = {**pred, **data}


if __name__ == '__main__':
    opt = parser.parse_args()
    # print(opt)#Notesp  print

    # make sure the flags are properly used
    assert not (opt.opencv_display and not opt.viz), 'Must use --viz with --opencv_display'
    assert not (opt.opencv_display and not opt.fast_viz), 'Cannot use --opencv_display without --fast_viz'
    assert not (opt.fast_viz and not opt.viz), 'Must use --viz with --fast_viz'
    assert not (opt.fast_viz and opt.viz_extension == 'pdf'), 'Cannot use pdf extension with --fast_viz'

    # store viz results
    eval_output_dir = Path(opt.eval_output_dir)
    eval_output_dir.mkdir(exist_ok=True, parents=True)
    # print('Will write visualization images to',
    #     'directory \"{}\"'.format(eval_output_dir)) #Notesp  print
    descriptor_dim = opt.descriptor_dim
    if opt.descriptor_dim < 1.5:  # NOTEsp 点数为1的话 就用4维的向量
        descriptor_dim = 4
    config = {
        'superglue': {
            'weights': opt.superglue,
            'sinkhorn_iterations': opt.sinkhorn_iterations,
            'match_threshold': opt.match_threshold,
            'descriptor_dim': descriptor_dim,
            'use_other': opt.use_other,
            'part_covis':opt.part_covis
        }
    }

    # load training test data
    test_set = NewCustomDataset_cluster(opt.train_path + "whole" + "/test", 'test', opt.use_other, opt.descriptor_dim)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=opt.batch_size)

    # dl = DataLoader(my_ds, batch_size=64, shuffle=True, num_workers=1)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.cuda()  # make sure it trains on GPU
        # superglue = torch.nn.DataParallel(superglue, device_ids=[0, 1])
    else:
        print("### CUDA not available ###")
    ac_best = 0
        # saved_model = torch.load(args.model)
    superglue.load_state_dict(torch.load(opt.model))
        # model = torch.load(opt.model)
    scr, avg_purity, num_rate = test(superglue, test_loader)
        # print(ac)


