from pathlib import Path
import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from load_data import NewCustomDataset_cluster, NewCustomDataset_all
import torch.multiprocessing
from models.superglue import SuperGlue


# Notesp 命令行执行程序时打开使用
# import os, resource

# 设置资源限制
# resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))
#
# # 配置内存分配器
# os.environ["PYTHONMALLOC"] = "malloc_debug"

# 配置CUDA
# if torch.cuda.is_available():
    # torch.cuda.set_per_process_memory_fraction(0.8)
    # torch.cuda.memory._set_allocator_settings('max_split_size_mb:128')
#################################################################

torch.set_grad_enabled(True)
# torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser(
    description='Image pair matching and pose evaluation with SuperGlue',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--model', '-m', default='log/weight/all4.path',  # log/weight/best_1_all.pth
                    help='Give a model to test')
parser.add_argument('--model_save_name', '-m_in', default='log/weight/part4_useother.pth',  # best_4_test_5_no_filter.pth
                    help='Give a model to test')
parser.add_argument(
    '--descriptor_dim', type=int, default=4,
    help='number of descriptor length'
    ' (Must be positive)')
parser.add_argument(
    '--batch_size', type=int, default=10,# 太大就超内存了
    help='batch_size')
parser.add_argument(
    '--train_path', type=str, default='/home/lsp/matchpoint/target20_datas_diff/train_datasets/', #/home/lsp/matchpoint/data_prepare_G/10000_data_5s_train/
    help='Path to the directory of training imgs.')
parser.add_argument(
    '--use_other', type=bool, default=False,
    help='Whether to use RCS, size and other information')
parser.add_argument(
    '--part_covis', type=bool, default=False,
    help='Whether to use RCS, size and other information')
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
    '--match_threshold', type=float, default=0.00001,
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
    '--cuda_device', type=int, default=0,
    help='choose cuda device')

parser.add_argument(
    '--eval_input_dir', type=str, default='assets/scannet_sample_images/',
    help='Path to the directory that contains the images')
parser.add_argument(
    '--eval_output_dir', type=str, default='dump_match_pairs/',
    help='Path to the directory in which the .npz results and optional,'
            'visualizations are written')
parser.add_argument(
    '--epoch', type=int, default=500,
    help='Number of epoches')

random.seed(2)


def test(model1, testloader1):
    model1.eval()

    scr_mean, avg_purity_mean, acc_mean = np.array([]),np.array([]),np.array([])
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

        constraint_group = data['groups']

        accum_array = np.array([])
        i = 0.0
        satisfied_groups = 0
        total_purity = 0
        count_sum, count_all = 0, 0
        for pred_label in pred_labels:
            for group in constraint_group:
                if len(group) < 2:
                    continue

                group_i = group[1][torch.where(group[0]==i)]
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
                if len(unique_labels)>0:
                    total_purity += max_count / len(unique_labels)
                count_sum = count_sum + max_count
                count_all = count_all + constrained_labels.shape[0]
            i = i + 1

        scr = satisfied_groups/len(constraint_group) / len(pred_labels)
        avg_purity = total_purity /len(constraint_group) / len(pred_labels)
        acc  = count_sum/count_all
        scr_mean = np.append(scr_mean, scr)
        avg_purity_mean = np.append(avg_purity_mean, avg_purity)
        acc_mean = np.append(acc_mean, acc)

        # print('unique_labels:'+str(accum_array.mean()))
    print('satisfied:' + str(scr_mean.mean()) + '  purity:' + str(avg_purity_mean.mean()) + '  acc:' + str(acc_mean.mean()))
    return scr_mean.mean(), avg_purity_mean.mean(), acc_mean.mean()

        # for k, v in pred.items():
        #     pred[k] = v[0]
        # pred = {**pred, **data}


if __name__ == '__main__':
    opt = parser.parse_args()
    torch.cuda.set_device(opt.cuda_device)

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
    if opt.descriptor_dim < 1.5: #NOTEsp 点数为1的话 就用4维的向量
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

    #  load training test data
    train_set = NewCustomDataset_cluster(opt.train_path + "whole"+"/train", 'train', opt.use_other, opt.descriptor_dim)
    test_set = NewCustomDataset_cluster(opt.train_path + "whole"+"/test", 'test', opt.use_other, opt.descriptor_dim)
    train_loader = torch.utils.data.DataLoader(dataset=train_set, shuffle=False, batch_size=opt.batch_size, drop_last=False)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, shuffle=False, batch_size=100)

    # dl = DataLoader(my_ds, batch_size=64, shuffle=True, num_workers=1)

    superglue = SuperGlue(config.get('superglue', {}))

    if torch.cuda.is_available():
        superglue.cuda() # make sure it trains on GPU
        # superglue = torch.nn.DataParallel(superglue, device_ids=[0, 1])
    else:
        print("### CUDA not available ###")
    ac_best = 0
    if len(opt.model) == 0: # 决定测试还是训练
        optimizer = torch.optim.Adam(superglue.parameters(), lr=opt.learning_rate)
        mean_loss = []

        # start training
        for epoch in range(1, opt.epoch+1):
            epoch_loss = 0
            superglue.train()
            for i, preds in enumerate(train_loader):
                for pred in preds:
                    for k in pred:
                    # if k != 'file_name' and k!='image0' and k!='image1':
                        if type(pred[k]) == torch.Tensor:
                            pred[k] = Variable(pred[k].cuda())
                        else:
                            pred[k] = Variable(torch.stack(pred[k]).cuda())

                data = superglue(preds, lambda_constraint=1.0)
                for pred in preds:
                    for k, v in pred.items():
                        pred[k] = v[0]
                # preds = {**preds, **data}

                # if pred['skip_train'] == True: # image has no keypoint
                #     continue

                # process loss
                Loss = data['loss']
                epoch_loss += Loss.item()
                mean_loss.append(Loss)

                superglue.zero_grad()
                Loss.backward()
                optimizer.step()

                # for every 50 images, print progress and visualize the matches
                if (i+1) % 40== 0:
                    # pass
                    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                        .format(epoch, opt.epoch, i+1, len(train_loader), torch.mean(torch.stack(mean_loss)).item()))

            # save checkpoint when an epoch finishes
            epoch_loss /= len(train_loader)
            model_out_path = opt.model_save_name
            if epoch % 2 == 0:
                scr, avg_purity, acc = test(superglue, test_loader)
                if acc > ac_best:
                    ac_best = acc
                    torch.save(superglue.state_dict(), model_out_path)
                    print ('Epoch [{}/{}], Checkpoint saved to {} satisfied:{} acc:{}'
                        .format(epoch, opt.epoch,  model_out_path, scr, acc))

    else:
        # saved_model = torch.load(args.model)
        superglue.load_state_dict(torch.load(opt.model))
        # model = torch.load(opt.model)
        scr, avg_purity, num_rate = test(superglue, test_loader)
        # print(ac)


