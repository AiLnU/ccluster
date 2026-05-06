import numpy as np
import torch
import os
import cv2
import pickle
import random
import datetime
import tqdm

from scipy.spatial.distance import cdist
from torch.utils.data import Dataset


class NewCustomDataset_all(Dataset):
    def __init__(self, data_path: str, model: str='test', target_length: int = 8, target_nums: int = 20):
        """

        :param data_path: 数据存放路径
        :param seq_length: 序列长度
        """
        self.data_path = data_path
        self.target_nums = target_nums
        self.model = model
        self.target_length = target_length

        # if model == 'test':
        #     random.seed(1)

        all_data_files = os.listdir(data_path)
        self.all_data = []
        for file in all_data_files:
            with open(os.path.join(data_path, file), 'rb') as f:
                data = pickle.load(f)
                self.all_data.append(data)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        cur_data = self.all_data[index]

        x_data_0 = dict(
            pos_seq=torch.tensor(cur_data['position'][0], dtype=torch.float32),
            vr_seq=torch.tensor(cur_data['relative_velocity'][0], dtype=torch.float32),
            size_seq=torch.tensor(cur_data['size_seq'][0], dtype=torch.float32),
        )
        x_data_1 = dict(
            pos_seq=torch.tensor(cur_data['position'][1], dtype=torch.float32),
            vr_seq=torch.tensor(cur_data['relative_velocity'][1], dtype=torch.float32),
            size_seq=torch.tensor(cur_data['size_seq'][1], dtype=torch.float32),
        )

        label_id = torch.tensor(cur_data['label'], dtype=torch.int16)

        # for debugging
        for val1, val2 in zip(x_data_0.values(), x_data_1.values()):
            if torch.isnan(val1).any() or torch.isnan(val2).any():
                raise ValueError
        if torch.isnan(label_id).any():
            raise ValueError

        # if torch.isnan(list(x_data_0.values())).any() or torch.isnan(list(x_data_1.values())).any() or torch.isnan(
        #         label_id).any():
        #     raise ValueError

        label_data = torch.zeros((self.target_nums, self.target_nums), dtype=torch.float32)
        for i in range(self.target_nums):
            uav1_target_id = label_id[i][0]
            uav2_target_id = label_id[i][1]    # bug fixed, liu zhenyu
            label_data[uav1_target_id - 1][uav2_target_id - 1] = 1

        rdm0 = [random.randint(0, self.target_nums-1) for _ in range(self.target_nums)]
        rdm1 = [random.randint(0, self.target_nums-1) for _ in range(self.target_nums)]

        # if self.model == 'test':
        a = self.target_length #NOTEsp 轨迹的长度,为1则用其他信息
        kpt0 = x_data_0['pos_seq'][:, 0, :]
        kpt1 = x_data_1['pos_seq'][:, 0, :]

        if a > 1.5:
            desc0 = x_data_0['pos_seq'][:, :a, 0]/x_data_0['pos_seq'][:, :a, 1]  # NOTEsp 用的除法
            desc1 = x_data_1['pos_seq'][:, :a, 0]/x_data_1['pos_seq'][:, :a, 1]
        else:
            kpt0min = kpt0.min(0)[0]
            kpt0max = kpt0.max(0)[0]
            kpt1min = kpt1.min(0)[0]
            kpt1max = kpt1.max(0)[0]

            x0 = (kpt0 - kpt0min) / (kpt0max - kpt0min)
            x1 = (kpt1 - kpt1min) / (kpt1max - kpt1min)
            desc0 = torch.cat([x_data_0['pos_seq'][:, :a, 0] / x_data_0['pos_seq'][:, :a, 1], x_data_0['vr_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 1]],1)
            desc1 = torch.cat([x_data_1['pos_seq'][:, :a, 0] / x_data_1['pos_seq'][:, :a, 1], x_data_1['vr_seq'][:, :a, 0], x_data_1['size_seq'][:, :a, 0], x_data_1['size_seq'][:, :a, 1]],1)

        # label_cluster = torch.eye(self.target_nums, dtype=torch.float32)
        repeats = 2
        base = torch.arange(0, self.target_nums)
        label_cluster = base.repeat(repeats)
        # label_cluster = torch.cat((base, base), dim=0)
        # label_cluster = base.repeat_interleave(repeats)


        return{
            'keypoints0': kpt0,
            'keypoints1': kpt1,
            # 'descriptors0': x_data_0['pos_seq'][:, :a, 0]/x_data_0['pos_seq'][:, :a, 1], # NOTEsp 用的除法
            # 'descriptors1': x_data_1['pos_seq'][:, :a, 0]/x_data_1['pos_seq'][:, :a, 1],
            'descriptors0': desc0,  # NOTEsp 用的除法
            'descriptors1': desc1,
            'rdm0': rdm0,
            'rdm1': rdm1,
            'trj0': x_data_0['pos_seq'][:, :a],#NOTEsp 用于画图
            'trj1': x_data_1['pos_seq'][:, :a],
            'all_matches': label_data.transpose(0, 1),
            'label_cluster': label_cluster,
            # 'file_name': file_name
        }


class NewCustomDataset_test(Dataset): #NOTEsp 读整个场景数据 demo_whole
    def __init__(self, data_path: str, model: str='test', target_length: int = 8, target_nums: int = 20, i_interval = 1):
        """

        :param data_path: 数据存放路径
        :param seq_length: 序列长度
        """
        assert "whole" in data_path, "the whole data path may not correct."
        self.data_path = data_path
        self.target_nums = target_nums
        self.model = model
        self.target_length = target_length
        self.uavname = ['uav1', 'uav2', 'uav3', 'uav4', 'uav5', 'uav6', 'uav7', 'uav8', 'uav9']
        self.i_interval = i_interval

        # if model == 'test':
        #     random.seed(1)

        all_data_files = os.listdir(data_path)
        self.all_data = []
        for file in all_data_files:
            with open(os.path.join(data_path, file), 'rb') as f:
                data = pickle.load(f)
                # if model == 'test':
                #     rdm0 = [random.randint(0, target_nums - 1) for _ in range(target_nums)]
                #     rdm1 = [random.randint(0, target_nums - 1) for _ in range(target_nums)]
                #     unique_vec0 = sorted(list(set(rdm0)))
                #     unique_vec1 = sorted(list(set(rdm1)))
                #     data['position'] = data['position'][:, unique_vec0, :, :]
                #     data['relative_velocity'] = data['relative_velocity'][:, unique_vec0, :, :]
                #     data['size_seq'] = data['position'][:, unique_vec0, :, :]
                    # desc0, kpts0, desc1, kpts1 = desc01[:, unique_vec0], kpts01[:, unique_vec0], desc11[:,
                    # a = 1
                self.all_data.append(data)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        cur_data = self.all_data[index]
        dataout = []

        for ii in range(9 - self.i_interval):
            cur_data_0 = cur_data[self.uavname[ii]]
            cur_data_1 = cur_data[self.uavname[ii+self.i_interval]]

            x_data_0 = dict(
                pos_seq=torch.tensor(cur_data_0['position'], dtype=torch.float32),
                vr_seq=torch.tensor(cur_data_0['relative_velocity'], dtype=torch.float32),
                size_seq=torch.tensor(cur_data_0['size_seq'], dtype=torch.float32),
            )
            x_data_1 = dict(
                pos_seq=torch.tensor(cur_data_1['position'], dtype=torch.float32),
                vr_seq=torch.tensor(cur_data_1['relative_velocity'], dtype=torch.float32),
                size_seq=torch.tensor(cur_data_1['size_seq'], dtype=torch.float32),
            )

            # label_id = torch.tensor(cur_data['label'], dtype=torch.int16)

            # for debugging
            for val1, val2 in zip(x_data_0.values(), x_data_1.values()):
                if torch.isnan(val1).any() or torch.isnan(val2).any():
                    raise ValueError

            label_data = torch.eye(self.target_nums, dtype=torch.float32)

            rdm0 = [random.randint(0, self.target_nums-1) for _ in range(self.target_nums)]
            rdm1 = [random.randint(0, self.target_nums-1) for _ in range(self.target_nums)]

            a = self.target_length #NOTEsp 轨迹的长度,为1则用其他信息
            kpt0 = x_data_0['pos_seq'][:, 0, :]
            kpt1 = x_data_1['pos_seq'][:, 0, :]

            if a > 1.5:
                desc0 = x_data_0['pos_seq'][:, :a, 0]/x_data_0['pos_seq'][:, :a, 1]  # NOTEsp 用的除法
                desc1 = x_data_1['pos_seq'][:, :a, 0]/x_data_1['pos_seq'][:, :a, 1]
            else:
                kpt0min = kpt0.min(0)[0]
                kpt0max = kpt0.max(0)[0]
                kpt1min = kpt1.min(0)[0]
                kpt1max = kpt1.max(0)[0]

                x0 = (kpt0 - kpt0min) / (kpt0max - kpt0min)
                x1 = (kpt1 - kpt1min) / (kpt1max - kpt1min)
                desc0 = torch.cat([x_data_0['pos_seq'][:, :a, 0] / x_data_0['pos_seq'][:, :a, 1], x_data_0['vr_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 1]],1)
                desc1 = torch.cat([x_data_1['pos_seq'][:, :a, 0] / x_data_1['pos_seq'][:, :a, 1], x_data_1['vr_seq'][:, :a, 0], x_data_1['size_seq'][:, :a, 0], x_data_1['size_seq'][:, :a, 1]],1)

            dataout.append({'keypoints0': kpt0,
            'keypoints1': kpt1,
            'descriptors0': desc0,  # NOTEsp 用的除法
            'descriptors1': desc1,
            'rdm0': rdm0,
            'rdm1': rdm1,
            'trj0': x_data_0['pos_seq'][:, :a],#NOTEsp 用于画图
            'trj1': x_data_1['pos_seq'][:, :a],
            'all_matches': label_data,
             })

        return dataout


class NewCustomDataset_cluster(Dataset): #NOTEsp 用于聚类的数据集，回头把其他数据读取的删了,无关的输入删除
    def __init__(self, data_path: str, model: str='test', use_other: bool = False, target_length: int = 8, target_nums: int = 20, i_interval = 1):
        """
        :param data_path: 数据存放路径
        :param seq_length: 序列长度
        """
        assert "whole" in data_path, "the whole data path may not correct."
        self.data_path = data_path
        self.target_nums = target_nums
        self.model = model
        self.use_other = use_other
        self.target_length = target_length
        self.uavname = ['uav1', 'uav2', 'uav3', 'uav4', 'uav5', 'uav6', 'uav7', 'uav8', 'uav9']
        # self.i_interval = i_interval
        self.all_data = []
        # all_data_files = os.listdir(data_path)#[:10]
        # 如果有数据就直接读取，如果没有就生成．
        if model == 'test':
            len = 1000
        else:
            len = 8998
        data_path_all = os.path.join('data', "all_data"+model+str(len)+".pkl")
        if os.path.exists(data_path_all):
            with open(data_path_all, 'rb') as f:
                self.all_data = pickle.load(f)
        # else:
        #     position_all = None
        #     pbar = tqdm.tqdm(total=len(all_data_files))
        #     for file in all_data_files:
        #         with open(os.path.join(data_path, file), 'rb') as f:
        #             data = pickle.load(f)
        #             for ii in range(len(self.uavname)):
        #                 if position_all is None:
        #                     position_all = data[self.uavname[ii]]['position']
        #                 else:
        #                     position_all = np.concatenate((position_all,data[self.uavname[ii]]['position']), axis=0)
        #             minmax = [np.min(position_all, axis=(0, 1)), np.max(position_all, axis=(0, 1))]# Notesp 归一化
        #             for ii in range(len(self.uavname)):
        #                 data[self.uavname[ii]]['position'] = (data[self.uavname[ii]]['position']-minmax[0])/(minmax[1]-minmax[0])
        #             self.all_data.append(data)
        #
        #         pbar.update(1)

            with open(data_path_all, 'wb') as f:
                pickle.dump(self.all_data, f, protocol=pickle.HIGHEST_PROTOCOL)

        # a=1
        # save_name = os.path.join(TRAIN_DATA_PATH, f"whole", data_mode, f"{dir_name}_data.pickle")
        # save_dict(self.all_data, save_name)

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        cur_data = self.all_data[index]
        dataout = []

        for ii in range(len(self.uavname)):
            cur_data_0 = cur_data[self.uavname[ii]]
            # cur_data_1 = cur_data[self.uavname[ii+self.i_interval]]

            x_data_0 = dict(
                pos_seq=torch.tensor(cur_data_0['position'], dtype=torch.float32),
                vr_seq=torch.tensor(cur_data_0['relative_velocity'], dtype=torch.float32),
                size_seq=torch.tensor(cur_data_0['size_seq'], dtype=torch.float32),
                RCS_seq=torch.tensor(cur_data_0['RCS_seq'], dtype=torch.float32),
            )

            for val1 in x_data_0.values():
                if torch.isnan(val1).any():
                    raise ValueError

            # repeats = 2
            label_cluster = torch.arange(0, self.target_nums)
            # label_cluster = base.repeat(repeats)

            #TODO 两种不同的随机数取的方式
            # rdm0 = [random.randint(0, self.target_nums-1) for _ in range(self.target_nums)]

            numbers = list(range(self.target_nums))
            to_remove = random.sample(numbers, int(self.target_nums*0.3))
            rdm0 = [num for num in numbers if num not in to_remove]

            a = self.target_length #NOTEsp 轨迹的长度,为1则用其他信息
            kpt0 = x_data_0['pos_seq'][:, 0, :]

            if a > 1.5:
                desc0 = torch.atan2(x_data_0['pos_seq'][:, :a, 0], x_data_0['pos_seq'][:, :a, 1])  # TODO 用　ａｔａｎ２
                if self.use_other:
                    desc0 = torch.stack([desc0, x_data_0['vr_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 0],
                                         x_data_0['RCS_seq'][:, :a, 0]], 1)

            else:
                # kpt0min = kpt0.min(0)[0]  # TODO 一个点的时候
                # kpt0max = kpt0.max(0)[0]
                desc_ = torch.atan2(x_data_0['pos_seq'][:, :a, 0], x_data_0['pos_seq'][:, :a, 1])
                # x0 = (kpt0 - kpt0min) / (kpt0max - kpt0min)
                desc0 = torch.cat([desc_, x_data_0['vr_seq'][:, :a, 0], x_data_0['size_seq'][:, :a, 0], x_data_0['RCS_seq'][:, :a, 0]], 1)

            dataout.append({'keypoints': kpt0,
            'descriptors': desc0,  # NOTEsp 用的除法
            'rdm': rdm0,
            'trj': x_data_0['pos_seq'][:, :a],#NOTEsp 用于画图
            'label_cluster': label_cluster,
            'label_cluster_rdm': torch.tensor(rdm0),
             })

        return dataout
