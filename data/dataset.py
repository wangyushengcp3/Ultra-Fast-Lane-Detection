import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos
import os

def loader_func(path):
    return Image.open(path)


class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane
    def __getitem__(self, index):
        name = self.list[index].split()[0]
        
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)

        return img, name

    def __len__(self):
        return len(self.list)


class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict, img_transform = None,target_transform = None,simu_transform = None,segment_transform=None,load_name=False):
        super(LaneClsDataset, self).__init__()
        self.data_dict = data_dict

        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform

        self.load_name = load_name

        list_path = os.path.join(data_dict['data_root'], 'list/train_gt.txt')
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.row_anchor = data_dict['row_anchor']
        self.row_anchor.sort()

    def __getitem__(self, index):
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]
        
        label_path = os.path.join(self.data_dict['data_root'], label_name)
        label = loader_func(label_path)

        img_path = os.path.join(self.data_dict['data_root'], img_name)
        img = loader_func(img_path)

        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)
        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors



        w, h = img.size
        cls_label = self._grid_pts(lane_pts, self.data_dict['griding_num'], w)
        # make the coordinates to classification label
        if self.data_dict['use_aux']:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)
        if self.img_transform is not None:
            img = self.img_transform(img)

        if self.data_dict['use_aux']:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane,n,2
        num_lane, n, n2 = pts.shape# 4 18 2
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane)) # 18， 4
        for i in range(num_lane):
            pti = pts[i, :, 1]
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
        return to_pts.astype(int)

    def _get_index(self, label):
        w, h = label.size
        if h != self.data_dict['input_size'][0]:
            scale_f = lambda x : int((x * 1.0/self.data_dict['input_size'][0]) * h)
            sample_tmp = list(map(scale_f,self.row_anchor))
        # 288 inx [121, 131, 141, 150, 160, 170, 180, 189, 199, 209, 219, 228, 238, 248, 258, 267, 277, 287]
        # 590 inx [247, 268, 288, 307, 327, 348, 368, 387, 407, 428, 448, 467, 487, 508, 528, 546, 567, 587]
        all_idx = np.zeros((self.data_dict['num_lanes'],len(sample_tmp),2)) #4*18*2 第一个通道存放590 对应的anchor 第二个通道存放车道线位置。如果不存在车道线位置为-1
        for i,r in enumerate(sample_tmp):
            label_r = np.asarray(label)[int(round(r))]
            for lane_idx in range(1, self.data_dict['num_lanes'] + 1):
                # culane label 四条车道线像素分别喂1 2 3 4
                pos = np.where(label_r == lane_idx)[0]
                if len(pos) == 0:
                    all_idx[lane_idx - 1, i, 0] = r
                    all_idx[lane_idx - 1, i, 1] = -1
                    continue
                pos = np.mean(pos)
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos

        # data augmentation: extend the lane to the boundary of image

        all_idx_cp = all_idx.copy()
        for i in range(self.data_dict['num_lanes']):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane
            valid = all_idx_cp[i,:,1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] == all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)#多项式拟合
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
