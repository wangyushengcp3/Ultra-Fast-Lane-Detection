import torch, os
import numpy as np

import torchvision.transforms as transforms
import data.mytransforms as mytransforms
from data.constant import tusimple_row_anchor, culane_row_anchor
from data.dataset import LaneClsDataset, LaneTestDataset

def get_train_loader(dataset_dict, local_rank):
    target_transform = transforms.Compose([
        mytransforms.FreeScaleMask((dataset_dict['h'], dataset_dict['w'])),
        mytransforms.MaskToTensor(),
    ])
    segment_transform = transforms.Compose([
        mytransforms.FreeScaleMask((dataset_dict['h'] // 8, dataset_dict['w'] // 8)),#36 100
        mytransforms.MaskToTensor(),
    ])
    img_transform = transforms.Compose([
        transforms.Resize((dataset_dict['h'], dataset_dict['w'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    simu_transform = mytransforms.Compose2([
        mytransforms.RandomRotate(6),
        mytransforms.RandomUDoffsetLABEL(100),
        mytransforms.RandomLROffsetLABEL(200)
    ])
    if dataset_dict['name'] == 'CULane':
        train_dataset = LaneClsDataset(dataset_dict, img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform, segment_transform=segment_transform)
    elif dataset_dict['name'] == 'Tusimple':
        train_dataset = LaneClsDataset(dataset_dict['data_root'],
                                           os.path.join(dataset_dict['data_root'], 'train_gt.txt'),
                                           img_transform=img_transform, target_transform=target_transform,
                                           simu_transform = simu_transform,
                                           griding_num=dataset_dict['griding_num'], 
                                           row_anchor = dataset_dict['row_anchor'],
                                           segment_transform=segment_transform,use_aux=dataset_dict['use_aux'], num_lanes = dataset_dict['num_lanes'])
    else:
        raise NotImplementedError

    if local_rank == -1:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_dict['batch_size'], 
                                                    shuffle=True, num_workers=dataset_dict['num_workers'])
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        torch.distributed.init_process_group(backend='nccl')
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dataset_dict['batch_size'], sampler=sampler, num_workers=dataset_dict['num_workers'])
    return train_loader

def get_test_loader(dataset_dict, local_rank):
    img_transforms = transforms.Compose([
        transforms.Resize((dataset_dict['h'], dataset_dict['w'])),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    if dataset_dict['name'] == 'CULane':
        test_dataset = LaneTestDataset(dataset_dict['data_root'],os.path.join(dataset_dict['data_root'], 'list/test.txt'),img_transform = img_transforms)
        cls_num_per_lane = 18
    elif dataset_dict['name'] == 'Tusimple':
        test_dataset = LaneTestDataset(dataset_dict['data_root'],os.path.join(dataset_dict['data_root'], 'test.txt'), img_transform = img_transforms)
        cls_num_per_lane = 56

    if local_rank == -1:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_dict['batch_size'], 
                                                    shuffle=False, num_workers=dataset_dict['num_workers'])
    else:
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(local_rank % num_gpus)
        torch.distributed.init_process_group(backend='nccl')
        sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dataset_dict['batch_size'], sampler=sampler, num_workers=dataset_dict['num_workers'])
    return test_loader

class SeqDistributedSampler(torch.utils.data.distributed.DistributedSampler):
    '''
    Change the behavior of DistributedSampler to sequential distributed sampling.
    The sequential sampling helps the stability of multi-thread testing, which needs multi-thread file io.
    Without sequentially sampling, the file io on thread may interfere other threads.
    '''
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False):
        super().__init__(dataset, num_replicas, rank, shuffle)
    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size


        num_per_rank = int(self.total_size // self.num_replicas)

        # sequential sampling
        indices = indices[num_per_rank * self.rank : num_per_rank * (self.rank + 1)]

        assert len(indices) == self.num_samples

        return iter(indices)