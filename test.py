import torch, os
from model.model import parsingNet
from utils.common import merge_config
from utils.dist_utils import dist_print
from evaluation.eval_wrapper import eval_lane
import yaml
import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='configs/culane.yaml')
    return parser.parse_args()



if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    with open(args.params) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(cfg['local_rank'])
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    dist_print('start testing...')
    assert cfg['network']['backbone'] in ['18','34','50','101','152','50next','101next','50wide','101wide']

    if cfg['dataset']['name'] == 'CULane':
        cls_num_per_lane = 18
    elif cfg['dataset']['name'] == 'Tusimple':
        cls_num_per_lane = 56
    else:
        raise NotImplementedError

    net = parsingNet(network=cfg['network'],datasets=cfg['dataset']).cuda()

    state_dict = torch.load(cfg['test']['test_model'], map_location = 'cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = False)

    # if distributed:
    #     net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])

    if not os.path.exists(cfg['test']['test_work_dir']):
        os.mkdir(cfg['test']['test_work_dir'])

    dataset = cfg['dataset']['name']
    data_root = cfg['dataset']['data_root']
    test_work_dir = cfg['test']['test_work_dir']
    griding_num = cfg['dataset']['griding_num']
    eval_lane(net, dataset, data_root, test_work_dir, griding_num, True, distributed)