import os, argparse
from utils.dist_utils import is_main_process, dist_print, DistSummaryWriter
from utils.config import Config
import torch
import logging
from termcolor import colored
from evaluation.eval_wrapper import eval_lane as el
class Logger:
    def __init__(self, local_rank, save_dir='./', use_tensorboard=True):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        self.rank = local_rank
        fmt = colored('[%(name)s]', 'magenta', attrs=['bold']) + colored('[%(asctime)s]', 'blue') + \
              colored('%(levelname)s:', 'green') + colored('%(message)s', 'white')
        logging.basicConfig(level=logging.INFO,
                            filename=os.path.join(save_dir, 'logs.log'),
                            datefmt='%A, %d %B %Y %H:%M:%S',  # 指定日期时间格式
                            format='%(asctime)s[line:%(lineno)d] %(levelname)s %(message)s',
                            filemode='w')
        self.log_dir = os.path.join(save_dir, 'logs')
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        formatter = logging.Formatter(fmt, datefmt="%m-%d %H:%M:%S")
        console.setFormatter(formatter)
        logging.getLogger().addHandler(console)
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError:
                raise ImportError(
                    'Please run "pip install future tensorboard" to install '
                    'the dependencies to use torch.utils.tensorboard '
                    '(applicable to PyTorch 1.1 or higher)')
            if self.rank < 1:
                logging.info('Using Tensorboard, logs will be saved in {}'.format(self.log_dir))
                self.writer = SummaryWriter(log_dir=self.log_dir)

    def log(self, string):
        if self.rank < 1:
            logging.info(string)

    def scalar_summary(self, tag, phase, value, step):
        if self.rank < 1:
            self.writer.add_scalars(tag, {phase: value}, step)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('config', help = 'path to config file')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--dataset', default = None, type = str)
    parser.add_argument('--data_root', default = None, type = str)
    parser.add_argument('--epoch', default = None, type = int)
    parser.add_argument('--batch_size', default = None, type = int)
    parser.add_argument('--optimizer', default = None, type = str)
    parser.add_argument('--learning_rate', default = None, type = float)
    parser.add_argument('--weight_decay', default = None, type = float)
    parser.add_argument('--momentum', default = None, type = float)
    parser.add_argument('--scheduler', default = None, type = str)
    parser.add_argument('--steps', default = None, type = int, nargs='+')
    parser.add_argument('--gamma', default = None, type = float)
    parser.add_argument('--warmup', default = None, type = str)
    parser.add_argument('--warmup_iters', default = None, type = int)
    parser.add_argument('--backbone', default = None, type = str)
    parser.add_argument('--griding_num', default = None, type = int)
    parser.add_argument('--use_aux', default = None, type = str2bool)
    parser.add_argument('--sim_loss_w', default = None, type = float)
    parser.add_argument('--shp_loss_w', default = None, type = float)
    parser.add_argument('--note', default = None, type = str)
    parser.add_argument('--log_path', default = None, type = str)
    parser.add_argument('--finetune', default = None, type = str)
    parser.add_argument('--resume', default = None, type = str)
    parser.add_argument('--test_model', default = None, type = str)
    parser.add_argument('--test_work_dir', default = None, type = str)
    parser.add_argument('--num_lanes', default = None, type = int)
    return parser


def set_logging(rank=-1):
    logging.basicConfig(format="%(message)s", level=logging.INFO if rank in [-1, 0] else logging.WARN)

def merge_config():
    args = get_args().parse_args()
    cfg = Config.fromfile(args.config)

    items = ['dataset','data_root','epoch','batch_size','optimizer','learning_rate',
    'weight_decay','momentum','scheduler','steps','gamma','warmup','warmup_iters',
    'use_aux','griding_num','backbone','sim_loss_w','shp_loss_w','note','log_path',
    'finetune','resume', 'test_model','test_work_dir', 'num_lanes']
    for item in items:
        if getattr(args, item) is not None:
            dist_print('merge ', item, ' config')
            setattr(cfg, item, getattr(args, item))
    return args, cfg


def save_model(model, optimizer, epoch, path='model_last.pth'):
    model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()

    state = {'model': model_state_dict, 'optimizer': optimizer.state_dict(), 'epoch': epoch}
    # state = {'model': model_state_dict}
    #assert os.path.exists(save_path)
    torch.save(state, path)

def eval_lane(net, dataset, data_root, test_work_dir, griding_num):
    el(net, dataset, data_root, test_work_dir, griding_num, True)

import pathspec

def cp_projects(to_path):
    if is_main_process():
        with open('./.gitignore','r') as fp:
            ign = fp.read()
        ign += '\n.git'
        spec = pathspec.PathSpec.from_lines(pathspec.patterns.GitWildMatchPattern, ign.splitlines())
        all_files = {os.path.join(root,name) for root,dirs,files in os.walk('./') for name in files}
        matches = spec.match_files(all_files)
        matches = set(matches)
        to_cp_files = all_files - matches
        # to_cp_files = [f[2:] for f in to_cp_files]
        # pdb.set_trace()
        for f in to_cp_files:
            dirs = os.path.join(to_path,'code',os.path.split(f[2:])[0])
            if not os.path.exists(dirs):
                os.makedirs(dirs)
            os.system('cp %s %s'%(f,os.path.join(to_path,'code',f[2:])))


import datetime, os
def get_work_dir(cfg):
    now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    hyper_param_str = '_lr_{}'.format(cfg['train']['learning_rate'])
    work_dir = os.path.join(cfg['log_path'], now + hyper_param_str)
    return work_dir

# def get_logger(work_dir, cfg):
#     logger = DistSummaryWriter(work_dir)
#     config_txt = os.path.join(work_dir, 'cfg.txt')
#     if is_main_process():
#         with open(config_txt, 'w') as fp:
#             fp.write(str(cfg))
#     return logger
