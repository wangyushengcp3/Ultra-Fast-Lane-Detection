import torch, os, datetime
import numpy as np

from model.model import parsingNet
from data.dataloader import get_train_loader,get_test_loader

from utils.dist_utils import dist_tqdm
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import save_model, Logger
from evaluation.eval_wrapper import generate_lines, call_culane_eval
import time
import argparse
import yaml

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--params", type=str, default='data/params.yaml')
    parser.add_argument('--local_rank', default=-1, type=int,
                        help='node rank for distributed training')
    return parser.parse_args()

def inference(net, data_label, use_aux):
    if use_aux:
        img, cls_label, seg_label = data_label
        img, cls_label, seg_label = img.cuda(), cls_label.long().cuda(), seg_label.long().cuda()
        cls_out, seg_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label, 'seg_out':seg_out, 'seg_label': seg_label}
    else:
        img, cls_label = data_label
        img, cls_label = img.cuda(), cls_label.long().cuda()
        cls_out = net(img)
        return {'cls_out': cls_out, 'cls_label': cls_label}

def resolve_val_data(results, use_aux):
    results['cls_out'] = torch.argmax(results['cls_out'], dim=1)
    if use_aux:
        results['seg_out'] = torch.argmax(results['seg_out'], dim=1)
    return results

def calc_loss(loss_dict, results, logger, global_step):
    loss = 0
    for i in range(len(loss_dict['name'])):
        data_src = loss_dict['data_src'][i]
        datas = [results[src] for src in data_src]
        loss_cur = loss_dict['op'][i](*datas)
        if global_step % 20 == 0:
            logger.scalar_summary('loss/'+loss_dict['name'][i], 'train', loss_cur, global_step)
        loss += loss_cur * loss_dict['weight'][i]
    return loss

def train(net, data_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, use_aux, local_rank):
    net.train()
    if local_rank != -1:
        data_loader.sampler.set_epoch(epoch)
    progress_bar = dist_tqdm(data_loader)
    t_data_0 = time.time()
    for b_idx, data_label in enumerate(progress_bar):
        t_data_1 = time.time()
        reset_metrics(metric_dict)
        global_step = epoch * len(data_loader) + b_idx

        t_net_0 = time.time()
        results = inference(net, data_label, use_aux)

        loss = calc_loss(loss_dict, results, logger, global_step)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(global_step)
        t_net_1 = time.time()

        results = resolve_val_data(results, use_aux)

        update_metrics(metric_dict, results)
        if global_step % 20 == 0:
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                logger.scalar_summary('metric/' + me_name,'train',  me_op.get(), global_step)
        logger.scalar_summary('meta/lr', 'train', optimizer.param_groups[0]['lr'], global_step)

        if hasattr(progress_bar,'set_postfix'):
            kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in zip(metric_dict['name'], metric_dict['op'])}
            log_msg = 'Epoch{}/{}|Iter{}'.format(epoch, scheduler.total_epoch,b_idx)
            # log_msg = 'Epoch{}/{}|Iter{} '.format(epoch, scheduler.total_epoch,
            #         global_step, b_idx, len(data_loader), optimizer.param_groups[0]['lr'])
            progress_bar.set_description(log_msg)
            progress_bar.set_postfix(loss = '%.3f' % float(loss), 
                                    **kwargs)
        t_data_0 = time.time()

def test(net, data_loader, dataset, work_dir, logger, use_aux=True):
    output_path = os.path.join(work_dir, 'culane_eval_tmp')
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    net.eval()
    if dataset['name'] == 'CULane':
        for i, data in enumerate(dist_tqdm(data_loader)):
            imgs, names = data
            imgs = imgs.cuda()
            with torch.no_grad():
                out = net(imgs)
            if len(out) == 2 and use_aux:
                out, seg_out = out

            generate_lines(out,imgs[0,0].shape,names,output_path,dataset['griding_num'],localization_type = 'rel',flip_updown = True)
        res = call_culane_eval(dataset['data_root'], 'culane_eval_tmp', work_dir)
        TP,FP,FN = 0,0,0
        for k, v in res.items():
            val = float(v['Fmeasure']) if 'nan' not in v['Fmeasure'] else 0
            val_tp,val_fp,val_fn = int(v['tp']),int(v['fp']),int(v['fn'])
            TP += val_tp
            FP += val_fp
            FN += val_fn
            logger.log('k:{} val{}'.format(k,val))
        P = TP * 1.0/(TP + FP)
        R = TP * 1.0/(TP + FN)
        F = 2*P*R/(P + R)
        logger.log('F:{}'.format(F))
        return F
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    args = parse_args()
    with open(args.params) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # data dict
    logger = Logger(args.local_rank, cfg['log_path'])
    logger.log('start training')
    assert cfg['network']['backbone'] in ['resnet_18','34','mobilenetv2']

    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
    w, h = cfg['dataset']['w'],cfg['dataset']['h']
    net = parsingNet(network=cfg['network'],datasets=cfg['dataset']).cuda()
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids = [args.local_rank])
    # try:
    #     from thop import profile
    #     macs, params = profile(net, inputs=(torch.zeros(1, 3, h, w).to(device)))
    #     ms = 'FLOPs:  %.2f GFLOPS, Params: %.2f M'%(params/ 1E9, params/ 1E6)
    # except:
    #     ms = 'Model profile error'
    # logger.log(ms)
    train_loader = get_train_loader(cfg['dataset'], args.local_rank)
    test_loader = get_test_loader(cfg['dataset'], args.local_rank)
    optimizer = get_optimizer(net, cfg['train'])

    if cfg['finetune'] is not None:
        state_all = torch.load(cfg['finetune'])['model']
        state_clip = {}  # only use backbone parameters
        for k,v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)
    if  cfg['resume'] is not None:
        logger.log('==> Resume model from ' + cfg['resume'])
        resume_dict = torch.load(cfg['resume'], map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg['resume'])[1][2:5]) + 1
    else:
        resume_epoch = 0
    scheduler = get_scheduler(optimizer, cfg['train'], len(train_loader))
    logger.log('Train Datasets Totoal: %d'%len(train_loader))
    metric_dict = get_metric_dict(cfg['dataset'])
    loss_dict = get_loss_dict(cfg)
    
    max_F = 0
    for epoch in range(resume_epoch, cfg['train']['epoch']):
        train(net, train_loader, loss_dict, optimizer, scheduler,logger, epoch, metric_dict, cfg['dataset']['use_aux'], args.local_rank)
        save_model(net, optimizer, epoch)
        if cfg['test']['val_intervals'] > 0 and epoch % cfg['test']['val_intervals'] == 0:
            F = test(net, test_loader, cfg['dataset'], cfg['log_path'], logger)
            if F > max_F:
                save_model(net, optimizer, epoch, 'model_best.pth')
                max_F = F