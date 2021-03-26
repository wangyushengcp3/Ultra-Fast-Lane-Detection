import torch, os, cv2
from model.model import parsingNet
import torch
import scipy.special, tqdm
import numpy as np
import argparse
import yaml
def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--params', default = 'configs/culane.yaml', type = str)
    parser.add_argument('--batch_size', default = 1, type = int)
    parser.add_argument('--weights', default = 'weights/20210309_141417_lr_0.1/ep014.pth', type = str)
    parser.add_argument('--img-size', nargs='+', type=int, default=[256, 512], help='image size')  # height, width
    return parser

if __name__ == '__main__':
    args = get_args().parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand

    with open(args.params) as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # data dict

    net = parsingNet(network=cfg['network'],datasets=cfg['dataset']).cuda()

    state_dict = torch.load(args.weights, map_location='cpu')['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict=False)
    net.eval()
    print('val done!!!')

    img = torch.zeros(args.batch_size, 3, *args.img_size)  # image size(1,3,320,192) iDetection
    img = img.cuda()
    with torch.no_grad():
        out = net(img)
    # ONNX export
    try:
        import onnx
        from onnxsim import simplify

        print('\nStarting ONNX export with onnx %s...' % onnx.__version__)
        f = args.weights.replace('.pth', '.onnx')  # filename
        torch.onnx.export(net, img, f, verbose=False, opset_version=11, input_names=['images'],
                          output_names=['output'])

        # Checks
        onnx_model = onnx.load(f)  # load onnx model
        model_simp, check = simplify(onnx_model)
        assert check, "Simplified ONNX model could not be validated"
        onnx.save(model_simp, f)
        print(onnx.helper.printable_graph(onnx_model.graph))  # print a human readable model
        print('ONNX export success, saved as %s' % f)
    except Exception as e:
        print('ONNX export failure: %s' % e)
