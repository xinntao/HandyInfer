import argparse
import cv2
import torch
import torch.nn.functional as F

from handyinfer.saliency_detection import init_saliency_detection_model
from handyinfer.utils import tensor2img_fast


def main(args):
    # initialize model
    sod_net = init_saliency_detection_model(args.model_name)

    img = cv2.imread(args.img_path)
    with torch.no_grad():
        pred = sod_net(img)
        pred = F.interpolate(pred, img.shape[0:2], mode='bilinear', align_corners=False)
        pred = tensor2img_fast(pred)

    # save img
    if args.save_path is not None:
        cv2.imwrite(args.save_path, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='inference/data/jump_cat.png')
    parser.add_argument('--save_path', type=str, default='result_saliency_detection.png')
    parser.add_argument('--model_name', type=str, default='inspyrenet')
    parser.add_argument('--half', action='store_true')
    args = parser.parse_args()

    main(args)
