import argparse
import cv2
import torch

from handyinfer.depth_estimation import init_depth_estimation_model
from handyinfer.utils import img2tensor
from handyinfer.visualization import vis_depth_estimation


def main(args):
    device = torch.device('cuda')
    depth_net = init_depth_estimation_model(args.model_name)

    img = cv2.imread(args.img_path)
    img = img2tensor(img) / 255.
    img = img.to(device).unsqueeze(0)

    with torch.no_grad():
        pred = depth_net.infer(img)
        pred = vis_depth_estimation(pred)

    # save img
    if args.save_path is not None:
        cv2.imwrite(args.save_path, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_path', type=str, default='inference/data/test_depth_estimation.jpg')
    parser.add_argument('--save_path', type=str, default='result_depth_estimate.png')
    parser.add_argument('--model_name', type=str, default='ZoeD_N')
    args = parser.parse_args()

    main(args)
