import argparse

import cv2
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '4'

import torch

from backbones import get_model
from utils.utils_callbacks import CallBackVerification


@torch.no_grad()
def inference(weight, name, img):
    # if img is None:
    #     img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    # else:
    #     img = cv2.imread(img)
    #     img = cv2.resize(img, (112, 112))
    #
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = np.transpose(img, (2, 0, 1))
    # img = torch.from_numpy(img).unsqueeze(0).float()
    # img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    net = net.to('cuda')
    target = ['MFR1/u_u', 'MFR1/u_m', 'MFR1/m_m', 'MFR2/u_u', 'MFR2/u_m', 'MFR2/m_m']
    print(CallBackVerification(1, 0, target, '/home/hankyul/hdd_ext/face/test')(1, net))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='r50', help='backbone network')
    parser.add_argument('--weight', type=str, default='')
    parser.add_argument('--img', type=str, default=None)
    args = parser.parse_args()
    inference(args.weight, args.network, args.img)
