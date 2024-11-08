"""
## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881
"""

import numpy as np
import os
import argparse
from tqdm import tqdm
import torch.nn.functional as F
import torch.nn as nn
import torch

from skimage import img_as_ubyte
# from basicsr.models.archs.restormer_arch import Restormer
from basicsr.models.archs.adair_attnv1_myfiltersmallv3_sca_arch import AdaIR_s3
import cv2
import utils
from natsort import natsorted
from glob import glob
from pdb import set_trace as stx

import lpips

alex = lpips.LPIPS(net='alex').cuda()

parser = argparse.ArgumentParser(description='Single Image Defocus Deblurring using Restormer')

parser.add_argument('--input_dir', default='./Datasets/test/DPDD', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/Single_Image_Defocus_Deblurring/', type=str,
                    help='Directory for results')
parser.add_argument('--weights', default='./pretrained_models/net_g_276000.pth', type=str, help='Path to weights')
parser.add_argument('--save_images', action='store_true', help='Save denoised images in result directory')

args = parser.parse_args()

####### Load yaml #######
yaml_file = 'Options/DefocusDeblur_Single_8bit_Restormer.yml'
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

model_restoration = AdaIR_s3(**x['network_g'])

checkpoint = torch.load(args.weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

result_dir = args.result_dir
if args.save_images:
    os.makedirs(result_dir, exist_ok=True)

filesI = natsorted(glob(os.path.join(args.input_dir, 'inputC', '*.png')))
filesC = natsorted(glob(os.path.join(args.input_dir, 'target', '*.png')))

indoor_labels = np.load('./Datasets/test/DPDD/indoor_labels.npy')
outdoor_labels = np.load('./Datasets/test/DPDD/outdoor_labels.npy')


def pad(input_,h,w):
    factor = 8
    # Padding in case images are not multiples of 8
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    ####         add   myself        #############
    h, w = input_.shape[2], input_.shape[3]
    max_dim = max(h, w)
    pad_h = max_dim - h if h < max_dim else 0
    pad_w = max_dim - w if w < max_dim else 0
    input_ = F.pad(input_, (0, pad_w + factor, 0, pad_h + factor), 'reflect')

    return input_
    #####       add end  ###########


psnr, mae, ssim, pips = [], [], [], []
with torch.no_grad():
    for fileI, fileC in tqdm(zip(filesI, filesC), total=len(filesC)):

        imgI = np.float32(utils.load_img(fileI)) / 255.
        imgC = np.float32(utils.load_img(fileC)) / 255.

        h,w = imgI.shape[0], imgI.shape[1]

        patchI = torch.from_numpy(imgI).unsqueeze(0).permute(0, 3, 1, 2).cuda()
        patchC = torch.from_numpy(imgC).unsqueeze(0).permute(0, 3, 1, 2).cuda()

        print(patchI.shape)

        patchI = pad(patchI,h,w)
        patchC = pad(patchC,h,w)

        restored = model_restoration(patchI)
        restored = torch.clamp(restored, 0, 1)
        pips.append(alex(patchC, restored, normalize=True).item())


        # Unpad images to original dimensions
        restored = restored[:, :, :h, :w]

        restored = restored.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        print(restored.shape)


        psnr.append(utils.PSNR(imgC, restored))
        mae.append(utils.MAE(imgC, restored))
        ssim.append(utils.SSIM(imgC, restored))
        if args.save_images:
            save_file = os.path.join(result_dir, os.path.split(fileC)[-1])
            restored = np.uint8((restored * 255).round())
            utils.save_img(save_file, restored)

psnr, mae, ssim, pips = np.array(psnr), np.array(mae), np.array(ssim), np.array(pips)

psnr_indoor, mae_indoor, ssim_indoor, pips_indoor = psnr[indoor_labels - 1], mae[indoor_labels - 1], ssim[
    indoor_labels - 1], pips[indoor_labels - 1]
psnr_outdoor, mae_outdoor, ssim_outdoor, pips_outdoor = psnr[outdoor_labels - 1], mae[outdoor_labels - 1], ssim[
    outdoor_labels - 1], pips[outdoor_labels - 1]

print("Overall: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr), np.mean(ssim), np.mean(mae),
                                                                    np.mean(pips)))
print("Indoor:  PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_indoor), np.mean(ssim_indoor),
                                                                    np.mean(mae_indoor), np.mean(pips_indoor)))
print("Outdoor: PSNR {:4f} SSIM {:4f} MAE {:4f} LPIPS {:4f}".format(np.mean(psnr_outdoor), np.mean(ssim_outdoor),
                                                                    np.mean(mae_outdoor), np.mean(pips_outdoor)))
