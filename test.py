# -*- coding: utf-8 -*-

import time
import os
import math
import argparse
from glob import glob
from collections import OrderedDict
import random
import warnings
from datetime import datetime

import cv2
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from skimage.io import imread, imsave

from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms

from dataset import Dataset

import denseUnet
from metrics import dice_coef, batch_iou, mean_iou, iou_score ,ppv,sensitivity
import losses
from utils import str2bool, count_params
import joblib
from hausdorff import hausdorff_distance
import imageio
# from torchsummary import summary

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="jiu0Monkey_Dense_Unet_woDS",
                        help='model name')
    parser.add_argument('--mode', default=None,
                        help='GetPicture or Calculate')

    args = parser.parse_args()

    return args


def main():
    val_args = parse_args()

    args = joblib.load('models/%s/args.pkl'%val_args.name)
    

    if not os.path.exists('output/%s' %args.name):
        os.makedirs('output/%s' %args.name)

    print('Config -----')
    for arg in vars(args):
        print('%s: %s' %(arg, getattr(args, arg)))
    print('------------')

    joblib.dump(args, 'models/%s/args.pkl' %args.name)

    # create model
    print("=> creating model %s" %args.arch)
    model = denseUnet.__dict__[args.arch](args)

    # model = model.cuda()

    # Data loading code
    img_paths = glob(r'C:\Users\NIT\Desktop\DenseUnet_BraTs\BraTS2Dpreprocessing\testImage\*')
    mask_paths = glob(r'C:\Users\NIT\Desktop\DenseUnet_BraTs\BraTS2Dpreprocessing\testMask\*')

    val_img_paths = img_paths
    val_mask_paths = mask_paths

    #train_img_paths, val_img_paths, train_mask_paths, val_mask_paths = \
    #   train_test_split(img_paths, mask_paths, test_size=0.2, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %args.name))
    model.eval()

    val_dataset = Dataset(args, val_img_paths, val_mask_paths)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False)

    if val_args.mode == "GetPicture":

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')

            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    # input = input.cuda()
                    # #target = target.cuda()
                    
                    # compute output
                    if args.deepsupervision:
                        output = model(input)[-1]
                    else:
                        output = model(input)
                        
                    #print("img_paths[i]:%s" % img_paths[i])
                    output = torch.sigmoid(output).data.cpu().numpy()
                    print(output)
                    # color_img = cv2.resize(output,(224,224))
                    # cv2.imshow("output",color_img)
                    # cv2.waitKey(0)
                    img_paths = val_img_paths[args.batch_size*i:args.batch_size*(i+1)]
                    #print("output_shape:%s"%str(output.shape))


                    for i in range(output.shape[0]):
                        """
                        wtName = os.path.basename(img_paths[i])
                        overNum = wtName.find(".npy")
                        wtName = wtName[0:overNum]
                        wtName = wtName + "_WT" + ".png"
                        imsave('output/%s/'%args.name + wtName, (output[i,0,:,:]*255).astype('uint8'))
                        tcName = os.path.basename(img_paths[i])
                        overNum = tcName.find(".npy")
                        tcName = tcName[0:overNum]
                        tcName = tcName + "_TC" + ".png"
                        imsave('output/%s/'%args.name + tcName, (output[i,1,:,:]*255).astype('uint8'))
                        etName = os.path.basename(img_paths[i])
                        overNum = etName.find(".npy")
                        etName = etName[0:overNum]
                        etName = etName + "_ET" + ".png"
                        imsave('output/%s/'%args.name + etName, (output[i,2,:,:]*255).astype('uint8'))
                        """
                        npName = os.path.basename(img_paths[i])
                        overNum = npName.find(".npy")
                        rgbName = npName[0:overNum]
                        rgbName = rgbName  + ".png"
                        rgbPic = np.zeros([160, 160, 3], dtype=np.uint8)
                        for idx in range(output.shape[2]):
                            for idy in range(output.shape[3]):
                                if output[i,0,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 0
                                    rgbPic[idx, idy, 1] = 128
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,1,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 0
                                    rgbPic[idx, idy, 2] = 0
                                if output[i,2,idx,idy] > 0.5:
                                    rgbPic[idx, idy, 0] = 255
                                    rgbPic[idx, idy, 1] = 255
                                    rgbPic[idx, idy, 2] = 0
                        imsave('output/%s/'%args.name + rgbName,rgbPic)

            # torch.cuda.empty_cache()
       
        print("Saving GT,numpy to picture")
        val_gt_path = 'output/%s/'%args.name + "GT/"
        if not os.path.exists(val_gt_path):
            os.mkdir(val_gt_path)
        for idx in tqdm(range(len(val_mask_paths))):
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            rgbName = name + ".png"

            npmask = np.load(mask_path)

            GtColor = np.zeros([npmask.shape[0],npmask.shape[1],3], dtype=np.uint8)
            for idx in range(npmask.shape[0]):
                for idy in range(npmask.shape[1]):
                   
                    if npmask[idx, idy] == 1:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 0
                        GtColor[idx, idy, 2] = 0
                   
                    elif npmask[idx, idy] == 2:
                        GtColor[idx, idy, 0] = 0
                        GtColor[idx, idy, 1] = 128
                        GtColor[idx, idy, 2] = 0
                    
                    elif npmask[idx, idy] == 4:
                        GtColor[idx, idy, 0] = 255
                        GtColor[idx, idy, 1] = 255
                        GtColor[idx, idy, 2] = 0

            #imsave(val_gt_path + rgbName, GtColor)
            imageio.imwrite(val_gt_path + rgbName, GtColor)
            """
            mask_path = val_mask_paths[idx]
            name = os.path.basename(mask_path)
            overNum = name.find(".npy")
            name = name[0:overNum]
            wtName = name + "_WT" + ".png"
            tcName = name + "_TC" + ".png"
            etName = name + "_ET" + ".png"

            npmask = np.load(mask_path)

            WT_Label = npmask.copy()
            WT_Label[npmask == 1] = 1.
            WT_Label[npmask == 2] = 1.
            WT_Label[npmask == 4] = 1.
            TC_Label = npmask.copy()
            TC_Label[npmask == 1] = 1.
            TC_Label[npmask == 2] = 0.
            TC_Label[npmask == 4] = 1.
            ET_Label = npmask.copy()
            ET_Label[npmask == 1] = 0.
            ET_Label[npmask == 2] = 0.
            ET_Label[npmask == 4] = 1.

            imsave(val_gt_path + wtName, (WT_Label * 255).astype('uint8'))
            imsave(val_gt_path + tcName, (TC_Label * 255).astype('uint8'))
            imsave(val_gt_path + etName, (ET_Label * 255).astype('uint8'))
            """
        print("Done!")



    if val_args.mode == "Calculate":

        wt_dices = []
        tc_dices = []
        et_dices = []
        wt_sensitivities = []
        tc_sensitivities = []
        et_sensitivities = []
        wt_ppvs = []
        tc_ppvs = []
        et_ppvs = []
        wt_Hausdorf = []
        tc_Hausdorf = []
        et_Hausdorf = []

        wtMaskList = []
        tcMaskList = []
        etMaskList = []
        wtPbList = []
        tcPbList = []
        etPbList = []

        maskPath = glob("output/%s/" % args.name + "GT\*.png")
        pbPath = glob("output/%s/" % args.name + "*.png")
        if len(maskPath) == 0:
            print("No mask found!")
            return

        for myi in tqdm(range(len(maskPath))):
            mask = imread(maskPath[myi])
            pb = imread(pbPath[myi])

            wtmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            wtpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            tcmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            tcpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            etmaskregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)
            etpbregion = np.zeros([mask.shape[0], mask.shape[1]], dtype=np.float32)

            for idx in range(mask.shape[0]):
                for idy in range(mask.shape[1]):
                    
                    if mask[idx, idy, :].any() != 0:
                        wtmaskregion[idx, idy] = 1
                    if pb[idx, idy, :].any() != 0:
                        wtpbregion[idx, idy] = 1
                    
                    if mask[idx, idy, 0] == 255:
                        tcmaskregion[idx, idy] = 1
                    if pb[idx, idy, 0] == 255:
                        tcpbregion[idx, idy] = 1
                    
                    if mask[idx, idy, 1] == 128:
                        etmaskregion[idx, idy] = 1
                    if pb[idx, idy, 1] == 128:
                        etpbregion[idx, idy] = 1
           
            dice = dice_coef(wtpbregion,wtmaskregion)
            wt_dices.append(dice)
            ppv_n = ppv(wtpbregion, wtmaskregion)
            wt_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(wtmaskregion, wtpbregion)
            wt_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(wtpbregion, wtmaskregion)
            wt_sensitivities.append(sensitivity_n)
      
            dice = dice_coef(tcpbregion, tcmaskregion)
            tc_dices.append(dice)
            ppv_n = ppv(tcpbregion, tcmaskregion)
            tc_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(tcmaskregion, tcpbregion)
            tc_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(tcpbregion, tcmaskregion)
            tc_sensitivities.append(sensitivity_n)
   
            dice = dice_coef(etpbregion, etmaskregion)
            et_dices.append(dice)
            ppv_n = ppv(etpbregion, etmaskregion)
            et_ppvs.append(ppv_n)
            Hausdorff = hausdorff_distance(etmaskregion, etpbregion)
            et_Hausdorf.append(Hausdorff)
            sensitivity_n = sensitivity(etpbregion, etmaskregion)
            et_sensitivities.append(sensitivity_n)

        print('WT Dice: %.4f' % np.mean(wt_dices))
        print('TC Dice: %.4f' % np.mean(tc_dices))
        print('ET Dice: %.4f' % np.mean(et_dices))
        print("=============")
        print('WT PPV: %.4f' % np.mean(wt_ppvs))
        print('TC PPV: %.4f' % np.mean(tc_ppvs))
        print('ET PPV: %.4f' % np.mean(et_ppvs))
        print("=============")
        print('WT sensitivity: %.4f' % np.mean(wt_sensitivities))
        print('TC sensitivity: %.4f' % np.mean(tc_sensitivities))
        print('ET sensitivity: %.4f' % np.mean(et_sensitivities))
        print("=============")
        print('WT Hausdorff: %.4f' % np.mean(wt_Hausdorf))
        print('TC Hausdorff: %.4f' % np.mean(tc_Hausdorf))
        print('ET Hausdorff: %.4f' % np.mean(et_Hausdorf))
        print("=============")
        print_iou_score()


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(np.asarray(img, dtype=np.float32))
    return images


def iou_score(output, target):
    # smooth = 1e-5

    output_ = output > 0.5
    target_ = target > 0.5
    intersection = (output_ & target_).sum()
    union = (output_ | target_).sum()

    return (intersection ) / (union )

def print_iou_score():
    

    gt = load_images_from_folder(r"C:\Users\NIT\Desktop\DenseUnet_BraTs\output\jiu0Monkey_Dense_Unet_woDS\GT")
    pred = load_images_from_folder(r"C:\Users\NIT\Desktop\DenseUnet_BraTs\output\jiu0Monkey_Dense_Unet_woDS\pred")

    totalScrore = 0
    for i in range(len(gt)):
        gtCurr = gt[i]
        predCurr = pred[i]
        score = "{:.4f}".format(iou_score(gtCurr,predCurr))
        totalScrore += float(score)
        print(str(i+1)+" : "+score)
    
    print("Mean Score = "+str(totalScrore/float(len(gt))))
    


if __name__ == '__main__':
    main()
