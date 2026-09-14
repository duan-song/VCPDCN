import torch
import torch.nn.functional as F
import sys
import numpy as np
import os, argparse
import cv2
from models.VCPDCN import VFMambaNet
from data import test_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='0', help='select gpu id')
parser.add_argument('--test_path',type=str,default='',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

#set device for test
if opt.gpu_id=='0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id=='1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id == '2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')
elif opt.gpu_id == '3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')

#load the model
model = VFMambaNet(type='VFMamba-S')
pth = 'weight path'
model.load_state_dict(torch.load(pth), strict=False)
model.cuda()
model.eval()
print("load the training weight from {}".format(pth))


test_datasets = ['CAMO', 'CHAMELEON', 'COD10K', 'NC4K']

for dataset in test_datasets:
    save_path = './test_maps/' + 'VMamba/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/Depth/'

    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    print(len(test_loader))
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.cuda()
#        depth = depth.repeat(1,3,1,1).cuda()
        results, _, _, _ = model(image, depth)
        res = results
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        cv2.imwrite(save_path + name, res*255)
        print('save img to: ',save_path+name)
    print('Test Done!')
