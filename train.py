import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

sys.path.append('./models')
import numpy as np
from datetime import datetime
# from models.CPNet import CPNet
#from models.VFMamba_model import VFMambaNet
from models.VCPDCN import VFMambaNet
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
from loss import IOU, SSIM, SimMaxLoss, SimMinLoss, SimMaxLoss_Patch

maxloss = SimMaxLoss(metric='cos', alpha=0.25).cuda()
minloss = SimMinLoss(metric='cos').cuda()
maxloss_two = SimMaxLoss_Patch(metric='cos', alpha=0.25).cuda()


def get_coef(iter_percentage=1, method="cos", milestones=(0, 1)):
    min_point, max_point = min(milestones), max(milestones)
    min_coef, max_coef = 0, 1

    ual_coef = 1.0
    if iter_percentage < min_point:
        ual_coef = min_coef
    elif iter_percentage > max_point:
        ual_coef = max_coef
    else:
        if method == "linear":
            ratio = (max_coef - min_coef) / (max_point - min_point)
            ual_coef = ratio * (iter_percentage - min_point)
        elif method == "cos":
            perc = (iter_percentage - min_point) / (max_point - min_point)
            normalized_coef = (1 - np.cos(perc * np.pi)) / 2
            ual_coef = normalized_coef * (max_coef - min_coef) + min_coef
    return ual_coef


def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()


def Mask_Loss(pred, target, reduction='mean'):
    gts = target
    gts_b = 1 - target
    gts_b = gts_b.cuda()

    pr1 = pred[0]
    pr2 = pred[1]

    pd1 = pred[2]
    pd2 = pred[3]

    b, c, h, w = pr1.shape

    gts = F.interpolate(gts, size=(h, w), mode='bilinear')
    gts_b = F.interpolate(gts_b, size=(h, w), mode='bilinear')

    # 先对输出做归一化处理
    pr1 = torch.sigmoid(pr1)
    pr2 = torch.sigmoid(pr2)
    pd1 = torch.sigmoid(pd1)
    pd2 = torch.sigmoid(pd2)

    # BCE LOSS
    bce_loss = nn.BCELoss()
    losses = bce_loss(pr1, gts) + bce_loss(pr2, gts_b) + bce_loss(pd1, gts) + bce_loss(pd2, gts_b)

    return losses


def Hybrid_Loss(pred, target, reduction='mean'):
    # 先对输出做归一化处理
    pred = torch.sigmoid(pred)

    # BCE LOSS
    bce_loss = nn.BCELoss()
    bce_out = bce_loss(pred, target)

    # IOU LOSS
    iou_loss = IOU(reduction=reduction)
    iou_out = iou_loss(pred, target)

    # SSIM LOSS
    ssim_loss = SSIM(window_size=11)
    ssim_out = ssim_loss(pred, target)

    # hybrid_loss = [bce_out, iou_out, ssim_out]
    losses = bce_out + iou_out + ssim_out

    return losses


def cross_entropy2d_edge(input, target, reduction='mean'):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


if opt.gpu_id == '0':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    print('USE GPU 0')
elif opt.gpu_id == '1':
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    print('USE GPU 1')
elif opt.gpu_id == '2':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    print('USE GPU 2')
elif opt.gpu_id == '3':
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    print('USE GPU 3')
cudnn.benchmark = True

image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root
edge_root = opt.edge_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
# test_texture_root = opt.test_texture_root
save_path = opt.save_path

logging.basicConfig(filename=save_path + 'VCPMCN.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("CPNet-Train")
model = VFMambaNet(type='VFMamba-S')

finetuning_pth = opt.fintuning
if finetuning_pth is not None:
    model.load_state_dict(torch.load(finetuning_pth), strict=False)

num_parms = 0
#if (opt.load_pre is not None):
#    model.load_state_dict(torch.load(opt.load_pre))
#    print('load model from ', opt.load_pre)

# if (opt.load_pre is not None):
#    model.load_pre(opt.load)
#    print('load model from ', opt.load)

model.cuda()
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

# set the path

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load data
print('load data...')
train_loader = get_loader(image_root, gt_root, depth_root, edge_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root, test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, save_path,
        opt.decay_epoch))

# set loss function
CE = torch.nn.BCEWithLogitsLoss()
ECE = torch.nn.BCELoss()
grad_loss_func = torch.nn.MSELoss()
step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()

    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth, edge) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            gts_b = 1 - gts
            gts_b = gts_b.cuda()
            edge = edge.cuda()
            depth = depth.cuda()

            pre1, pre2, pre3, loss_align = model(images, depth)

            hybrid_loss = Hybrid_Loss(pre1, gts) + Hybrid_Loss(pre2, gts) + Hybrid_Loss(pre3, gts)

            loss =  0.3 * hybrid_loss + 0.1 * loss_align

            loss.backward()

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}||sal_loss:{:4f} '.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], loss.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f},  sal_loss:{:4f} , mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'],
                               loss.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = pre1[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 5 == 0:
            torch.save(model.state_dict(), save_path + 'network_epoch_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'network_epoch_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


def bce2d_new(input, target, reduction=None):
    assert (input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg / num_total
    beta = 1.1 * num_pos / num_total
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


# test function
def test(test_loader, model, epoch, save_path):
    global best_mae, best_epoch
    model.eval()
    with torch.no_grad():
        mae_sum = 0
        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.cuda()

            results, _, _, _ = model(image, depth)
            res = results
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), save_path + 'network_epoch_best.pth')
                print('best epoch:{}'.format(epoch))
        logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)

        train(train_loader, model, optimizer, epoch, save_path)
        if epoch > 0:
            test(test_loader, model, epoch, save_path)
