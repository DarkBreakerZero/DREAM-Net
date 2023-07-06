import os
from datasets import npz_proj_img_reader_func
import scipy
import numpy as np
import torch.nn
from torch.utils.data import DataLoader
from torch.backends import cudnn
import time
import torch.optim as optim
import argparse

from utils import recon_ops
from models import *
from utils import *
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def train(train_loader, model, sparse_rate, criterion, optimizer, scheduler, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.train()
    end = time.time()

    for data in train_loader:

        hdProj = data["hdproj"]

        mask = np.zeros(hdProj.size(), np.float32)
        mask = torch.FloatTensor(mask)
        mask[:, :, ::sparse_rate, :] = 1

        ldProj = data["ldproj"]
        ldProj = F.upsample(ldProj[:, :, ::sparse_rate, :], size=(ldProj.size(2), ldProj.size(3)), mode='bilinear') * (1-mask) + ldProj * mask
        hdCT = data["hdct"]
        ldCT = data["ldct"]

        hdProj = hdProj.cuda()
        ldProj = ldProj.cuda()
        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()
        mask = mask.cuda()

        if epoch > 10:
            ldProj, hdProj, ldCT, hdCT = MixUpDualDomain_AUG().aug(ldProj, hdProj, ldCT, hdCT)

        proj_net, img_net = model(ldProj, ldCT, mask)

        loss1 = 20 * F.l1_loss(proj_net, hdProj)
        loss2 = F.mse_loss(img_net, hdCT)
        loss = loss1 + loss2

        losses.update(loss2.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    writer.add_scalars('losses_img', {'train_mae_loss': losses.avg}, epoch + 1)
    writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], epoch + 1)

    writer.add_image('train img/label-fbp-result img', normalization(torch.cat([hdCT[0, :, :, :], img_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('train img/label-result-reproj proj', normalization(torch.cat([hdProj[0, :, :, :], proj_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('train img/residual img', normalization(torch.abs(hdCT[0, :, :, :] - img_net[0, :, :, :])), epoch + 1)
    writer.add_image('train img/residual proj', normalization(torch.abs(hdProj[0, :, :, :] - proj_net[0, :, :, :])), epoch + 1)

    scheduler.step()

    print('Train Epoch: {}\t train_mae_loss: {:.6f}\t'.format(epoch + 1, losses.avg))

def valid(valid_loader, model, sparse_rate, criterion, writer, epoch):

    batch_time = AverageMeter()
    losses = AverageMeter()
    model.eval()
    end = time.time()

    for data in valid_loader:

        hdProj = data["hdproj"]

        mask = np.zeros(hdProj.size(), np.float32)
        mask = torch.FloatTensor(mask)
        mask[:, :, ::sparse_rate, :] = 1

        ldProj = data["ldproj"]
        ldProj = F.upsample(ldProj[:, :, ::sparse_rate, :], size=(ldProj.size(2), ldProj.size(3)), mode='bilinear') * (1-mask) + ldProj * mask
        hdCT = data["hdct"]
        ldCT = data["ldct"]

        hdProj = hdProj.cuda()
        ldProj = ldProj.cuda()
        hdCT = hdCT.cuda()
        ldCT = ldCT.cuda()
        mask = mask.cuda()

        with torch.no_grad():

            proj_net, img_net = model(ldProj, ldCT, mask)

            loss1 = 20 * F.l1_loss(proj_net, hdProj)
            loss2 = F.mse_loss(img_net, hdCT)

        losses.update(loss2.item(), hdCT.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

    writer.add_scalars('losses_img', {'valid_mae_loss': losses.avg}, epoch+1)
    writer.add_scalars('losses_proj', {'valid_proj_loss': loss1.item()}, epoch+1)
    writer.add_image('valid img/label-result img', normalization(torch.cat([hdCT[0, :, :, :], img_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/label-result proj', normalization(torch.cat([hdProj[0, :, :, :], proj_net[0, :, :, :]], 2)), epoch + 1)
    writer.add_image('valid img/residual img', normalization(torch.abs(hdCT[0, :, :, :] - img_net[0, :, :, :])), epoch + 1)
    writer.add_image('valid img/residual proj', normalization(torch.abs(hdProj[0, :, :, :] - proj_net[0, :, :, :])), epoch + 1)

    print('Valid Epoch: {}\t valid_mae_loss: {:.6f}\t'.format(epoch + 1, losses.avg))

if __name__ == "__main__":

    cudnn.benchmark = True

    batch_size = 1
    views = 576
    sparse_rate = 8

    method = 'DreamNet_V' + str(views//sparse_rate)
    result_path = './runs/' + method + '/logs/'
    save_dir = './runs/' + method + '/checkpoints/'

    # Get dataset
    train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./train_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    # train_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./valid_clear_list_s1e6.txt')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16, shuffle=True)

    valid_dataset = npz_proj_img_reader_func.npz_proj_img_reader(paired_data_txt='./valid_clear_list_s1e6_v' + str(views//sparse_rate) + '.txt')
    valid_loader = DataLoader(valid_dataset, batch_size=1, num_workers=16, shuffle=True)

    angles = np.linspace(0, 2*np.pi, views, endpoint=False)
    op_example = recon_ops(angles=angles)

    model = DreamNetDense(op_example, 3, 32)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

    if os.path.exists(save_dir) is False:

        model = model.cuda()

    else:
        checkpoint_latest = torch.load(find_lastest_file(save_dir))
        model = load_model(model, checkpoint_latest).cuda()
        optimizer.load_state_dict(checkpoint_latest['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint_latest['lr_scheduler'])
        print('Latest checkpoint {0} loaded.'.format(find_lastest_file(save_dir)))

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    log_dir = os.path.join(result_path, time_str)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)

    print("*"*20 + "Start Train" + "*"*20)

    for epoch in range(0, 100):

        print("*" * 20 + "Epoch: " + str(epoch + 1).rjust(4, '0') + "*" * 20)

        train(train_loader, model, sparse_rate, criterion, optimizer, scheduler, writer, epoch)
        valid(valid_loader, model, sparse_rate, criterion, writer, epoch)

        save_model(model, optimizer, epoch + 1, save_dir, scheduler)