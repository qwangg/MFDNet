import os
from config import Config
import torch

torch.backends.cudnn.benchmark = True
from SSIM import SSIM
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import time
import numpy as np
import utils
from data_RGB import get_training_data, get_validation_data
from MFDNet import HPCNet as mfdnet
import losses
from tqdm import tqdm

if __name__ == "__main__":
    opt = Config('training.yml')

    gpus = ','.join([str(i) for i in opt.GPU])
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpus
    torch.cuda.is_available()

    file_psnr = 'MFD_PSNR.txt'
    file_loss = 'MFD_LOSS.txt'
    # Set Seeds
    random.seed(1234)
    np.random.seed(1234)
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    start_epoch = 1
    session = 'MFDNet'

    result_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'results', session)
    model_dir = os.path.join(opt.TRAINING.SAVE_DIR, 'models', session)

    utils.mkdir(result_dir)
    utils.mkdir(model_dir)

    train_dir = opt.TRAINING.TRAIN_DIR
    val_dir = opt.TRAINING.VAL_DIR

    # Model
    model_restoration = mfdnet()
    model_restoration.cuda()

    device_ids = [i for i in range(torch.cuda.device_count())]
    if torch.cuda.device_count() > 0:
        print("\n\nLet's use", torch.cuda.device_count(), "GPUs!\n\n")

    new_lr = opt.OPTIM.LR_INITIAL

    optimizer = optim.Adam(model_restoration.parameters(), lr=new_lr, betas=(0.9, 0.999), eps=1e-8)

    # Scheduler
    scheduler = optim.lr_scheduler.StepLR(step_size=60, gamma=0.8, optimizer=optimizer)
    scheduler.step()

    # Resume
    if opt.TRAINING.RESUME:
        path_chk_rest = utils.get_last_path(model_dir, '_latest.pth')
        utils.load_checkpoint(model_restoration, path_chk_rest)
        start_epoch = utils.load_start_epoch(path_chk_rest) + 1
        utils.load_optim(optimizer, path_chk_rest)

        for i in range(1, start_epoch):
            scheduler.step()
        new_lr = scheduler.get_lr()[0]
        print('------------------------------------------------------------------------------')
        print("==> Resuming Training with learning rate:", new_lr)
        print('------------------------------------------------------------------------------')

    if len(device_ids) > 1:
        model_restoration = nn.DataParallel(model_restoration, device_ids=device_ids)

    # Loss
    criterion_char = losses.CharbonnierLoss()
    criterion_edge = losses.EdgeLoss()
    criterion_SSIM = SSIM()

    # DataLoaders
    train_dataset = get_training_data(train_dir, {'patch_size': opt.TRAINING.TRAIN_PS})
    train_loader = DataLoader(dataset=train_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=True,
                              num_workers=4, drop_last=False, pin_memory=True)

    val_dataset = get_validation_data(val_dir, {'patch_size': opt.TRAINING.VAL_PS})
    val_loader = DataLoader(dataset=val_dataset, batch_size=opt.OPTIM.BATCH_SIZE, shuffle=False,
                            num_workers=4, drop_last=False, pin_memory=True)

    print('===> Start Epoch {} End Epoch {}'.format(start_epoch, opt.OPTIM.NUM_EPOCHS + 1))
    print('===> Loading datasets')

    best_psnr = 0
    best_epoch = 0

    model_restoration.train().cuda()
    for epoch in range(start_epoch, opt.OPTIM.NUM_EPOCHS + 1):
        epoch_start_time = time.time()
        epoch_loss = 0
        SSIM_all = 0
        train_id = 1
        train_sample = 0
        # model_restoration.train()
        # model_restoration.train().cuda()
        for i, data in enumerate(tqdm(train_loader), 0):
            # zero_grad
            for param in model_restoration.parameters():
                param.grad = None

            target = data[0].cuda()
            input_ = data[1].cuda()
            criterion_char.cuda()
            criterion_edge.cuda()
            criterion_SSIM.cuda()

            restored = model_restoration(input_)
            restored[0] = restored[0].cuda()
            restored[1] = restored[1].cuda()
            # Compute loss at each stage
            loss_char0 = criterion_char(restored[0], target)
            loss_char1 = criterion_char(restored[1], input_)
            loss_edge0 = criterion_edge(restored[0], target)
            loss_edge1 = criterion_edge(restored[1], input_)
            loss_SSIM0 = criterion_SSIM(restored[0], target)
            loss_SSIM1 = criterion_SSIM(restored[1], input_)
            loss = 0.3 * (loss_char0 + 0.2 * loss_char1) + (0.2 * (loss_edge0)) - (
                    0.15 * (loss_SSIM0 + 0.2 * loss_SSIM1))  # 0.05,0.2
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            SSIM_all += loss_SSIM0.item()
            train_sample += 1
        SSIM = SSIM_all / train_sample
        # Evaluation
        if epoch % opt.TRAINING.VAL_AFTER_EVERY == 0:
            model_restoration.eval()
            psnr_val_rgb = []
            for ii, data_val in enumerate(val_loader, 0):
                target = data_val[0].cuda()
                input_ = data_val[1].cuda()

                with torch.no_grad():
                    restored = model_restoration(input_)
                restored = restored[0].cuda()

                for res, tar in zip(restored, target):
                    psnr_val_rgb.append(utils.torchPSNR(res, tar))

            psnr_val_rgb = torch.stack(psnr_val_rgb).mean().item()

            if psnr_val_rgb > best_psnr:
                best_psnr = psnr_val_rgb
                best_epoch = epoch
                torch.save({'epoch': epoch,
                            'state_dict': model_restoration.state_dict(),
                            'optimizer': optimizer.state_dict()
                            }, os.path.join(model_dir, "model_best.pth"))

            print(
                "[epoch %d PSNR: %.4f --- best_epoch %d Best_PSNR %.4f]" % (epoch, psnr_val_rgb, best_epoch, best_psnr))
            format_str1 = 'Epoch: %d, PSNR: %.4f, best_epoch: %d, Best_PSNR: %.4f'
            a = str(format_str1 % (epoch, psnr_val_rgb, best_epoch, best_psnr))
            PSNR_file = open(file_psnr, 'a+')
            PSNR_file.write(a)
            PSNR_file.write('\n')
            PSNR_file.close()
            torch.save({'epoch': epoch,
                        'state_dict': model_restoration.state_dict(),
                        'optimizer': optimizer.state_dict()
                        }, os.path.join(model_dir, f"model_epoch_{epoch}.pth"))

        scheduler.step()

        print("------------------------------------------------------------------")
        print("Epoch: {}\tTime: {:.4f}\tLoss: {:.4f}\tSSIM: {:.4f}\tLearningRate {:.8f}".format(epoch,
                                                                                                time.time() - epoch_start_time,
                                                                                                epoch_loss, SSIM,
                                                                                                scheduler.get_lr()[0]))
        print("------------------------------------------------------------------")
        format_str = 'Epoch: %d, Time: %.4f, Loss: %.4f, SSIM: %.4f, LearningRate: %.8f'
        a = str(format_str % (epoch, time.time() - epoch_start_time, epoch_loss, SSIM, scheduler.get_lr()[0]))
        loss_file = open(file_loss, 'a+')
        loss_file.write(a)
        loss_file.write('\n')
        loss_file.close()
        torch.save({'epoch': epoch,
                    'state_dict': model_restoration.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, os.path.join(model_dir, "model_latest"))
