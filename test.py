import os
import argparse
from tqdm import tqdm
import time
import torch
from torch.utils.data import DataLoader
import utils
from data_RGB import get_test_data
from MFDNet import HPCNet as mfdnet
from skimage import img_as_ubyte


parser = argparse.ArgumentParser(description='Image Deraining using MPRNet')

parser.add_argument('--input_dir', default='./Datasets/test/', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='./results/', type=str, help='Directory for results')
parser.add_argument('--weights', default='./checkpoints/checkpoints_mfd.pth', type=str,  # './tip_rebuttal_checkpoints/MFD_loss4_latest420.pth'
                    help='Path to weights')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

model_restoration = mfdnet()
utils.load_checkpoint(model_restoration, args.weights)
print("===>Testing using weights: ", args.weights)

# model_restoration.cuda()
# model_restoration.eval()
# model_restoration.to(device)
model_restoration.eval().cuda()

datasets = ['Test100']

for dataset in datasets:
    rgb_dir_test = os.path.join(args.input_dir, dataset, 'input')
    test_dataset = get_test_data(rgb_dir_test, img_options={})
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False,
                             drop_last=False, pin_memory=True)  # num_workers=4,

    result_dir = os.path.join(args.result_dir, dataset)
    utils.mkdir(result_dir)

    all_time = 0
    count = 0
    with torch.no_grad():
        for ii, data_test in enumerate(tqdm(test_loader), 0):
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()

            input_ = data_test[0].cuda()
            filenames = data_test[1]
            b, c, h, w = input_.shape
            st_time = time.time()
            restored = model_restoration(input_)
            ed_time = time.time()
            cost_time = ed_time - st_time
            all_time += cost_time
            count += 1

            restored = torch.clamp(restored[0], 0, 1)
            restored = restored.permute(0, 2, 3, 1).cpu().detach().numpy()

            for batch in range(len(restored)):
                restored_img = img_as_ubyte(restored[batch])
                utils.save_img((os.path.join(result_dir, filenames[batch] + '.png')), restored_img)

    print('spent {} s.'.format(all_time))
    print('spent {} s per item.'.format(all_time / count))
