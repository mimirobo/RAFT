import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder


counter = 0
DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = np.stack((img,)*3, axis=-1)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))
 
    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]
        

def viz(img, flo , dir_name):
    global counter
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    #cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.imwrite(dir_name+'/img-{}.png'.format(counter), flo)
    print("Optical FLow {} saved in {}".format(counter, dir_name))
    counter = counter + 1
    #cv2.waitKey()


def demo(args):
    global counter
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(args.model))

    model = model.module
    model.to(DEVICE)
    model.eval()

    parent_dir = args.input_dir + "/*"
    all_sub_dirs = glob.glob(parent_dir)
    #Create Export Sub-Dirs
    for sub_dir_path in all_sub_dirs:
        output_dir_path = args.output_dir + "/" + os.path.basename(sub_dir_path)
        if os.path.exists(output_dir_path):
            continue
        os.mkdir(output_dir_path)
        counter = 0
        if len(os.listdir(sub_dir_path) ) == 0:
            continue
        with torch.no_grad():
            images = glob.glob(os.path.join(sub_dir_path, '*.png')) + \
                    glob.glob(os.path.join(sub_dir_path, '*.jpg')) + \
                    glob.glob(os.path.join(sub_dir_path, '*.tif'))

            images = load_image_list(images)
            for i in range(images.shape[0]-1):
                image1 = images[i,None]
                image2 = images[i+1,None]

                flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
                viz(image1, flow_up, output_dir_path)
        print("Sub-Dir {} done".format(os.path.basename(sub_dir_path)))
    print("Completed")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    #parser.add_argument('--path', help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--input_dir', help="input parent directory for groups of sub-dirs")
    parser.add_argument('--output_dir', help="output parent directory for optical flow results")
    args = parser.parse_args()

    demo(args)
