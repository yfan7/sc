import argparse
import os
from ssl import OP_NO_TLSv1
import nibabel as nib
# from visdom import Visdom
# viz = Visdom(port=8850)
import sys
import random
sys.path.append("..")
sys.path.append(".")
import numpy as np
import time
import torch as th
from PIL import Image
import torch.distributed as dist
from guided_diffusion import dist_util, logger
import torchvision.utils as vutils
from guided_diffusion.utils import staple
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
import torchvision.transforms as transforms
from torchsummary import summary
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import SimpleITK as sitk
from scripts.lesions_eval import *
from guided_diffusion.scloader import SCDataset

seed=10
th.manual_seed(seed)
th.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
from itertools import islice


def prep_for_testing(config):
    savefolder_weights = os.path.join(config["savefolder_root"], f'{config["fold_name"]}')
    if not os.path.exists(savefolder_weights):
        print(f'folder to evaluate: {savefolder_weights} does not exist!!!')
    visuals_save = os.path.join(savefolder_weights,f'visuals_step_{config["load_epoch"]}_test')
    if not os.path.exists(visuals_save):
        os.makedirs(visuals_save)
    return savefolder_weights,visuals_save


def visualize(img):
    _min = img.min()
    _max = img.max()
    normalized_img = (img - _min)/ (_max - _min)
    return normalized_img


def eval(config,savefolder_weights,visuals_save):
    
    args = create_argparser(savefolder_weights,config).parse_args()
    args.diffusion_steps = 2000
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    if args.data_name =='sc':
        args.in_ch = 6
        args.out_ch = 6
        logger.log("creating model and diffusion...")

        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        all_images = []
        state_dict = dist_util.load_state_dict(args.model_path, map_location=lambda storage, loc: storage.cuda(config["gpu_device"]))
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # name = k[7:] # remove `module.`
            if 'module.' in k:
                new_state_dict[k[7:]] = v
                # load params
            else:
                new_state_dict = state_dict

        model.load_state_dict(new_state_dict)

        model.to('cuda')
        if args.use_fp16:
            model.convert_to_fp16()
        model.eval()
        ds = SCDataset(test_flag=True)
        datal = th.utils.data.DataLoader(
                ds,
                batch_size=args.batch_size,
                shuffle=False)
        data = iter(datal)
        for _ in islice(data, 16):
            pass
        index = 17
        for sample in data:
            masked_img,img,mask = sample
            B,C,H,W = img.shape
            c =  th.randn(B,C,H,W)
            all_img = th.cat((masked_img,c),dim=1)
            logger.log("sampling...")
            start = th.cuda.Event(enable_timing=True)
            end = th.cuda.Event(enable_timing=True)
            enslist = []
            sample_list = []
            cal_list = []
            interm_list = []
            for i in range(args.num_ensemble):  #this is for the generation of an ensemble of 5 masks.
                model_kwargs = {}
                start.record()
                sample_fn = (
                    diffusion.p_sample_loop_known if not args.use_ddim else diffusion.ddim_sample_loop_known
                )
                sample, x_noisy, org, cal, cal_out, intermediates = sample_fn(
                    model,
                    (args.batch_size, 3, 256, 256), all_img,None,None,None,
                    step = args.diffusion_steps,
                    clip_denoised=args.clip_denoised,
                    save_interm = True,
                    model_kwargs=model_kwargs,
                )
                end.record()
                th.cuda.synchronize()
                print('time for 1 sample', start.elapsed_time(end))  #time measurement for the generation of 1 sample
                ca = th.tensor(cal)
                samp = th.tensor(sample)
                co = th.tensor(cal_out)
                del sample, cal, cal_out
                enslist.append(co)
                sample_list.append(samp)
                cal_list.append(ca)
                interm_list.append(intermediates)

            ensres = staple(th.stack(enslist,dim=0))
            sampres = staple(th.stack(sample_list,dim=0))
            calres = staple(th.stack(cal_list,dim=0))
            for i in range(B):
                pred = ensres.detach().cpu().numpy().squeeze(0)[i,...]
                print('pred shape',pred.shape)
                partial = masked_img.detach().cpu().numpy()[i,...]
                print('partial shape',partial.shape)
                gt = img.detach().cpu().numpy()[i,...]
                print('gt shape',gt.shape)
                this_mask = mask.detach().cpu().numpy()
                this_mask = np.concatenate((this_mask, this_mask,this_mask),axis = 0)
                print('this_mask shape',this_mask.shape)
                all_visuals = np.stack((pred,gt,partial,this_mask),axis = 0)
                np.savez(os.path.join(visuals_save,f'{index}.npz'), my_array=all_visuals)
                index += 1
            if intermediates is not None:
                indices = list(range(0, 50, 5))
                np_intermediates = []
                for i in indices:
                    interm = intermediates[i]
                    interm = interm.detach().cpu().numpy().squeeze()
                    np_intermediates.append(interm)
                np.savez(os.path.join(visuals_save,'interm.npz'), my_array=all_visuals)


def create_argparser(savefolder_weights,config):
    defaults = dict(
        data_name = 'sc',
        clip_denoised=True,
        num_samples=1,
        batch_size=6,
        dims = config['train_dims'],
        use_ddim=False,
        model_path=os.path.join(savefolder_weights,f'savedmodel{config["load_epoch"]}.pt'),
        num_ensemble=5,      #number of samples in the ensemble
        gpu_dev = str(config['gpu_device']),
        out_dir= os.path.join(savefolder_weights,'results/'),
        multi_gpu = None, #"0,1,2"
        debug = False,
        learn_sigma= True,
        attention_resolutions = "16,8",
        diffusion_steps = 1000,
        num_res_blocks=2,
        num_heads=1,
        image_size = 32,
        num_channels=64,
        sc = config['sc']
    )

    model_defaults = model_and_diffusion_defaults()
    # print('model_defaults',model_defaults)
    model_defaults.update(defaults)
    # print(model_defaults)
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, model_defaults)
    return parser


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./eval_configs_all.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    th.cuda.empty_cache()
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    # use_device = config['gpu_device']
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(use_device)
    # th.cuda.set_device(use_device)
    # print(th.cuda.current_device())
    savefolder_weights,visuals_save = prep_for_testing(config)
    eval(config,savefolder_weights,visuals_save)