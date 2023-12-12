
import sys
import argparse
sys.path.append("..")
sys.path.append(".")
from guided_diffusion import dist_util, logger
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
import torch as th
from guided_diffusion.train_util import TrainLoop
from visdom import Visdom
viz = Visdom(port=8850)
import torchvision.transforms as transforms
import yaml
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import json
import GPUtil
from guided_diffusion.scloader import SCDataset
def prep_for_training(config):
    savefolder_weights = os.path.join(config["savefolder_root"], f'{config["fold_name"]}')
    if not os.path.exists(savefolder_weights):
        os.makedirs(savefolder_weights)

    return savefolder_weights


def main(config,savefolder_weights = None):
    args = create_argparser(savefolder_weights,config).parse_args()
    args.diffusion_steps = 1000
    dist_util.setup_dist(args)
    logger.configure(dir = args.out_dir)

    logger.log("creating data loader...")
    if args.data_name =='sc':
        ds = SCDataset(test_flag=False)
        args.in_ch = 6
        args.out_ch = 6
        val_ds_wrapper = None

    datal= th.utils.data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True)
    data = iter(datal)
    

    logger.log("creating model and diffusion...")
    
    
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )


    param_size = 0
    buffer_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()

    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2
    print('Size: {:.3f} fMB'.format(size_all_mb))

    GPUtil.showUtilization()

    if args.multi_gpu:
        model = th.nn.DataParallel(model,device_ids=[int(id) for id in args.multi_gpu.split(',')])
        # model.to(device = th.device('cuda', int(args.gpu_dev)))
        model.to('cuda')
    else:
        model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion,  maxt=args.diffusion_steps)

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        classifier=None,
        data=data,
        dataloader=datal,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        sc = args.sc
    ).run_loop()


def create_argparser(savefolder_weights,config):
    defaults = dict(
        data_name = 'sc',
        schedule_sampler="uniform",
        lr=1e-4,
        dims = config['train_dims'],
        loss_weight= config['loss_weight'],
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=8,
        microbatch=2,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=100,
        # save_interval=5000,
        save_interval=1000,
        # resume_checkpoint=os.path.join(savefolder_weights,'savedmodel181000.pt'), #'"./results/pretrainedmodel.pt",
        resume_checkpoint = '',
        use_fp16=False,
        fp16_scale_growth=1e-3,
        gpu_dev = str(config['gpu_device']),
        multi_gpu =False,
        out_dir=savefolder_weights,
        attention_resolutions = "16,8",
        num_res_blocks=2,
        num_heads=1,
        image_size = 32,
        num_channels = 64,
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
    parser.add_argument('--config', type=str, default='./training_configs_all.yaml')
    args = parser.parse_args()
    with open(args.config, 'r') as stream:
        config = yaml.safe_load(stream)
    th.cuda.empty_cache()
    device = 'cuda' if th.cuda.is_available() else 'cpu'
    use_device = config['gpu_device']
    os.environ["CUDA_VISIBLE_DEVICES"] = str(use_device)
    th.cuda.set_device(use_device)
    th.cuda.current_device()
    training_data,testing_data,case_id2cohort,savefolder_weights,path_dict,id_names_thresed,save_val_stats_path = prep_for_training(config)
    main(config, training_data,testing_data,case_id2cohort,savefolder_weights,path_dict,save_val_stats_path)




