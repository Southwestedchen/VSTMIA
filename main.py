import os
import subprocess
import time
import random
import argparse

time.sleep(random.uniform(0, 2))

def get_best_gpu():
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        best_gpu_id = "0"
        max_free_memory = -1
        for line in result.stdout.strip().split('\n'):
            if line:
                gpu_id, free_memory = line.split(',')
                gpu_id = gpu_id.strip()
                free_memory = int(free_memory.strip())
                if free_memory > max_free_memory:
                    max_free_memory = free_memory
                    best_gpu_id = gpu_id
        print(f"Auto-selected GPU: {best_gpu_id} with free memory: {max_free_memory} MB")
        return best_gpu_id
    except Exception as e:
        print(f"Failed to auto-detect GPU memory: {e}, defaulting to GPU 0")
        return "0"

os.environ['CUDA_VISIBLE_DEVICES'] = get_best_gpu()

import torch
from train import FederatedLearning
from data_processing import process_data
import logging
from utils import path_exists, set_seed
import trajectory
import attack

def init_logging(args):
    log_path='./log_file/' + args.model+'/'+args.dataset
    path_exists(log_path)
    logging.basicConfig(
        format='%(message)s',
        level=logging.INFO,
        handlers=[
            logging.FileHandler(log_path + '/' + args.log_name,mode='w'),
            logging.StreamHandler()
        ]
    )


def init_args():
    parser = argparse.ArgumentParser(description='VLMFLMIA parameters')
    parser.add_argument('--dataset', type=str, default='STL10', choices=['STL10', 'location'])
    parser.add_argument('--client_num', type=int, default=5)
    parser.add_argument('--data_split',type=str,default='uniform')
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'nn'])
    parser.add_argument('--save_path', type=str, default='./models')
    
    parser.add_argument('--random_client_mode',type=bool,default= False)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--optimizer', type=str, default='SGD')
    parser.add_argument('--training_round', type=int, default=200)
    parser.add_argument('--participant', type=int, default=5)
    parser.add_argument('--attacker_client_idx',type=int,default=0)
    parser.add_argument('--collusion_client_idx',type=int,nargs="+",default=[1,2])
    parser.add_argument('--save_client_model_idx',type=int,nargs="+",default=[0,1])
    parser.add_argument('--data_path', type=str, default='./datas')
    parser.add_argument('--model_path', type=str, default='./models_main')
    parser.add_argument('-alpha', type=float, default=0.2)
    
    parser.add_argument('--split_ratio', type=float, default=0.5)
    parser.add_argument('--random_seed', type=int, default=123)
    parser.add_argument('--log_name', type=str, default='train_models')
    parser.add_argument('--lr',type=float,default=0.01)
    parser.add_argument('--steplr',type=bool,default=False)
    parser.add_argument('--lr_gamma',type=float,default=0.99)
    parser.add_argument('--lr_step',type=int,default=1)
    parser.add_argument('--method',type=str,default='ours',choices=['ours'])

    parser.add_argument('--vlm_type', type=str, default='qwen3_2b',
                        choices=['qwen3_2b'])
    parser.add_argument('--vlm_path', type=str, default=None)

    parser.add_argument('--data_process_flag', type=bool, default=False)
    parser.add_argument('--train_model', type=bool,default=False)

    parser.add_argument('--regenerate_plots', default=False, action='store_true')
    parser.add_argument('--no_phase2', dest='phase2', action='store_false', default=True)
    parser.add_argument('--lambda_cap', type=float, default=1.0)
    parser.add_argument('--calib_threshold', type=float, default=0.1)
    parser.add_argument('--head_ratio', type=float, default=0.5)
    parser.add_argument('--tail_ratio', type=float, default=0.5)
    
    parser.add_argument('--plot_width', type=float, default=6.22)
    parser.add_argument('--plot_height', type=float, default=2.67)
    parser.add_argument('--plot_dpi', type=int, default=150)

    return parser.parse_args()


if __name__ == '__main__':
    args = init_args()
    set_seed(args.random_seed)
    init_logging(args)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.data_process_flag is True:
        process_data(args,download=True)
    if args.train_model is True:
        FL = FederatedLearning(args)
        FL.train_FL_models()
    if args.method == 'ours':
        if args.regenerate_plots:
            trajectory_generator = trajectory.ours(args=args,size=1000)
            trajectory_generator.make_loader_for_vlm()
        else:
            print("[Skip] Skipping trajectory image generation and using existing images for the attack.")
        attack.run_attack(dataset=args.dataset, model=args.model, max_samples=2000,
                        vlm_type=args.vlm_type, vlm_path=args.vlm_path,
                        enable_phase2=args.phase2,
                        lambda_cap=args.lambda_cap,
                        calib_threshold=args.calib_threshold,
                        head_ratio=args.head_ratio,
                        tail_ratio=args.tail_ratio)
    print(f'model:{args.model},dataset:{args.dataset}')
