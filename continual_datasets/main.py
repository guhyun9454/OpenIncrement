import sys
import argparse 
import datetime
import random
import numpy as np
import time
import torch
import importlib
import json
import os


from pathlib import Path

from timm.scheduler import create_scheduler
from timm.optim import create_optimizer

from continual_datasets.build_incremental_scenario import build_continual_dataloader
from continual_datasets.dataset_utils import set_data_config, get_ood_dataset, get_dataset, find_tasks_with_unseen_classes, UnknownWrapper
import models #여기서 models.py의 @register_model이 실행되고, timm의 모델 레지스트리에 등록, create_model를 통해 custom vit가 호출됨
import utils
import os
from utils import seed_everything

import warnings
warnings.filterwarnings('ignore', 'Argument interpolation should be of type InterpolationMode instead of int')
warnings.filterwarnings("ignore","The given NumPy array is not writable, and PyTorch does not support non-writable tensors")


def main(args):
    args = set_data_config(args)
    device = torch.device(args.device)



    # args를 save 폴더에 저장
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        args_dict = vars(args)
        args_path = os.path.join(args.save, 'args.json')
        with open(args_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        print(f"Arguments saved to {args_path}")

    seed_everything(args.seed)
    
    data_loader, class_mask, domain_list = build_continual_dataloader(args)
    if args.ood_dataset:
        data_loader[-1]['ood'] = get_ood_dataset(args.ood_dataset, args)

    if args.develop_tasks: return

    try:
        engine_module = importlib.import_module(f"engines.{args.method}")
    except ImportError:
        raise ValueError(f"Unknown engine type: {args.method}")
    Engine = engine_module.Engine
    model = engine_module.load_model(args)
    model.to(args.device)
    engine = Engine(model=model, device=args.device, class_mask=class_mask, domain_list=domain_list, args=args)
    
    print(args)
    args.wandb = False
    if args.wandb_run and args.wandb_project:
        import wandb
        import getpass

        args.wandb = True
        wandb.init(entity="OODVIL",project=args.wandb_project, name=args.wandb_run, config=args)
        wandb.config.update({"username": getpass.getuser()})
    
    if args.eval or args.ood_eval:
        print(f"{'Evaluation Only':=^60}")
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))

        for task_id in range(args.num_tasks):
            checkpoint_path = os.path.join(args.save, 'checkpoint/task{}_checkpoint.pth'.format(task_id+1))
            # 각 엔진에 맞는 체크포인트 로드 로직을 사용
            model = engine.load_checkpoint(model, checkpoint_path)
            
            _ = engine.evaluate_till_now(model, data_loader, device, task_id, class_mask, acc_matrix, args = args)
            if args.ood_dataset and args.ood_eval:
                print(f"{f'Task {task_id+1} OOD Evaluation':=^60}")
                ood_start = time.time()
                # 현재 태스크까지의 ID 데이터셋만 사용
                all_id_datasets = torch.utils.data.ConcatDataset([data_loader[t]['val'].dataset for t in range(task_id+1)])
                ood_loader = data_loader[-1]['ood']
                engine.evaluate_ood(model, all_id_datasets, ood_loader, device, args, task_id)
                ood_duration = time.time() - ood_start
                print(f"Task {task_id+1} OOD evaluation completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")
        return

        
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    optimizer = create_optimizer(args, model)

    if args.sched != 'constant':
        lr_scheduler, _ = create_scheduler(args, optimizer)
    elif args.sched == 'constant':
        lr_scheduler = None

    criterion = torch.nn.CrossEntropyLoss().to(device)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    engine.train_and_evaluate(model,criterion, data_loader, optimizer, 
                       lr_scheduler, device, class_mask, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    if args.wandb:
        import wandb
        wandb.config.update({"runtime": total_time_str})
    print(f"Total training time: {total_time_str}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser('OOD-VIL')
    
    parser.add_argument('--batch-size', default=24, type=int, help='Batch size per device')
    parser.add_argument('--epochs', default=5, type=int)

    # Model parameters
    parser.add_argument('--method', default='ICON', type=str, help='Engine type to use (e.g., ICON, FT)')
    parser.add_argument('--model', default=None, type=str, metavar='MODEL', help='Name of model to train')
    parser.add_argument('--pretrained', default=True, help='Load pretrained model or not')
    parser.add_argument('--linear_probing', action='store_true', help='Enable linear probing mode (freeze backbone, only train classifier)')


    # Optimizer parameters
    parser.add_argument('--opt', default='adam', type=str, metavar='OPTIMIZER', help='Optimizer (default: "adam"')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='weight decay (default: 0.0)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='constant', type=str, metavar='SCHEDULER', help='LR scheduler (default: "constant"')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate')

    # Data parameters
    parser.add_argument('--data_path', default='/local_datasets/', type=str, help='dataset path')
    parser.add_argument('--dataset', default='iDigits', type=str, help='dataset name')
    parser.add_argument('--shuffle', default=False, help='shuffle the data order')
    parser.add_argument('--save', default='./save', help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda', help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int, help='number of workers for data loading')

    # Simple replay buffer
    parser.add_argument('--replay_per_task', type=int, default=0, help='Number of samples to store per task for simple replay (0 disables).')

    # Continual learning parameters
    parser.add_argument('--num_tasks', default=10, type=int, help='number of sequential tasks')
    parser.add_argument('--IL_mode', type=str, default='cil', choices=['cil', 'dil', 'vil', 'joint'], help='Incremental Learning mode')

    # Misc (기타) parameters
    parser.add_argument('--print_freq', type=int, default=1000, help = 'The frequency of printing')
    parser.add_argument('--develop_tasks', '-d', action='store_true', default=False)
    parser.add_argument('--develop', '-dev', action='store_true', default=False)
    parser.add_argument('--verbose', '-v', action='store_true', default=False)

    # ood evaluation
    parser.add_argument('--ood_method', default='ALL', type=str, help='OOD detection method')
    parser.add_argument('--ood_dataset', default=None, type=str, help='OOD dataset name')
    parser.add_argument('--ood_eval', action='store_true', help='Perform ood evaluation only')
    parser.add_argument("--ood_develop", type=int, default=None)

    #wandb
    parser.add_argument('--wandb_run', type=str, default=None, help='Wandb run name')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name')

    # === OOD method hyper-parameters ===
    parser.add_argument('--energy_temperature', type=float, default=1.0, help='Temperature for ENERGY postprocessor')
    # GEN
    parser.add_argument('--gen_gamma', type=float, default=0.01, help='Gamma for GEN / PRO_GEN postprocessor')
    parser.add_argument('--gen_M', type=int, default=3, help='Top-M probabilities used in GEN / PRO_GEN postprocessor')
    # PRO-GEN
    parser.add_argument('--pro_gen_noise_level', type=float, default=5e-4, help='Noise level for PRO_GEN postprocessor')
    parser.add_argument('--pro_gen_gd_steps', type=int, default=3, help='Gradient descent steps for PRO_GEN postprocessor')
    # PRO-MSP
    parser.add_argument('--pro_msp_temperature', type=float, default=1.0, help='Temperature for PRO_MSP postprocessor')
    parser.add_argument('--pro_msp_noise_level', type=float, default=0.003, help='Noise level for PRO_MSP postprocessor')
    parser.add_argument('--pro_msp_gd_steps', type=int, default=1, help='Gradient descent steps for PRO_MSP postprocessor')
    # PRO-MSP-T
    parser.add_argument('--pro_msp_t_temperature', type=float, default=1.0, help='Temperature for PRO_MSP_T postprocessor')
    parser.add_argument('--pro_msp_t_noise_level', type=float, default=0.003, help='Noise level for PRO_MSP_T postprocessor')
    parser.add_argument('--pro_msp_t_gd_steps', type=int, default=1, help='Gradient descent steps for PRO_MSP_T postprocessor')
    # PRO-ENT
    parser.add_argument('--pro_ent_noise_level', type=float, default=0.0014, help='Noise level for PRO_ENT postprocessor')
    parser.add_argument('--pro_ent_gd_steps', type=int, default=2, help='Gradient descent steps for PRO_ENT postprocessor')

    args = parser.parse_args()
    Path(args.data_path).mkdir(parents=True, exist_ok=True)
    utils.update_ood_hyperparams(args)
    main(args)

    sys.exit(0)