# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
from pathlib import Path

import numpy as np
import torch
from .models import build_ACT_model, build_CNNMLP_model

import IPython
e = IPython.embed

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=2, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=200, type=int)
    parser.add_argument('--clip_max_norm', default=0.1, type=float,
                        help='gradient clipping max norm')

    # Model parameters
    parser.add_argument('--backbone', default='resnet18', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--dilation', action='store_true',
                        help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--camera_names', default=[], type=list,
                        help="A list of camera names")

    # * Transformer
    parser.add_argument('--enc_layers', default=4, type=int)
    parser.add_argument('--dec_layers', default=6, type=int)
    parser.add_argument('--dim_feedforward', default=2048, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num_queries', default=400, type=int)
    parser.add_argument('--pre_norm', action='store_true')

    # * Segmentation
    parser.add_argument('--masks', action='store_true')

    # repeat args from imitate_episodes to avoid error
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=False)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=False)
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=False)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=False)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=False)
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--temporal_agg', action='store_true')

    # ← 新增：与 imitate_episodes.py 保持一致
    parser.add_argument('--state_dim', action='store', type=int, default=None,
                        help='训练使用前几维数据')

    return parser


def build_ACT_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    # ← 核心修复：parse_known_args 忽略所有未注册参数，不再报错
    args, _ = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_ACT_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    return model, optimizer


def build_CNNMLP_model_and_optimizer(args_override):
    parser = argparse.ArgumentParser('DETR training and evaluation script',
                                     parents=[get_args_parser()])
    # ← 核心修复：同上
    args, _ = parser.parse_known_args()

    for k, v in args_override.items():
        setattr(args, k, v)

    model = build_CNNMLP_model(args)
    model.cuda()

    param_dicts = [
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters()
                       if "backbone" in n and p.requires_grad],
            "lr": args.lr_backbone,
        },
    ]
    optimizer = torch.optim.AdamW(param_dicts, lr=args.lr,
                                  weight_decay=args.weight_decay)
    return model, optimizer