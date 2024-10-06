from Generator.train import train
from Generator.craftadv import craftadv

import torch.nn as nn
import torch.nn.functional as F
import torch
import random
import numpy as np
import os
import argparse
parser = argparse.ArgumentParser(description='GAKer')

parser.add_argument('--state', type=str, default='train_model', choices=['train_model', 'craftadv','advtest'],help='Mode for model training or evaluation')
parser.add_argument('--Source_Model', type=str, default='ResNet50',help='Source Model')
parser.add_argument('--epoch', type=int, default=20, help='Number of epochs')#设置训练的 epoch 数量，默认值为 20，也就是模型会训练 20 次完整的训练集
parser.add_argument('--batch_size', type=int, default=16, help='Batch size')#指定每次训练的批大小，默认是 16，即模型每次会处理 16 个数据样本
#不太确定
parser.add_argument('--channel', type=int, default=32, help='Channel value')#--channel 参数指定了模型某层（或者多层）卷积操作的输出通道数量。默认值是 32，意味着这层卷积操作会生成 32 个特征图，这通常是 CNN 中的常见做法
#--channel_mult：指定通道的倍数，是一个列表，默认值是 [1, 2, 3, 4]。它可以用于构建更复杂的卷积层，改变网络结构。
parser.add_argument('--channel_mult', nargs='+', type=int, default=[1, 2, 3, 4],
                    help='List of channel multipliers')
#指定残差块（ResNet）的数量，默认是 1。残差块是 ResNet 的核心组件，残差网络需要了解
parser.add_argument('--num_res_blocks', type=int, default=1, help='Number of residual blocks')
#设置学习率，默认是 1e-4。学习率决定了模型在每次更新权重时步长的大小
parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
parser.add_argument('--device', type=str, default='cuda:0', help='Device for model training')
#--Generator_save_dir：指定生成器模型的保存路径，默认是 ./GAKer_saved_checkpoint/。
parser.add_argument('--Generator_save_dir', type=str, default='./GAKer_saved_checkpoint/', help='Directory to save checkpoints')
parser.add_argument('--test_load_weight', type=str, default='ckpt_19_ResNet50_.pt', help='Weight file for testing')
parser.add_argument('--set_targets', type=str, default='targets_200', help='target index of imagenet')
parser.add_argument('--unknown', type=str, default='False', help='if unknown or not')
parser.add_argument('--target_select', type=str, default='1', help='target_image_select')
parser.add_argument('--ran_best', type=str, help='Description of ran_best parameter')
args = parser.parse_args()

def main():

    if args.state == 'train_model':
        epoch = args.epoch
    else:
        ckpt = args.test_load_weight
        epoch = args.epoch
    modelConfig = {
        "state": args.state,
        "Source_Model": args.Source_Model,
        "epoch": epoch,
        "batch_size": args.batch_size,
        "channel": args.channel,
        "channel_mult": args.channel_mult,
        "num_res_blocks": args.num_res_blocks,
        "lr": args.lr,
        "device": args.device,
        "test_load_weight": args.test_load_weight,
        "Generator_save_dir": args.Generator_save_dir,
        'set_targets':args.set_targets,
        'unknown':args.unknown,
        'target_select':args.target_select,
    }
    
    if modelConfig["state"] == "train_model":
        train(modelConfig)
    elif modelConfig["state"] == "craftadv":
        craftadv(modelConfig)


        
if __name__ == '__main__':
    
    main()
    
    
