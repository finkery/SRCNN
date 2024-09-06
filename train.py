import argparse
import os
import copy

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import mtf_sr
from loss import combined_loss
from datasets import TrainDataset,EvalDataset
from utils import AverageMeter,calculate_slope_aspect

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-file',type=str,required=True)
    parser.add_argument('--eval-file',type=str,required=True)
    parser.add_argument('--output-dir',type=str,required=True)
    parser.add_argument('--scale',type=int,default=2)
    parser.add_argument('--lr',type=float,default=1e-4)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--num-epoch',type=int,default=400)
    parser.add_argument('--num-workers',type=int,default=8)
    parser.add_argument('--seed',type=int,default=123)
    args = parser.parse_args()

    args.output_dir = os.path.join(args.ouput_dir,'x{}'.format(args.scale))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = mtf_sr().to(device)
    criterion = combined_loss()

    optimizer = optim.Adam(lr = args.lr)
    train_dataset = TrainDataset(args.train_file)
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    eval_dataset = EvalDataset(args.eval_file)
    eval_dataloader = DataLoader(
        dataset = eval_dataset,
        batch_size=1
    )
    for data in train_dataloader:
        inputs,labels = data
        print(len(inputs))
        break
    # for epoch in range(args.num_epoch):
    #     model.train()
    #     epoch_losses = AverageMeter()

    #     with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
    #         t.set_description('epoch:{}/{}'.format(epoch,args.num_epoch - 1))
    #         for data in train_dataloader:
    #             inputs,labels = data

    #             inputs = inputs.to(device)
    #             aspect_inputs = calculate_slope_aspect()
    #             labels = labels.to(device)

    #             preds = model(inputs)

