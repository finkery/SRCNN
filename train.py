import argparse
import os
import copy
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from models import mtf_sr
from loss import combined_loss
from datasets import TrainDataset,EvalDataset
from utils import AverageMeter,calculate_slope_aspect,calc_rmse

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

    args.output_dir = os.path.join(args.output_dir,'x{}'.format(args.scale))

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(args.seed)

    model = mtf_sr().to(device)
    params = model.parameters() 
    optimizer = optim.Adam(params, lr = args.lr)
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

    best_weight = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_rmse = 0

    for epoch in range(args.num_epoch):
        model.train()
        epoch_losses = AverageMeter()

        with tqdm(total=(len(train_dataset) - len(train_dataset) % args.batch_size)) as t:
            t.set_description('epoch:{}/{}'.format(epoch,args.num_epoch - 1))
            for data in train_dataloader:
                inputs,label = data
                inputs = inputs.to(device)
                label = label.to(device)
                slope_inputs,aspect_inputs = calculate_slope_aspect(inputs)
                inputs = torch.stack((inputs,slope_inputs,aspect_inputs),axis=1)
                inputs = torch.where(torch.isnan(inputs) | torch.isinf(inputs), torch.zeros_like(inputs), inputs)
                label = torch.where(torch.isnan(label) | torch.isinf(label), torch.zeros_like(label), label)
                preds = model(inputs)
                preds = preds.squeeze(1)
                loss = combined_loss(label,preds)
                epoch_losses.update(loss.item(),len(inputs))

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                t.set_postfix(loss=f'{epoch_losses.avg:.6f}')
                t.update(len(inputs))
        
        torch.save(model.state_dict(),os.path.join(args.output_dir,"epoch_{}.pth".format(epoch)))

        model.eval()
        epoch_rmse = AverageMeter()

        for data in eval_dataloader:

            inputs,labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            slope_inputs,aspect_inputs = calculate_slope_aspect(inputs,eval=True)
            # print(inputs)
            inputs = torch.stack((inputs,slope_inputs,aspect_inputs),axis=1,)
            inputs = torch.where(torch.isnan(inputs) | torch.isinf(inputs), torch.zeros_like(inputs), inputs)
            labels = torch.where(torch.isnan(labels) | torch.isinf(labels), torch.zeros_like(labels), labels)
            preds = model(inputs)
            epoch_rmse.update(calc_rmse(preds,labels),len(inputs))

            print('eval_rmse {:.2f}'.format(epoch_rmse.avg))

            if epoch_rmse.avg > best_rmse:
                best_epoch = epoch
                best_rmse = epoch_rmse.avg
                best_weight = copy.deepcopy(model.state_dict())

    print("best_epoch:{} best_rmse:{:.2f}".format(best_epoch,best_rmse))
    torch.save(best_epoch,os.path.join(args.output_dir,'best.pth'))

