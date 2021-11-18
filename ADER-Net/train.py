import numpy as np
import torch
import torch.nn as nn
import time
import math
import os
from sklearn.metrics import accuracy_score

from data_load import getTrainVal_loader
from options import parser

from ADER_Net import ADER_Net


def main():
    global args, best_score,save_epoch
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    torch.manual_seed(2017)
    torch.cuda.manual_seed(2017)
    np.random.seed(2017)

    model = ADER_Net()
    model = model.cuda()
    params = model.parameters()
    optimizer = torch.optim.Adam(params, args.lr,weight_decay=args.decay)
    best_loss = 1000.0

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):

            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            best_loss = checkpoint['best_loss']
            best_acc = checkpoint['best_acc']
            print("=> loaded checkpoint '{}' epoch {}  best loss {}  best_acc {} ".format(args.resume, checkpoint['epoch'], \
                                                                    best_loss,best_acc))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.pretrained:
        if os.path.isfile(args.pretrained):

            checkpoint = torch.load(args.pretrained)
            pretrained_dict = checkpoint['state_dict']
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            # 3. load the new state dict
            model.load_state_dict(model_dict)
            print("\033[91m%s\033[0m" %'the model has loaded  pretrained weights of saliency model')


    dataloaders = getTrainVal_loader(train_dataset_path=args.train_dataset_dir, mode_tmp='train')


    weights = torch.tensor([330, 144, 2268, 828, 828, 330], dtype=torch.float32)
    weights = torch.tensor([torch.max(weights) / x for x in weights])
    criterion = nn.CrossEntropyLoss(weight=weights).cuda()



    for epoch in range(args.start_epoch, args.epochs):
        print("-" * 65)
        print("\033[91m%s\033[0m" % "Epoch {}/{}".format(epoch + 1, args.epochs))

        adjust_learning_rate(optimizer, epoch)
        train_loss = train(dataloaders, model, criterion, optimizer, epoch)
        valid_loss,acc_tmp = validate(dataloaders, model, criterion, epoch)


        is_best = abs(valid_loss) < abs(best_loss)
        best_loss = min(abs(valid_loss) , abs(best_loss))

        if is_best:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
            }, is_best,filename='ADER.pth.tar')



def train(data_loader, model, criterion_tmp, optimizer_tmp, epoch):
    losses = AverageMeter()
    model.train()
    train_loader = data_loader['train']
    print('lr--------> ', optimizer_tmp.param_groups[0]['lr'])

    cnt = 0
    for i,(input, target) in enumerate(train_loader):

        input = input.cuda()
        target = target.cuda()
        output = model(input)

        loss = criterion_tmp(output, target)
        losses.update(loss.item(), input.size(0))


        optimizer_tmp.zero_grad()
        loss.backward()
        optimizer_tmp.step()

        cnt = cnt + 1
        if (cnt + 1) % args.log_interval == 0:
            print('[Epoch %3d]Training %3d of %3d:  loss = %.5f' % (epoch + 1, cnt + 1, len(train_loader), losses.avg))

    print("{} Loss: {}".format('train', round(losses.avg,5)))


    return losses.avg


def validate(data_loader, model, criterion_tmp, epoch):
    losses = AverageMeter()
    acces = AverageMeter()
    model.eval()
    val_loader = data_loader['val']

    cnt = 0
    with torch.no_grad():
        for i,(input, target) in enumerate(val_loader):

            input = input.cuda()
            target = target.cuda()
            output = model(input)

            loss = criterion_tmp(output, target)

            output = output.argmax(dim=1)
            acc = accuracy_score(output.cpu().numpy(), target.cpu().numpy())

            losses.update(loss.item(), input.size(0))
            acces.update(acc)

            cnt = cnt + 1
            if (cnt + 1) % args.log_interval == 0:
                print('[Epoch %3d]Training %3d of %3d: acc = %.5f, loss = %.5f' % (epoch + 1, cnt + 1, len(val_loader), acces.avg, losses.avg))


    print("{} Loss: {}   acc:{}".format('val', round(losses.avg,5),round(acces.avg,5)))


    return losses.avg,acces.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 20 epochs"""
    lr = args.lr * (0.1 ** (epoch // 20))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


pth_save_path = 'weight'
def save_checkpoint(state, is_best, filename):
    if not os.path.exists(pth_save_path):
        os.makedirs(pth_save_path)

    if is_best:
        best_name = os.path.join(pth_save_path, filename)
        torch.save(state, best_name)
        print("\033[91m%s\033[0m" % 'saving weight...........................')
        print("\033[91m%s\033[0m" % '****************************************')


if __name__ == '__main__':
    main()
 


