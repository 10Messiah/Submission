import argparse

##################################################for train############################################################
parser = argparse.ArgumentParser()
# dataset dir
parser.add_argument('--train_dataset_dir',type=str,default='') # add the dir you save the images for training

# train parameters
parser.add_argument('--batch_size',type=int,default=2,help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate for training')
parser.add_argument('--epochs', type=int, default=40, help='epoch for training')
parser.add_argument('--log_interval',type=int,default=1,help='interval for printing training information')


parser.add_argument('--decay', type=float, default=0.0001, help='weight decay for training')
parser.add_argument('--start_epoch', type=int, default=0, help='start epoch for resuming training')
parser.add_argument('--checkpoint', type=int, default=25, help='checkpoint for save model')

parser.add_argument('--resume', default='', type=str,help='path to latest checkpoint (default: none)')
# parser.add_argument('--resume', default='weight/null.pth.tar', type=str,help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', default='', type=str,help='load pretrained weight in new model')
# parser.add_argument('--pretrained', default='weight/saliency_weights_used.pth.tar', type=str,help='load pretrained weight in new model')
parser.add_argument('--gpu', default='0', type=str,metavar='N', help='GPU NO. (default: 0)')


##################################################for test############################################################
# dataset type
parser.add_argument('--test_dataset_dir', type=str, default='')# add the dir you save the images for test
# model dir
parser.add_argument('--model_dir', type=str,default='weight/ADER.pth.tar')




