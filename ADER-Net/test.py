import numpy as np
import torch
from options import parser
from data_load import getTest_loader
from ADER_Net import ADER_Net
args = parser.parse_args()
f = open('results/res.txt',mode='w')



def test(model,device,test_loader):
    model.eval()
    cnt = 0

    with torch.no_grad():
        for input,target in test_loader:
            image=input
            image=image.unsqueeze(0).to(device)
            pred=model(image)

            pred = pred.squeeze(0)
            pred = torch.softmax(pred,dim=0)
            pred = pred.cpu().numpy()
            pred_tmp = [round(i, 3) for i in pred]

            target = target.item()
            target_tmp = []
            for i in range(0,target):
                target_tmp.append(0.0)
            target_tmp.append(1.0)
            for i in range(target+1,6):
                target_tmp.append(0.0)
            target_tmp = [round(i, 2) for i in target_tmp]

            f.write(str(pred_tmp)+'\n')
            f.write(str(target_tmp) + '\n')
            cnt = cnt+ 1
            if(cnt %10 == 0):
                print('testing:',str(round(cnt,2)))



def main_test():

    np.random.seed(18)
    dataloaders=getTest_loader(test_dataset_path_tmp=args.test_dataset_dir,mode_tmp='test')



    # init the model
    model_weight=args.model_dir
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")
    model=ADER_Net()
    checkpoint = torch.load(model_weight)
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    print("Begin test, Device: {}".format(device))

    test(model,device,dataloaders)

if __name__ == '__main__':
    main_test()

