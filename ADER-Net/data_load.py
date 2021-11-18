import numpy as np
import torch
from torchvision import datasets,transforms
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler
from PIL import Image
from options import parser


args = parser.parse_args()
trans_size_height = 192
trans_size_width = 320



class myDataset(Dataset):

    def __init__(self,dataset_path,transform=None,mode='train'):

        super(myDataset,self).__init__()
        self.dataset_path=dataset_path
        self.transform=transform
        self.mode = mode
        if(self.mode == 'train'):
            with open('training.txt') as f: # add the dir you save the txt file for training
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    self.label_list.append(line[11:13])
                    pic_path = []
                    for i in range(0, 16):
                        pic_path.append(self.dataset_path + line[30 + 18 * i:44 + 18 * i])
                    self.img_list.append(pic_path)
        elif(self.mode == 'test'):
            with open('test.txt') as f:# add the dir you save the txt file for test
                f.readline()
                self.img_list = []
                self.label_list = []
                for line in f:
                    self.label_list.append(line[11:13])
                    pic_path = []
                    for i in range(0, 16):
                        pic_path.append(self.dataset_path + line[30 + 18 * i:44 + 18 * i])
                    self.img_list.append(pic_path)


    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):

        label = self.label_list[index]
        label = int(label[-1:])
        if label not in[0,1,2,3,4,5]:
            label = 0
        label_res = torch.tensor(label)

        image = self.img_list[index]

        traffic_picture_res = torch.zeros(16, 3, trans_size_height,trans_size_width)

        for i in range(0,16):
            image_tmp = Image.open(image[i])
            traffic_picture_tmp = self.transform['traffic'](image_tmp)
            traffic_picture_res[i] = traffic_picture_tmp

        # print(traffic_picture_res.shape)
        traffic_picture_res = traffic_picture_res.permute(1,0,2,3)
        return traffic_picture_res,label_res



def getTrainVal_loader(train_dataset_path ,mode_tmp ,shuffle=True,val_split=0.15):


    data_transforms = {
        'traffic': transforms.Compose([
        transforms.Resize((trans_size_height+20, trans_size_width+40)),
        transforms.RandomCrop((trans_size_height,trans_size_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
        ])
    }

    trainval_dataset = myDataset(dataset_path = train_dataset_path,transform=data_transforms,mode=mode_tmp)
    dataset_size=len(trainval_dataset)
    indices=list(range(dataset_size))
    split=int(np.floor(val_split*dataset_size))
    if shuffle:
        np.random.shuffle(indices)

    train_indices,val_indices=indices[split:],indices[:split]


    train_sampler=SubsetRandomSampler(train_indices)
    valid_sampler=SubsetRandomSampler(val_indices)

    train_loader=DataLoader(trainval_dataset,batch_size=args.batch_size,sampler=train_sampler,num_workers=16,drop_last=True)
    val_loader=DataLoader(trainval_dataset,batch_size=args.batch_size,sampler=valid_sampler,num_workers=16,drop_last=True)

    trainval_loaders={'train':train_loader,'val':val_loader}

    return trainval_loaders



def getTest_loader(test_dataset_path_tmp ,mode_tmp,shuffle=True):


    data_transforms={
        'traffic' : transforms.Compose([
            transforms.Resize((trans_size_height,trans_size_width)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])
    }

    test_dataset = myDataset(dataset_path=test_dataset_path_tmp, transform=data_transforms,mode=mode_tmp)



    return test_dataset

