# -*- coding: utf-8 -*-
"""
Created on Tue Jul  5 10:23:59 2022

@author: user
"""

from __future__ import print_function, division
import torch.nn as nn
from torch.autograd import Variable
import torch 
from torch.utils.data import Dataset,DataLoader
from PIL import Image
from torchvision import datasets, transforms, models
import time
import os

device = torch.device("cuda:0")
use_gpu = torch.cuda.is_available()
alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789'

def img_loader(img_path):
    img = Image.open(img_path)
    return img.convert("RGB")

class CaptchaData(Dataset):
    def __init__(self, image_path,
                 transform=None, alphabet=alphabet):
        super(Dataset, self).__init__()

        self.transform = transform
        self.alphabet = alphabet
        self.samples = image_path

    def __len__(self):
        #print(len(self.samples))
        return 1

    def __getitem__(self, index):
        img_path = self.samples

        img = img_loader(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img

def predict(output):
    output = output.view(-1, 36)
    output = nn.functional.softmax(output, dim=1)
    output = torch.argmax(output, dim=1)
    ans_list = []
    for i in output:
        ans_char = alphabet[i]
        ans_list.append(ans_char)
    ans = "".join(map(str, ans_list))
    #ans = [ans]
    #print(ans)
    return ans

def return_dataloader(img):
    val_image_path = img

    val_transforms = transforms.Compose([
        transforms.ToTensor(),
                                    ])

    dataset_sizes = 1

    val_dataset = CaptchaData(val_image_path,transform = val_transforms)
    val_data_loader = DataLoader(val_dataset, batch_size=dataset_sizes, num_workers=0,shuffle=True)
    return val_data_loader
def test_model(model,val_data_loader):
    for data in val_data_loader:
        inputs = data
        if use_gpu == True :
            inputs = Variable(inputs.cuda(device))
        outputs = model(inputs)
        return predict(outputs)

def test_acc(ResulltList,TestList):
    total = len(ResulltList)
    correctt = 0
    for i in range(total):
        if ResulltList[i] == TestList[i][:4]:
            correctt+=1
    return [correctt,total,correctt/total]

if __name__ == '__main__':
    TestStart = time.time()
    model1 = torch.load('densenet121_ep50_fulldata.pkl')
    if use_gpu == True:
        model1 = model1.cuda(device)
    rdl1 = return_dataloader('image.jpg')
    result = test_model(model1, rdl1)
    TestEnd = time.time()
    print(result)
    print(f'Single Respense：{TestEnd - TestStart:.2f} Second')


