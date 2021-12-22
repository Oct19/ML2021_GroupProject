import argparse
import os
import glob
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
from models import API_Net
from datasets import TestDataset, RandomDataset, default_loader
from utils import accuracy, AverageMeter, save_checkpoint
from tqdm import tqdm
from datetime import datetime

parser = argparse.ArgumentParser(description='PyTorch ImageNet Testing')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('-b', '--batch-size', default=1, type=int, metavar='N', help='mini-batch size (default: 10)')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transforms1 = transforms.Compose([transforms.Resize([512, 512]),
                                 transforms.CenterCrop([448, 448]),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                 mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225)
                                 )])

def main():
    global args
    args = parser.parse_args()
    # create model
    model = API_Net()
    model = model.to(device)
    model.conv = nn.DataParallel(model.conv)

    model.load_state_dict(dict(torch.load('checkpoint.pth',map_location = device))['state_dict'])

    cudnn.benchmark = True
    # Data loading code
    print('START TIME:', time.asctime(time.localtime(time.time())))
    test(model)
    #test(val_loader, model)

def test(model):
    # create result file
    result_filename = datetime.now().strftime('test_result_%Y_%m_%d_%H_%M.txt')
    f = open(os.path.join('result/' ,result_filename),'w')
    predict_list = []
    model.eval()
    dataroot = 'datasets/stanford_cars/cars_test/'
    datalist = open('datasets/stanford_cars/test_list.txt', 'r').readlines()
    with torch.no_grad():
        for ss in datalist[0:50]:
            imgname = ss.split(' ')[0]
            imgpath = dataroot + imgname
            img = default_loader(imgpath)
            img = transforms1(img)
            img = img.unsqueeze(0)
            img = img.to(device)
            logits = model(img, targets = None, flag = 'val')
            _, predict = logits.topk(1, 0, True, True)
            predict = int(predict.squeeze())
            print(f'{imgname} {predict}')
            predict_list.append(imgname+' '+str(predict)+'\n')

    f.writelines(predict_list)
    f.close()

def eval_with_txt():
    # Get latest result file
    list_of_files = glob.glob('result/*txt') 
    latest_file = max(list_of_files, key=os.path.getmtime)
    f1 = open('result/Groundtruth.txt', 'r').readlines()
    f2 = open(latest_file, 'r').readlines()
    summ = 0
    errorlist = []
    for i in range(len(f2)):
        predict = int(f2[i].split(' ')[1])
        label = int(f1[i].split(' ')[1])
        if predict == label:
            summ += 1
        else:
            errorlist.append(f2[i].split(' ')[0] + '\t' + str(label) + '\t\t' +str(predict) + '\n')
    # write evaluation file
    f = open(os.path.splitext(latest_file)[0] + '_eval.log', 'w')
    accuracy = summ/len(f2)
    print(str(accuracy))
    f.writelines(latest_file + '\n')
    f.writelines('Test size: ' + str(len(f2)) + '\n')
    f.writelines('Accuracy: ' + str(accuracy) + '\n')
    f.writelines('Test error\tLabel\tPredict\n')
    f.writelines(errorlist)
    f.close()

if __name__ == '__main__':
    # main()
    eval_with_txt()