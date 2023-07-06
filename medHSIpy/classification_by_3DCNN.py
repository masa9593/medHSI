# -*- coding: utf-8 -*
from random import seed
from tools import hio, util
from learning.LearnModel import Learn_Model, valid
from models.ChoiceModel import Choice_Model
import matplotlib.pyplot as plt
import os

import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import torch.nn as nn

import numpy as np

conf = util.parse_config()

# image data
random_state = 10
DEFAULT_HEIGHT = 64
n_components = 64 * 6

#target data
Class = {
    'Malignant' : 0,
    'Benign' : 1
}

# learning setting
# model_name = 'vggLike_3DCNN'
# model_name = 'ResNetLike_3DCNN'
model_name = 'my_3DCNN'
# model_name = 'sample_3DCNN'

batch_size = 4
learningRate = 0.001
num_of_epoch = 300

x_train, x_test, y_train, y_test = hio.get_train_test_for_clasification(Class, height=DEFAULT_HEIGHT, n_components=n_components, is_split=True, show_image=True, random_state=random_state)

print(y_train)
print(y_test)
print(sum(y_train==0), sum(y_train==1))
print(sum(y_test==0), sum(y_test==1))

class medHSI_Dataset(Dataset):
    def __init__(self, images, targets, transform, n_components=10):
        self.images = images
        self.targets = targets
        self.transform = transform
        self.n_components = n_components
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, index):
        original_image = self.images[index]
        image = self.transform(original_image)
        image = self.decompose(image, n_components=self.n_components)
        image = image.view(1, image.size(0),image.size(1),image.size(2))
        
        target = self.targets[index]
        
        dataset = {'image': image, 'target': target, 'original_image': original_image}
        
        return dataset
        
    def show_montage_dataset(self, channel=401, name='train'):
        from tools import util
        util.show_montage(self.images, channel=channel, savename=name+'_montage.jpg')
    
    def decompose(self, hsi, n_components=10):
        numDecom = hsi.shape[0] - n_components
        decom_hsi = hsi[numDecom//2:numDecom//2+n_components, :, :]
        
        return decom_hsi

target_class = list(Class.values())

originalTransform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(tuple(util.Average_for_hsiList(x_train)), tuple(util.Std_for_hsiList(x_train)))])

train_dataset = medHSI_Dataset(x_train, y_train, transform=originalTransform, n_components=n_components)
train_dataset.show_montage_dataset(channel=401, name='train')
test_dataset = medHSI_Dataset(x_test, y_test, transform=originalTransform, n_components=n_components)
test_dataset.show_montage_dataset(channel=401, name='test')

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = 'cpu'
print(device)

n = 5

r = []

for i in range(n):
    model = Choice_Model(model_name, train_dataset[0]['image'], target_class).to(torch.float).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model) # make parallel
        torch.backends.cudnn.benchmark = True
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate, momentum=0.9)
    
    model, result = Learn_Model(train_dataloader, test_dataloader, model, num_of_epoch, criterion, optimizer, device)
    
    r.append([result['train']['loss'][-1], result['train']['acc'][-1], result['test']['loss'][-1], result['test']['acc'][-1]])

for i in range(n):
    print(r[i])

# train_loss, train_acc, train_Correct_Data, train_False_Data = valid(model, train_dataloader, criterion, device, is_test=True)

'''
print(len(train_Correct_Data))
if len(train_Correct_Data) != 0:
    print(train_Correct_Data[0].shape)
    
print(len(train_False_Data))
if len(train_False_Data) != 0:
    print(train_False_Data[0].shape)
'''

'''
if len(train_Correct_Data) != 0:
    util.show_montage(train_Correct_Data, channel=401, savename='train_Correct_Data_montage.jpg')

if len(train_False_Data) != 0:
    util.show_montage(train_False_Data, channel=401, savename='train_False_Data_montage.jpg')
'''

val_loss, val_acc, Correct_Data, False_Data = valid(model, test_dataloader, criterion, device, is_test=True)

print(len(Correct_Data))
if len(Correct_Data) != 0:
    print(Correct_Data[0].shape)
    
print(len(False_Data))
if len(False_Data) != 0:
    print(False_Data[0].shape)

if len(Correct_Data) != 0:
    util.show_montage(Correct_Data, channel=401, savename='Correct_Data_montage.jpg')

if len(False_Data) != 0:
    util.show_montage(False_Data, channel=401, savename='False_Data_montage.jpg')

x = np.arange(num_of_epoch)

loss_max = max(max(result['train']['loss'], result['test']['loss'], key=max))

plt.subplot(1, 2, 1, title='Loss')
plt.plot(x, result['train']['loss'], color='red', label='train')
plt.plot(x, result['test']['loss'], color='blue', label='test')
plt.ylim(0, loss_max+0.3)
plt.legend()

plt.subplot(1, 2, 2, title='Acc')
plt.plot(x, result['train']['acc'], color='red', label='train')
plt.plot(x, result['test']['acc'], color='blue', label='test')
plt.ylim(0, 1)

plt.legend()

filename = os.path.join(conf['Directories']['outputDir'], 'T20211207-python', '学習曲線.png')
plt.savefig(filename)

plt.show()

from torchinfo import summary

summary(model=model, input_size=(batch_size, 1, n_components, DEFAULT_HEIGHT, DEFAULT_HEIGHT))

