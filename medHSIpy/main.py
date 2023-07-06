import torch
import os
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from torchinfo import summary
import shutil

from tools import util
from learning.LearnModel import Learn_Model, valid
from models.ChoiceModel import Choice_Model
from dataset import MedHSIDataset, show_hsi_montage

conf = util.parse_config()

def save_learning_curve(result, num_of_epoch, num_attempt):
    x = np.arange(num_of_epoch)

    loss_max = max(max(result['train']['loss']), max(result['test']['loss']))

    fig, axes = plt.subplots(1, 2)
    axes[0].set_title('Loss')
    axes[0].plot(x, result['train']['loss'], color='red', label='train')
    axes[0].plot(x, result['test']['loss'], color='blue', label='test')
    axes[0].set_ylim(0, loss_max + 0.3)
    axes[0].legend()

    axes[1].set_title('Acc')
    axes[1].plot(x, result['train']['acc'], color='red', label='train')
    axes[1].plot(x, result['test']['acc'], color='blue', label='test')
    axes[1].set_ylim(0, 1)
    axes[1].legend()

    filename = os.path.join(conf['Directories']['outputDir'], 'T20230411', f'learning_curve{num_attempt}.png')
    plt.savefig(filename)

def disp_result(correct_data, false_data, num_attempt):
    print('______validation_______')
    print(f'num of correct, false: {len(correct_data)}, {len(false_data)}')
    
    if len(correct_data) != 0:
        show_hsi_montage(correct_data, save_name=f'correct_data_{num_attempt}_montage.jpg')
    
    if len(false_data) != 0:
        show_hsi_montage(false_data, save_name=f'false_data_{num_attempt}_montage.jpg')

def main():
    is_test = 0
    batch_size = 4
    # model_name = 'vggLike_3DCNN'
    # model_name = 'ResNetLike_3DCNN'
    model_name = 'my_3DCNN'
    # model_name = 'sample_3DCNN'
    learningRate = 0.001
    num_of_epoch = 300
    if is_test == 1:
        num_of_epoch = 10
    
    history = []
    
    dataset_path_lists = ['dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    
    remove_directory = 'output-python/T20230411'
    if os.path.isdir(remove_directory):
        shutil.rmtree(remove_directory)
        os.mkdir(remove_directory)
    
    for i, test_path in enumerate(dataset_path_lists):
        result_file_path = os.path.join(conf['Directories']['outputDir'], 'T20230411', "result.txt")
        f = open(result_file_path, 'a')
        f.write(f"{i} result \n")
        f.close()
        
        train_path_lists = []
        train_path_lists.extend(dataset_path_lists[:i])
        train_path_lists.extend(dataset_path_lists[i+1:])
        
        train_dataset = MedHSIDataset(train_path_lists)
        test_dataset = MedHSIDataset(test_path)
        target_class = train_dataset.return_target_class()
        
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        
        f = open(result_file_path, 'a')
        f.write(f'train path list: {train_path_lists}, num: {len(train_dataset)} \n')
        f.write(f'test path list: {test_path}, num: {len(test_dataset)} \n')
        f.close()
        
        print(f'train path list: {train_path_lists}, num: {len(train_dataset)}')
        print(f'test path list: {test_path}, num: {len(test_dataset)}')
        
        model = Choice_Model(model_name, train_dataset[0]['image'], target_class).to(torch.float)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate, momentum=0.9)
        
        model, result = Learn_Model(train_dataloader, test_dataloader, model, num_of_epoch, criterion, optimizer, device, is_test=is_test)
        
        _, _, correct_data, false_data = valid(model, test_dataloader, criterion, device, is_test=True, result_file_path=result_file_path)
        
        disp_result(correct_data, false_data, i)
        save_learning_curve(result, num_of_epoch, i)
        
        history.append([result['train']['loss'][-1], result['test']['loss'][-1], result['train']['acc'][-1], result['test']['acc'][-1]])
        
        f = open(result_file_path, 'a')
        f.write(f"\n")
        f.close()
    
    
    f = open(result_file_path, "a")
    print("train_loss, test_loss, train_acc, test_acc")
    f.write("\n")
    f.write("train_loss, test_loss, train_acc, test_acc \n")
    for his in history:
        print(his)
        f.write(f"{his} \n")
    f.close()
    
    num_of_channel, num_of_components, width, height = train_dataset[0]['image'].shape
    summary(model=model, input_size=(batch_size, num_of_channel, num_of_components, width, height))

def main_no_crossvalidation():
    batch_size = 4
    # model_name = 'vggLike_3DCNN'
    # model_name = 'ResNetLike_3DCNN'
    model_name = 'my_3DCNN'
    # model_name = 'sample_3DCNN'
    learningRate = 0.001
    num_of_epoch = 10
    
    num_of_attempts = 5
    history = []
    
    train_dataset = MedHSIDataset('train_dataset')
    test_dataset = MedHSIDataset('test_dataset')
    target_class = train_dataset.return_target_class()
    
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'
    print(device)
    
    for i in range(num_of_attempts):
        model = Choice_Model(model_name, train_dataset[0]['image'], target_class).to(torch.float)
        if device == 'cuda':
            model = torch.nn.DataParallel(model) # make parallel
            torch.backends.cudnn.benchmark = True
        
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(params=model.parameters(), lr=learningRate, momentum=0.9)
        
        model, result = Learn_Model(train_dataloader, test_dataloader, model, num_of_epoch, criterion, optimizer, device)
        
        _, _, correct_data, false_data = valid(model, test_dataloader, criterion, device, is_test=True)
        
        disp_result(correct_data, false_data, i)
        save_learning_curve(result, num_of_epoch, i)
        
        history.append([result['train']['loss'][-1], result['train']['acc'][-1], result['test']['loss'][-1], result['test']['acc'][-1]])
    
    for i in range(num_of_attempts):
        print(history[i])
    
    num_of_channel, num_of_components, width, height = train_dataset[0]['image'].shape
    summary(model=model, input_size=(batch_size, num_of_channel, num_of_components, width, height))


if __name__ == '__main__':
    main()