import os
import glob

import numpy as np
import cv2
import h5py
import matplotlib.pyplot as plt
import skimage.io, skimage.util
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset

from tools import hio, util
from remove_files import empty_directory

CONF = util.parse_config()
PARTITION = '_______________'

class HSIDataset():
    def __init__(self) -> None:
        pass
        
    def decompose(self, num_of_components):
        num_decom = self.image.shape[0] - num_of_components
        self.image = self.image[num_decom//2:num_decom//2+num_of_components, :, :]
        self.wavelength_information = self.wavelength_information[num_decom//2:num_decom//2+num_of_components]
        return self.image
    
    def mask(self):
        self.image = self.image * self.mask_image
        self.noisy_image = self.noisy_image * self.mask_image
        self.original_image = self.original_image * self.mask_image
        # self.show_hsi(save_name='after_mask.jpg')
        return self.image
    
    def split(self, patch_size, class_correspondence_table):
        image_size = np.array(self.image.shape)
        num_cut = np.floor(image_size[1:3] / patch_size).astype(int)
        
        cropped_image = self.__center_crop_hsi(self.image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        cropped_original_image = self.__center_crop_hsi(self.original_image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        cropped_label_image = self.__center_crop_hsi(self.label_image, num_cut[0]*patch_size, num_cut[1]*patch_size)
        # patchIndex = np.meshgrid(np.arange(0,st[0], dtype=np.int32), np.arange(0,st[1], dtype=np.int32))
        
        self.split_dataset = []
        
        for x in range(num_cut[0]):
            for y in range(num_cut[1]):
                dataset = HSIDataset()
                dataset.image = cropped_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.original_image = cropped_original_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.label_image = cropped_label_image[:, (0 + x*patch_size):(patch_size + x*patch_size), (0 + y*patch_size):(patch_size + y*patch_size)]
                dataset.wavelength_information = self.wavelength_information
                dataset.ID = self.ID
                
                if np.sum(dataset.label_image) == 0:
                    if 'Non Cancer' in class_correspondence_table.keys():
                        dataset.label = class_correspondence_table['Non Cancer']
                    else:
                        continue
                else:
                    dataset.label = self.label
                
                if np.sum(dataset.image) > patch_size:
                    self.split_dataset.append(dataset)
                    
                # self.show_hsi(save_name=f'{str(self.ID)}{str(x)}{str(y)}.jpg')
                    
        return self.split_dataset
    
    def show_hsi(self, save_directory=None, save_name='hsi.jpg'):
        rgb_image = util.get_display_image(self.original_image, channel = 401)
        if rgb_image.dtype == 'float':
            rgb_image = (rgb_image * 255).astype('uint8')
        
        if save_directory is None:
            save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411')
        save_path = os.path.join(save_directory, save_name)
        
        skimage.io.imsave(save_path, rgb_image)
        
        return 1
    
    def show_spectral_of_image(self, hsi, save_directory=None, save_name=None):
        width = hsi.shape[1]
        height = hsi.shape[2]
        
        pixel_spectral_list = []
        label_list = []
        pixel_spectral_list.append(hsi[:, width//2, height//2])
        label_list.append("central")
        pixel_spectral_list.append(self.__search_max_spectral(hsi))
        label_list.append("max")
        
        self.__show_spectral_of_pixel(pixel_spectral_list, label_list, save_directory, save_name)
        
        return 1
    
    def __search_max_spectral(self, hsi):
        sum_hsi = np.sum(hsi, axis=0)

        # 合計値が最大となる1次元配列のインデックスを取得
        max_index = np.unravel_index(np.argmax(sum_hsi), sum_hsi.shape)

        # 最大値を持つ1次元配列とその要素を表示
        max_spectral = hsi[:, max_index[0], max_index[1]]
        
        return max_spectral
    
    def __show_spectral_of_pixel(self, pixel_spectral_list, label_list, save_directory=None, save_name=None):
        num_of_spectral = len(pixel_spectral_list[0])
        if num_of_spectral == 401:
            x = self.wavelength_information
        else:
            x = range(len(pixel_spectral_list[0]))
        
        fig, ax = plt.subplots()
        for pixel_spectral, label in zip(pixel_spectral_list, label_list):
            ax.plot(x, pixel_spectral, label=label)
        
        ax.legend()
        
        if save_directory is None:
            save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411')
        if save_name is None:
            save_name = 'image' + str(self.ID)
        save_path = os.path.join(save_directory, save_name)
        plt.savefig(save_path)
        plt.close(fig)
    
    def moving_average_of_image(self, num_of_weight=5):
        if num_of_weight % 2 == 0:
            num_of_weight = num_of_weight + 1
            print(f'num of weight is changed to {num_of_weight}')
        
        moving_filter = np.ones(num_of_weight) / num_of_weight
        
        width = self.image.shape[1]
        height = self.image.shape[2]
        for i in range(width):
            for j in range(height):
                self.image[:,i,j] = np.convolve(self.image[:,i,j], moving_filter, mode='same')
        
        self.original_image = self.image
        
        return self.image
    
    def set_ID(self, ID, used_IDs):
        if type(ID) is int:
            ID = self.__change_three_digit_number_to_str(ID)
        elif type(ID) is str:
            pass
        else:
            raise ValueError('Unknown type')
        
        if ID in used_IDs:
            self.ID = ID
            return 1
        
        return 0
    
    def set_sample_ID(self, sample_ID):
        if type(sample_ID) is int:
            sample_ID = self.__change_three_digit_number_to_str(sample_ID)
        elif type(sample_ID) is str:
            pass
        else:
            raise ValueError('Unknown type')
        
        self.sample_ID = sample_ID
        return 1
    
    def set_image_and_wavelength_information(self, file_name):  # image shape: (channel, width, height)
        if file_name[-3:] != 'raw':
            return 0
        
        file_path = os.path.join(CONF['Directories']['dataDir'], 'NGMeet_dataset', file_name + '.h5')
        with h5py.File(file_path, 'r') as f:
            self.image = np.array(f['NGMeet'])
            self.noisy_image = np.array(f['SpectralImage'])
            self.original_image = self.image
            self.wavelength_information = np.array(f['Wavelengths'])
        if self.image.shape[0] != 401:
            raise ValueError('wrong image shape')
        
        if self.image is None:
            return 0
        
        return 1
    
    def set_label(self, label_data_file, class_correspondence_table):
        for j, label_sample_ID in enumerate(label_data_file['SampleID']):
            label_sample_ID = self.__change_three_digit_number_to_str(label_sample_ID)
            if self.sample_ID == label_sample_ID:
                label = label_data_file['Type'][j]
                if label in class_correspondence_table.keys():
                    self.label = class_correspondence_table[label]
                    return 1
        return 0
    
    def set_label_image(self):  # image shape: (channel(1), width, height)
        file_path = os.path.join(CONF['Directories']['labelDir'], self.ID + '.png')
        image = self.__load_color_image_as_gray_image(file_path)
        self.label_image = self.__get_binary_image_from_image(image)
        self.label_image = self.__cut_image_to_same_size(self.label_image, self.image)
        
        if self.label_image is None:
            return 0
        
        return 1
    
    def set_mask_image(self):  # image shape: (channel(1), width, height)
        file_path = os.path.join(CONF['Directories']['maskDir'], self.ID + '.png')
        image = self.__load_color_image_as_gray_image(file_path)
        self.mask_image = self.__get_binary_image_from_image(image)
        self.mask_image = self.__cut_image_to_same_size(self.mask_image, self.image)
        
        if self.mask_image is None:
            return 0
        
        return 1

    def __load_color_image_as_gray_image(self, file_path): # -> (channel * width * height)
        label_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        rot_img = label_img.transpose(1, 0)
        rot_img = rot_img[np.newaxis,:,:]
        return rot_img

    def __get_binary_image_from_image(self, image): 
        if image.shape[0] == 3:
            gray_img = (image[0:1, :, :] + image[1:2, :, :] + image[2:3, :, :]) / 3
        elif image.shape[0] == 1:
            gray_img = image
        else:
            raise ValueError('Unsupposed mask channel number')
        
        binary_img = np.where(gray_img==0, 0, 1)
        return binary_img
    
    def __cut_image_to_same_size(self, cut_image, image):
        return cut_image[:image.shape[0], :image.shape[1], :image.shape[2]]
    
    def __change_three_digit_number_to_str(self, three_digit_number):
        return str(three_digit_number + 1000)[1:]

    def __center_crop_hsi(self, hsi, target_width, target_height):        
        width = hsi.shape[1]
        height = hsi.shape[2]
        
        if target_width is None:
            target_width = min(width, height)

        if target_height is None:
            target_height = min(width, height)

        left = int(np.ceil((width - target_width) / 2))
        right = width - int(np.floor((width - target_width) / 2))

        top = int(np.ceil((height - target_height) / 2))
        bottom = height - int(np.floor((height - target_height) / 2))

        if np.ndim(hsi) > 2:
            croppedImg = hsi[:, left:right, top:bottom]
        else:
            croppedImg = hsi[left:right, top:bottom]
            
        return croppedImg    




class MedHSIDataset(Dataset):
    def __init__(self, path) -> None:
        super().__init__()
        self.file_list = self.__get_path(path)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        with h5py.File(self.file_list[index], 'r') as f:
            image = f['image'][:]
            image = image[np.newaxis, :, :, :]
            label = f['label'][0]
            original_image = f['original_image'][:]
        image = torch.from_numpy(image.astype(np.float32)).clone()
        dataset = {
            'image': image,
            'label': label,
            'original_image': original_image
        }
        return dataset
    
    def __get_path(self, path):
        if type(path) is list:
            file_list = []
            for item in path:
                file_list.extend(glob.glob('./' + item + '/*'))
        else:
            file_list = glob.glob('./' + path + '/*')
        return file_list
    
    def return_target_class(self):
        with h5py.File(self.file_list[0], 'r') as f:
            target_class = f['target_class'][:]
        return target_class





def load_dataset(data_file, label_data_file, used_IDs, class_correspondence_table):
    hsi_dataset_list = []
    for i, ID in enumerate(data_file['ID']):
        hsi_dataset = HSIDataset()
        
        flag = hsi_dataset.set_ID(ID, used_IDs)
        if flag != 1:
            continue
        
        sample_ID = data_file['SampleID'][i][:3]
        flag = hsi_dataset.set_sample_ID(sample_ID)
        if flag != 1:
            continue
                
        image_file_name = data_file['Filename'][i]
        flag = hsi_dataset.set_image_and_wavelength_information(image_file_name)
        if flag != 1:
            continue
        
        flag = hsi_dataset.set_label(label_data_file, class_correspondence_table)
        if flag != 1:
            continue
        
        flag = hsi_dataset.set_label_image()
        if flag != 1:
            continue
        
        flag = hsi_dataset.set_mask_image()
        if flag != 1:
            continue
        
        print(f'sample {ID} finished.')
        hsi_dataset_list.append(hsi_dataset)
    
    print('finished loading dataset')
    print(PARTITION)
    return hsi_dataset_list

def preprocess(hsi_dataset_list, num_of_components, patch_size, test_size, random_state, used_IDs, class_correspondence_table):
    """
    train_IDs, test_IDs = train_test_split(used_IDs, test_size=test_size, random_state=random_state)
    train_preprocessed_hsi_dataset_list = []
    test_preprocessed_hsi_dataset_list = []
    """
    IDs_1 = ['348', '230', '227', '157', '233', '308']
    IDs_2 = ['260', '218', '212']
    IDs_3 = ['251', '263', '150', '342', '187', '236', '324']
    IDs_4 = ['181', '266', '290', '160', '361']
    IDs_5 = ['199', '284', '321', '163', '215', '333']
    preprocessed_hsi_dataset_list_1 = []
    preprocessed_hsi_dataset_list_2 = []
    preprocessed_hsi_dataset_list_3 = []
    preprocessed_hsi_dataset_list_4 = []
    preprocessed_hsi_dataset_list_5 = []
    preprocessed_hsi_dataset_list = []
    for hsi_dataset in hsi_dataset_list:
        hsi_dataset.mask()
        
        save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411')
        save_name = 'NGMeet' + str(hsi_dataset.ID)
        hsi_dataset.show_spectral_of_image(hsi_dataset.image, save_directory=save_directory, save_name=save_name)
        
        save_name = 'noisy' + str(hsi_dataset.ID)
        hsi_dataset.show_spectral_of_image(hsi_dataset.noisy_image, save_directory=save_directory, save_name=save_name)
        
        # hsi_dataset.moving_average_of_image(num_of_weight=5)
        # save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411', 'after')
        # hsi_dataset.show_spectral_of_image(save_directory=save_directory)
        hsi_dataset.decompose(num_of_components)
        hsi_dataset.split(patch_size, class_correspondence_table)
        
        preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
        """
        if hsi_dataset.ID in train_IDs:
            train_preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
        elif hsi_dataset.ID in test_IDs:
            test_preprocessed_hsi_dataset_list.extend(hsi_dataset.split_dataset)
        else:
            print(f'No ID: {hsi_dataset.ID}')
            continue
        """
        
        if hsi_dataset.ID in IDs_1:
            preprocessed_hsi_dataset_list_1.extend(hsi_dataset.split_dataset)
        elif hsi_dataset.ID in IDs_2:
            preprocessed_hsi_dataset_list_2.extend(hsi_dataset.split_dataset)
        elif hsi_dataset.ID in IDs_3:
            preprocessed_hsi_dataset_list_3.extend(hsi_dataset.split_dataset)
        elif hsi_dataset.ID in IDs_4:
            preprocessed_hsi_dataset_list_4.extend(hsi_dataset.split_dataset)
        elif hsi_dataset.ID in IDs_5:
            preprocessed_hsi_dataset_list_5.extend(hsi_dataset.split_dataset)
        else:
            print(f'No ID: {hsi_dataset.ID}')
            print(f'label: {hsi_dataset.label}')
            continue
        
        print(f'preprocess of {hsi_dataset.ID} finished')
    
    average = average_for_hsiList(preprocessed_hsi_dataset_list)
    std = std_for_hsiList(preprocessed_hsi_dataset_list, average)
    preprocessed_hsi_dataset_list_1 = standardization_for_hsi_dataset_list(preprocessed_hsi_dataset_list_1, average, std)
    preprocessed_hsi_dataset_list_2 = standardization_for_hsi_dataset_list(preprocessed_hsi_dataset_list_2, average, std)
    preprocessed_hsi_dataset_list_3 = standardization_for_hsi_dataset_list(preprocessed_hsi_dataset_list_3, average, std)
    preprocessed_hsi_dataset_list_4 = standardization_for_hsi_dataset_list(preprocessed_hsi_dataset_list_4, average, std)
    preprocessed_hsi_dataset_list_5 = standardization_for_hsi_dataset_list(preprocessed_hsi_dataset_list_5, average, std)
    
    '''
    new_average = average_for_hsiList(train_preprocessed_hsi_dataset_list)
    new_std = std_for_hsiList(train_preprocessed_hsi_dataset_list, new_average)
    print(f'after standarization (average, std): ({new_average}, {new_std})')
    '''
    
    print('preprocess finished')
    print(PARTITION)
    return preprocessed_hsi_dataset_list_1, preprocessed_hsi_dataset_list_2, preprocessed_hsi_dataset_list_3, preprocessed_hsi_dataset_list_4, preprocessed_hsi_dataset_list_5, preprocessed_hsi_dataset_list


def average_for_hsiList(hsi_dataset_list):
    length = len(hsi_dataset_list)
    num_spectral = np.shape(hsi_dataset_list[0].image)[0]
    width = np.shape(hsi_dataset_list[0].image)[1]
    height = np.shape(hsi_dataset_list[0].image)[2]
    
    hsi_list = []
    for hsi_dataset in hsi_dataset_list:
        hsi_list.append(hsi_dataset.image)
    hsi_list_numpy = np.array(hsi_list)
    
    return np.array([np.sum(hsi_list_numpy[:,i,:,:]) / (length * width * height) for i in range(num_spectral)])


def std_for_hsiList(hsi_dataset_list, average):
    length = len(hsi_dataset_list)
    num_spectral = np.shape(hsi_dataset_list[0].image)[0]
    width = np.shape(hsi_dataset_list[0].image)[1]
    height = np.shape(hsi_dataset_list[0].image)[2]

    hsi_list = []
    for hsi_dataset in hsi_dataset_list:
        hsi_list.append(hsi_dataset.image)
    hsi_list_numpy = np.array(hsi_list)
    
    return np.array([np.sqrt(np.sum(np.square(hsi_list_numpy[:,i,:,:] - average[i])) / (length * width * height)) for i in range(num_spectral)])


def standardization_for_hsi_dataset_list(hsi_dataset_list, average, std):
    std_hsi_dataset_list = []
    for hsi_dataset in hsi_dataset_list:
        for i in range(len(std)):
            hsi_dataset.image[i] = (hsi_dataset.image[i] - average[i]) / std[i]
        std_hsi_dataset_list.append(hsi_dataset)
    
    return std_hsi_dataset_list


def write_hsi_dataset_list_as_h5(hsi_dataset_list, save_directory=None, phase='sample', target_class=None):
    import shutil
    
    if save_directory is None:
        if phase == 'train':
            save_directory = 'train_dataset'
        elif phase == 'test':
            save_directory = 'test_dataset'
        else:
            save_directory = 'sample_dataset'
    
    if os.path.isdir(save_directory):
        shutil.rmtree(save_directory)
        os.mkdir(save_directory)
    else:
        os.mkdir(save_directory)
    
    for i, hsi_dataset in enumerate(hsi_dataset_list):
        save_name = 'dataset' + str(hsi_dataset.ID) + '_' + str(i) + '.h5'
        save_path = os.path.join(save_directory, save_name)
        
        with h5py.File(save_path, 'w') as f:
            f.create_dataset('ID', data=hsi_dataset.ID)
            f.create_dataset('image', data=hsi_dataset.image)
            f.create_dataset('label', data=np.array([hsi_dataset.label]))
            f.create_dataset('original_image', data=hsi_dataset.original_image)
            f.create_dataset('target_class', data=np.array(target_class))
        
        print(f'write of sample {hsi_dataset.ID} finished')
    
    print('write finished')
    print(PARTITION)



def class_count(dataset_list, target_class):
    count_dict = {}
    id_dict = {}
    
    for item in target_class:
        count_dict[item] = 0
        id_dict[item] = []
    
    for dataset in dataset_list:
        if dataset.label in target_class:
            count_dict[dataset.label] += 1
            id_dict[dataset.label].append(dataset.ID)
    return count_dict, id_dict

def show_hsi_dataset_montage(hsi_dataset_list, save_directory=None, save_name='montage.jpg'):
    rgb_image_list = []
    
    for hsi_dataset in hsi_dataset_list:
        rgb_image = util.get_display_image(hsi_dataset.original_image, channel=401)
        if rgb_image.shape[0] != 64 or rgb_image.shape[1] != 64 or rgb_image.shape[2] != 3:
            print('_____________')
            print(hsi_dataset.ID)
            print(rgb_image.shape)
        if rgb_image.dtype == 'float':
            rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image_list.append(rgb_image)
    rgb_image_list = np.array(rgb_image_list, dtype='uint8')
    
    montage = skimage.util.montage(rgb_image_list, multichannel=True)
    
    if save_directory is None:
        save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411')
    
    save_path = os.path.join(save_directory, save_name)
    
    skimage.io.imsave(save_path, montage)

def show_hsi_montage(hsi_list, save_directory=None, save_name='montage.jpg'):
    rgb_image_list = []
    
    for hsi in hsi_list:
        rgb_image = util.get_display_image(hsi, channel=401)
        if rgb_image.shape[0] != 64 or rgb_image.shape[1] != 64 or rgb_image.shape[2] != 3:
            print('_____________')
            print(rgb_image.shape)
        if rgb_image.dtype == 'float':
            rgb_image = (rgb_image * 255).astype('uint8')
        rgb_image_list.append(rgb_image)
    rgb_image_list = np.array(rgb_image_list, dtype='uint8')
    
    montage = skimage.util.montage(rgb_image_list, multichannel=True)
    
    if save_directory is None:
        save_directory = os.path.join(CONF['Directories']['outputDir'], 'T20230411')
    
    save_path = os.path.join(save_directory, save_name)
    
    skimage.io.imsave(save_path, montage)

if __name__ == '__main__':
    import pandas as pd
    from collections import Counter
    
    remove_directries = [
        'output-python/T20230411', 'train_dataset', 'test_dataset', 'sample_dataset', 
        'dataset_all', 'dataset_1', 'dataset_2', 'dataset_3', 'dataset_4', 'dataset_5'
        ]
    empty_directory(remove_directries)
    
    num_of_components = 64 * 6
    patch_size = 64
    test_size = 0.1
    random_state = 1
    
    '''
    class_correspondence_table = {
        'Malignant': 0,
        'Benign': 1,
        'Non Cancer': 2
        }
    '''
    class_correspondence_table = {
        'Malignant': 0,
        'Benign': 1
        }
    
    target_class = list(class_correspondence_table.values())
    
    data_info_path = os.path.join(CONF['Directories']['importDir'], CONF['File Names']['dataInfoTableName'])
    data_file = pd.read_excel(data_info_path)
    
    label_data_info_path = os.path.join(CONF['Directories']['importDir'], CONF['File Names']['diagnosisInfoTableName'])
    label_data_file = pd.read_excel(label_data_info_path)
    
    used_IDs = [
        '150', '157', '160', '163', '175', '181', '187', '193', '196', '199', '205', '212', '215', '218', '227', '230', 
        '233', '236', '251', '260', '263', '266', '284', '290', '296', '308', '321', '324', '333', '342', '348', '352', '361'
        ]
    '''
    used_IDs = ['150', '157']
    '''
    
    hsi_dataset_list = load_dataset(data_file, label_data_file, used_IDs, class_correspondence_table)    
    hsi_dataset_list_1, hsi_dataset_list_2, hsi_dataset_list_3, hsi_dataset_list_4, hsi_dataset_list_5, hsi_dataset_list_all = preprocess(hsi_dataset_list, num_of_components, patch_size, test_size, random_state, used_IDs, class_correspondence_table)
    
    count_dict, id_dict = class_count(hsi_dataset_list_all, target_class)
    print(f'dataset_list_all num: {len(hsi_dataset_list_all)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    count_dict, id_dict = class_count(hsi_dataset_list_1, target_class)
    print(f'dataset_list_1 num: {len(hsi_dataset_list_1)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    count_dict, id_dict = class_count(hsi_dataset_list_2, target_class)
    print(f'dataset_list_2 num: {len(hsi_dataset_list_2)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    count_dict, id_dict = class_count(hsi_dataset_list_3, target_class)
    print(f'dataset_list_3 num: {len(hsi_dataset_list_3)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    count_dict, id_dict = class_count(hsi_dataset_list_4, target_class)
    print(f'dataset_list_4 num: {len(hsi_dataset_list_4)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    count_dict, id_dict = class_count(hsi_dataset_list_5, target_class)
    print(f'dataset_list_5 num: {len(hsi_dataset_list_5)}, count_dict: {count_dict}, id_0: {Counter(id_dict[0])}, id_1: {Counter(id_dict[1])}')
    
    write_hsi_dataset_list_as_h5(hsi_dataset_list_all, save_directory='dataset_all', target_class=target_class)
    write_hsi_dataset_list_as_h5(hsi_dataset_list_1, save_directory='dataset_1', target_class=target_class)
    write_hsi_dataset_list_as_h5(hsi_dataset_list_2, save_directory='dataset_2', target_class=target_class)
    write_hsi_dataset_list_as_h5(hsi_dataset_list_3, save_directory='dataset_3', target_class=target_class)
    write_hsi_dataset_list_as_h5(hsi_dataset_list_4, save_directory='dataset_4', target_class=target_class)
    write_hsi_dataset_list_as_h5(hsi_dataset_list_5, save_directory='dataset_5', target_class=target_class)
    
    show_hsi_dataset_montage(hsi_dataset_list_1, save_name='dataset1_montage.jpg')
    show_hsi_dataset_montage(hsi_dataset_list_2, save_name='dataset2_montage.jpg')
    show_hsi_dataset_montage(hsi_dataset_list_3, save_name='dataset3_montage.jpg')
    show_hsi_dataset_montage(hsi_dataset_list_4, save_name='dataset4_montage.jpg')
    show_hsi_dataset_montage(hsi_dataset_list_5, save_name='dataset5_montage.jpg')
    show_hsi_dataset_montage(hsi_dataset_list_all, save_name='dataset_all_montage.jpg')
    
    
    """
    write_hsi_dataset_list_as_h5(train_dataset_list, phase='train', target_class=target_class)
    write_hsi_dataset_list_as_h5(test_dataset_list, phase='test', target_class=target_class)
    print(f'num of train: {len(train_dataset_list)}, num of test: {len(test_dataset_list)}')
    print(f'num of train per class: {class_count(train_dataset_list, target_class)}')
    print(f'num of test per class: {class_count(test_dataset_list, target_class)}')
    
    show_hsi_dataset_montage(train_dataset_list, save_name='train_montage.jpg')
    show_hsi_dataset_montage(test_dataset_list, save_name='test_montage.jpg')
    show_hsi_dataset_montage(all_dataset_list, save_name='all_montage.jpg')
    """
    