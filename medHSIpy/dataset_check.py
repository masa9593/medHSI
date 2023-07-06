from torch.utils.data import DataLoader
import os
import skimage

from dataset import MedHSIDataset
from tools import util

conf = util.parse_config()

def show_hsi(hsi, save_directory=None, save_name='hsi.jpg'):
    rgb_image = util.get_display_image(hsi, channel = 401)
    if rgb_image.dtype == 'float':
        rgb_image = (rgb_image * 255).astype('uint8')
        
    if save_directory is None:
        save_directory = os.path.join(conf['Directories']['outputDir'], 'T20230411')
    save_path = os.path.join(save_directory, save_name)
        
    skimage.io.imsave(save_path, rgb_image)
        
    return 1

batch_size = 4
num_of_component = 64 * 6

train_dataset = MedHSIDataset('train_dataset')
test_dataset = MedHSIDataset('test_dataset')

target_class = train_dataset.return_target_class()

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

for i, dataset in enumerate(train_dataloader):
    for b in range(len(dataset['image'])):
        ID = dataset['ID'][b]
        image = dataset['image'][b][0]
        label = dataset['label'][b]
        original_image = dataset['original_image'][b]
        
        for x, image_2D in enumerate(image):
            for y, image_1D in enumerate(image_2D):
                for z, image_0D in enumerate(image_1D):
                    if image_0D != 0:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        dataset_image = train_dataset[i*batch_size+b]['image']
        dataset_original_image = train_dataset[i*batch_size+b]['original_image']
        print(f'train_dataset ID: {ID}, No: {i}, batch: {b}')
        print(f'image shape: {image.shape}, original_image shape: {original_image.shape}')
        print(f'dataset image: {train_dataset[i*batch_size+b]}')
        print(f'dataset image: {dataset_image[x][y][z]}, original_image: {dataset_original_image[num_of_component//2+x][y][z]}')
        print(f'image: {image[x][y][z]}, original_image: {original_image[num_of_component//2+x][y][z]}')
        print(f'label: {label}')
        show_hsi(original_image.detach().numpy().copy(), save_name=f'train_dataset_No{i}_batch{b}.jpg')

for i, dataset in enumerate(test_dataloader):
    for b in range(len(dataset['image'])):
        image = dataset['image'][b][0]
        label = dataset['label'][b]
        original_image = dataset['original_image'][b]
        
        for x, image_2D in enumerate(image):
            for y, image_1D in enumerate(image_2D):
                for z, image_0D in enumerate(image_1D):
                    if image_0D != 0:
                        break
                else:
                    continue
                break
            else:
                continue
            break
        
        print(f'test_dataset No: {i}, batch: {b}')
        print(f'image shape: {image.shape}, original_image shape: {original_image.shape}')
        print(f'image: {image[x][y][z]}, original_image: {original_image[num_of_component//2+x][y][z]}')
        print(f'label: {label}')
        show_hsi(original_image.detach().numpy().copy(), save_name=f'test_dataset_No{i}_batch{b}.jpg')
