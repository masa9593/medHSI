# -*- coding: utf-8 -*-
import os
import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    import hsi_utils
else:
    from . import hsi_utils
    from . import hsi_decompositions

# Image size should be multiple of 32
DEFAULT_HEIGHT = 64

def load_data():
    conf = hsi_utils.parse_config()
    fpath = os.path.join(conf['Directories']['outputDir'], conf['Folder Names']['datasets'], "hsi_pslRaw_full.h5")
    
    dataList, keyList = hsi_utils.load_dataset(fpath, 'image', ash5 = 1)
    
    '''
    sampleIds = []
    print("Target Sample Images", sampleIds)

    keepInd = [keyList.index('sample' + str(id)) for id in sampleIds]
    print(keepInd)

    if not keepInd is None:
        dataList = [ dataList[i] for i in keepInd]
    '''

    # Prepare input data
    croppedData = hsi_utils.center_crop_list(dataList, DEFAULT_HEIGHT, DEFAULT_HEIGHT, True)

    # Prepare labels
    labelpath = os.path.join(conf['Directories']['outputDir'],  conf['Folder Names']['labelsManual'])
    labelRgb = hsi_utils.load_label_images(labelpath, keyList)

    # for (x,y) in zip(dataList, labelRgb):
    #     if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
    #         print('Error: images have different size!')
    #         print(x.shape)
    #         print(y.shape)
    #         hsi_utils.show_display_image(x)
    #         plt.imshow(y, cmap='gray')
    #         plt.show()

    labelImages = hsi_utils.get_labels_from_mask(labelRgb)
    croppedLabels = hsi_utils.center_crop_list(labelImages)

    # for (x,y) in zip(croppedData, croppedLabels):
    #     hsi_utils.show_display_image(x)
    #     print(np.max(y), np.min(y))
    #     plt.imshow(y, cmap='gray')
    #     plt.show()

    return croppedData, croppedLabels

def get_train_test(): 
    from sklearn.model_selection import train_test_split

    croppedData, croppedLabels = load_data_for_classification()

    croppedData = np.array(croppedData, dtype=np.float32)
    croppedLabels = np.array(croppedLabels, dtype=np.float32)
    print(croppedData.shape)
    print(croppedLabels.shape)
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(croppedData,  croppedLabels, test_size=0.1, random_state=42)
    print('xtrain: ', len(x_train_raw),', xtest: ', len(x_test_raw))

    # for (x,y) in zip(x_train_raw, y_train_raw):
    #     hsi_utils.show_display_image(x)
    #     hsi_utils.show_image(y)

    return x_train_raw, x_test_raw, y_train, y_test

def load_data_for_classification(Class, height=DEFAULT_HEIGHT, n_components=401, is_split=False, show_image=False, random_state=None):
    from sklearn.model_selection import train_test_split
    dataList, labelList, IDs = hsi_utils.load_dataset_for_classification(Class, sampleType='image', ash5=1)
        
    '''
    sampleIds = []
    print("Target Sample Images", sampleIds)

    keepInd = [keyList.index('sample' + str(id)) for id in sampleIds]
    print(keepInd)

    if not keepInd is None:
        dataList = [ dataList[i] for i in keepInd]
    '''
    
    dataList_train, dataList_test, labelList_train, labelList_test, IDs_train, IDs_test = train_test_split(dataList,  labelList, IDs, test_size=0.1, random_state=random_state)
    print(f'train IDs: {IDs_train}, test IDs: {IDs_test}')

    # Prepare input data
    if is_split:
        croppedData_train, numListofPatch_train = hsi_utils.patch_split_hsiList(dataList_train, patchDim=height, channel=401)
        croppedData_test, numListofPatch_test = hsi_utils.patch_split_hsiList(dataList_test, patchDim=height, channel=401)
    else:
        croppedData = hsi_utils.center_crop_list(dataList, targetHeight=DEFAULT_HEIGHT, targetWidth=DEFAULT_HEIGHT)
    
    '''
    if show_image:
        hsi_utils.show_montage(croppedData_train, channel=401, savename='train_croppedData_montage.jpg')
        hsi_utils.show_montage(croppedData_test, channel=401, savename='test_croppedData_montage.jpg')
    '''
    
    '''
    PCA = hsi_decompositions.decompose(croppedData, method='pca', n_components=n_components)
    decomCroppedDataList = []
    for hsimg in croppedData:
        decomCroppedData = PCA.transform(hsi_utils.flatten_hsi(hsimg).transpose())
        decomCroppedData = decomCroppedData.reshape((hsimg.shape[0], hsimg.shape[1], -1))
        decomCroppedDataList.append(decomCroppedData)    
    '''
    
    '''
    decomCroppedDataList_train = hsi_decompositions.decompose(croppedData_train, method='crop', n_components=n_components)
    decomCroppedDataList_test = hsi_decompositions.decompose(croppedData_test, method='crop', n_components=n_components)
    '''
    
    # Prepare label data
    if is_split:
        patchLabelList_train = []
        
        for numofPatch, label in zip(numListofPatch_train, labelList_train):
            patchLabelList_train.extend([label] * numofPatch)
        
        labelList_train = patchLabelList_train
        
        
        patchLabelList_test = []
        
        for numofPatch, label in zip(numListofPatch_test, labelList_test):
            patchLabelList_test.extend([label] * numofPatch)
        
        labelList_test = patchLabelList_test
        
    # for (x,y) in zip(dataList, labelRgb):
    #     if x.shape[0] != y.shape[0] or x.shape[1] != y.shape[1]:
    #         print('Error: images have different size!')
    #         print(x.shape)
    #         print(y.shape)
    #         hsi_utils.show_display_image(x)
    #         plt.imshow(y, cmap='gray')
    #         plt.show()

    # for (x,y) in zip(croppedData, croppedLabels):
    #     hsi_utils.show_display_image(x)
    #     print(np.max(y), np.min(y))
    #     plt.imshow(y, cmap='gray')
    #     plt.show()

    return croppedData_train, croppedData_test, labelList_train, labelList_test

def get_train_test_for_clasification(Class, height=DEFAULT_HEIGHT, n_components=401, is_split=False, show_image=False, random_state=None): 
    croppedData_train, croppedData_test, label_train, label_test = load_data_for_classification(Class, height=height, n_components=n_components, is_split=is_split, show_image=show_image, random_state=random_state)

    croppedData_train = np.array(croppedData_train, dtype=np.float32)
    croppedData_test = np.array(croppedData_test, dtype=np.float32)
    label_train = np.array(label_train, dtype=np.int8)
    label_test = np.array(label_test, dtype=np.int8)

    print('image data size', croppedData_train.shape)
    print('label size', label_train.shape)
    print('xtrain: ', len(croppedData_train),', xtest: ', len(croppedData_test), 'ytrain: ', len(label_train),', ytest: ', len(label_test))

    # for (x,y) in zip(x_train_raw, y_train_raw):
    #     hsi_utils.show_display_image(x)
    #     hsi_utils.show_image(y)

    return croppedData_train, croppedData_test, label_train, label_test

from contextlib import redirect_stdout
from datetime import date

def get_model_filename(model, suffix='', extension='txt'):
    today = date.today()
    model_name = str(today)
    savedir = os.path.join(hsi_utils.conf['Directories']['outputDir'],
        hsi_utils.conf['Folder Names']['pythonTest'])
    filename = os.path.join(savedir, model_name + suffix + '.' + extension)
    return filename

def save_model_summary(model):
    filename = get_model_filename(model, '_modelsummary', 'txt')
    if __name__ != "__main__":
        print("Saved at: ", filename)
    with open(filename, 'w', encoding='utf-8') as f:
        with redirect_stdout(f):
            model.summary()

from keras.utils.vis_utils import plot_model

def save_model_graph(model):
    filename = get_model_filename(model, '_modelgraph', 'png')
    if __name__ != "__main__":
        print("Saved at: ", filename)
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

def save_model_info(model):
    save_model_summary(model)
    save_model_graph(model)

def show_label_montage(): 
    croppedData, croppedLabels = load_data()
    filename = os.path.join(hsi_utils.conf['Directories']['outputDir'], 'T20211207-python', 'normalized-montage.jpg')
    hsi_utils.show_montage(croppedData, filename, 'srgb')
    filename = os.path.join(hsi_utils.conf['Directories']['outputDir'], 'T20211207-python', 'labels-montage.jpg')
    hsi_utils.show_montage(croppedLabels, filename, 'grey')


def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

def visualize(hsi, gt, pred):
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(hsi_utils.get_display_image(hsi))

    plt.subplot(1,3,2)
    plt.title("Ground Truth")
    plt.imshow(gt)

    plt.subplot(1,3,3)
    plt.title("Prediction")
    plt.imshow(pred)
    plt.show()