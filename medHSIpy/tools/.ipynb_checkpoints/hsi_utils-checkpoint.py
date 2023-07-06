import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import cv2
import pandas as pd


######################### Path #########################

def get_filenames(folder, target='.mat'):
    origListdir = os.listdir(folder)
    origListdir = [os.path.join(folder, x) for x in origListdir if target in x]
    return origListdir

def makedir(fpath):
    try:
        os.mkdir(fpath)
    except OSError as error:
        print('folder exists')
    
######################### Messages #########################        
def not_implemented():
    print('Not yet implemented.')
    return

def not_supported(varname):
    print('Not supported [', varname, '].')
    return

######################### Config #########################
import configparser

dirSep = '\\'

def get_base_dir():
    cwd = os.getcwd()
    parts = cwd.split("/")
    parts = parts[0: parts.index('medHSI-main')+1]
    parts.insert(1, os.sep)
    base_dir = os.path.join(*parts)
    return base_dir

def get_module_path():
    module_path = os.path.join(get_base_dir(), 'medHSIpy', 'src')
    return module_path

def get_config_path():
    settings_file = os.path.join(get_base_dir(), "conf", "config.ini")
    return settings_file

def parse_config():
    settings_file = get_config_path()
    config = configparser.ConfigParser()
    config.read(settings_file, encoding = 'utf-8')
    # print("Loading from settings conf/config.ini \n")
    # print("Sections")
    # print(config.sections())
    return config

conf = parse_config()

def get_savedir():
    dirName = os.path.join(conf['Directories']['outputDir'], conf['Folder Names']['pythonTest'])
    return dirName

def get_tripletdir():
    dirName = os.path.join(conf['Directories']['matDir'], conf['Folder Names']['triplets'] )
    return dirName

######################### Load #########################
    
from scipy.io import loadmat
import h5py
import mat73

###expected form of hsi data: Height x Width x Wavelengths

def load_from_h5(fname):
    val = h5py.File(fname, 'r')
    return val

def load_from_mat73(fname, varname=''):
    mat = mat73.loadmat(fname)
    val = mat[varname]
    return val

def load_from_mat(fname, varname=''):
    val = loadmat(fname)[varname]
    return val

def load_target_mat(fname, varname):
    hsi = load_from_mat73(fname + '_target.mat', varname) 
    return hsi

def load_white_mat(fname, varname):
    hsi = load_from_mat73(fname + '_white.mat', varname) 
    return hsi

def load_black_mat(fname, varname):
    hsi = load_from_mat73(fname + '_black.mat', varname) 
    return hsi

def load_dataset(fpath, sampleType='pixel', ash5=0):
    f = load_from_h5(fpath)
    hsiList = []
    keyList = list(f.keys())

    for keyz in keyList:
        if ash5 == 1:
            val = f[keyz]
            print(val)
        else:
            val = f[keyz][:]
        
        if val.shape[2] != 311:
            val = np.transpose(val, [1, 2, 0])
        hsiList.append(val)

    dataList = []
    if sampleType == 'pixel':
        dataList = flatten_hsis(hsiList)
    elif sampleType == 'patch':
        not_implemented()
    elif sampleType == 'image':
        dataList = hsiList
    else:
        not_supported('SampleType')
    return dataList, keyList    

def load_dataset_for_classification(Class, sampleType='pixel', ash5=0):
    IDs = ['150', '157', '160', '163', '175', '181', '187', '193', '196', '199', '205', '212', '215', '218', '227', '230', 
           '233', '236', '251', '260', '263', '266', '284', '290', '296', '308', '321', '324', '333', '342', '348', '352', '361']
    IDs_Actually = np.array([]).astype('int')
    
    hsiList = []
    labelList = []
    
    conf = parse_config()
    dataInfoPath = os.path.join(conf['Directories']['importDir'], conf['File Names']['dataInfoTableName'])
    df = pd.read_excel(dataInfoPath)
    LabelDataInfoPath = os.path.join(conf['Directories']['importDir'], conf['File Names']['diagnosisInfoTableName'])
    ldf = pd.read_excel(LabelDataInfoPath)
    
    for i, ID in enumerate(df['ID']):
        ID = str(ID + 1000)[1:]
        if ID in IDs:
            if ash5 == 1:
                if df['Filename'][i][-3:] == 'raw':
                    fpath = os.path.join(conf['Directories']['dataDir'], df['Filename'][i] + '.h5')
                    f = load_from_h5(fpath)
                    val = f['SpectralImage']
                
                    if val.shape[2] != 401:
                        val = np.transpose(val, [1, 2, 0])
                        
                    fmaskpath = "/home/share/macropathology_hyperspectral_image/structured_data/output/pslRaw/00-Labelme/" + ID + '.png'
                    img = load_label_images_for_classification(fmaskpath)
                    mask = get_labels_from_mask_for_classification(img)
                    
                    mask_reshape = mask[:val.shape[0], :val.shape[1], :val.shape[2]]
                
                    val = val * mask_reshape
                
                    for j, LID in enumerate(ldf['SampleID']):
                        LID = str(LID + 1000)[1:]
                        if df['SampleID'][i][:3] == LID:
                            label = ldf['Type'][j]
                            if label in Class.keys():
                                IDs_Actually = np.append(IDs_Actually, ID)
                                hsiList.append(val)
                                labelList.append(Class[label])
                            break

    dataList = []
    if sampleType == 'pixel':
        dataList = flatten_hsis(hsiList)
    elif sampleType == 'patch':
        not_implemented()
    elif sampleType == 'image':
        dataList = hsiList
    else:
        not_supported('SampleType')
    return dataList, labelList, IDs_Actually

def load_images(fpath, keyList):
    images = []
    for filename in os.listdir(fpath):
        if 'sample'+str(filename[:3]) in keyList:
            img = cv2.imread(os.path.join(fpath,filename))
            if img is not None:
                images.append(img)
    return images

def load_label_images(fpath, keyList):
    imgList = load_images(fpath, keyList)
    rotImgList = [np.transpose(labelImg, [1, 0, 2]) for labelImg in imgList]
    return rotImgList

def load_images_for_classification(fpath):
    img = cv2.imread(fpath)
    return img

def load_label_images_for_classification(fpath):
    labelImg = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
    rotImg = labelImg.transpose(1, 0)
    rotImg = rotImg[:,:,np.newaxis]
    return rotImg


######################### Process #########################

def get_labels_from_mask(imgList):
    labels = []
    for img in imgList: 
        grayIm = cv2.convertScaleAbs(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) 
        #print("Min", np.min(np.min(grayIm)), "and Max ", np.max(np.max(grayIm)))     	
        (thresh, blackAndWhiteImage) = cv2.threshold(grayIm, 170, 255, cv2.THRESH_BINARY)  	
        labelImg = np.logical_not(blackAndWhiteImage)
        labelImg = labelImg.astype(np.int8)
        #plt.imshow(labelImg, cmap='gray')
        #plt.show()
        labels.append(labelImg)
    return labels

def get_labels_from_mask_for_classification(img): 
    if img.shape[2] == 3:
        grayIm = (img[:, :, 0:1] + img[:, :, 1:2] + img[:, :, 2:3]) / 3
    elif img.shape[2] == 1:
        grayIm = img
    else:
        raise ValueError('Unsupposed mask channel number')
    
    labelImg = np.where(grayIm==0, 0, 1)
    return labelImg
    
def center_crop_hsi(hsi, targetHeight=None, targetWidth=None):        

    width = hsi.shape[1]
    height = hsi.shape[0]
    
    if targetWidth is None:
        targetWidth = min(width, height)

    if targetHeight is None:
        targetHeight = min(width, height)

    left = int(np.ceil((width - targetWidth) / 2))
    right = width - int(np.floor((width - targetWidth) / 2))

    top = int(np.ceil((height - targetHeight) / 2))
    bottom = height - int(np.floor((height - targetHeight) / 2))

    if np.ndim(hsi) > 2:
        croppedImg = hsi[top:bottom, left:right,:]
    else:
        croppedImg = hsi[top:bottom, left:right]
        
    return croppedImg

def center_crop_list(dataList, targetHeight = 64, targetWidth = 64):
    croppedData = []
    for data in dataList:
        val = center_crop_hsi(data, targetWidth, targetHeight)
        croppedData.append(val)
            
    return croppedData
    
def normalize_hsi(hsi, white, black):
    normhsi = (hsi - black)  / (white - black + 0.0000001)
    return normhsi

def flatten_hsi(hsi):
    return np.reshape(hsi, (hsi.shape[0] * hsi.shape[1], hsi.shape[2])).transpose() 

def flatten_hsis(imgList):
    X = [flatten_hsi(x) for x in imgList]
    stacked = np.concatenate(X, axis=1).transpose()
    return stacked

def patch_split_hsi(hsi, patchDim=25):
    sb = np.array(hsi.shape)
    st = np.floor(sb[0:2] / patchDim).astype(int)
    cropped = center_crop_hsi(hsi, st[0]*patchDim, st[1]*patchDim)
    # patchIndex = np.meshgrid(np.arange(0,st[0], dtype=np.int32), np.arange(0,st[1], dtype=np.int32))
    patchList = []
    
    '''
    i = 0
    for x,y in zip(patchIndex[0].flatten(), patchIndex[1].flatten()): 
        #v = hsi[(0 + x*patchDim):(patchDim + x*patchDim), (0 + y*patchDim):(patchDim + y*patchDim),:]
        #print(np.max(v.flatten()))
        patchList[++i,:,:,:] = cropped[(0 + x*patchDim):(patchDim + x*patchDim), (0 + y*patchDim):(patchDim + y*patchDim),:]
        #print(np.max(patchList[i,:,:,:].flatten()))
    '''
    
    for x in range(st[0]):
        for y in range(st[1]):
            patch = cropped[(0 + x*patchDim):(patchDim + x*patchDim), (0 + y*patchDim):(patchDim + y*patchDim),:]
            if np.sum(patch) > patchDim:
                patchList.append(patch)
            
    return patchList, len(patchList)

def patch_split_hsiList(hsiList, patchDim=25, channel=401):
    patchHSIList = []
    numListofPatch = []
    for hsi in hsiList:
        patchList, numofPatch = patch_split_hsi(hsi, patchDim)
        patchHSIList.extend(patchList)
        numListofPatch.append(numofPatch)
        
    return np.array(patchHSIList), numListofPatch

######################### Reconstruct 3D #########################
import math

def xyz2rgb(imXYZ):
    d = imXYZ.shape
    r = math.prod(d[0:2])  
    w = d[-1]             
    XYZ = np.reshape(imXYZ, (r, w))
    
    M = [[3.2406, -1.5372, -0.4986],
        [-0.9689, 1.8758, 0.0414],
        [0.0557, -0.2040, 1.0570]]
    sRGB = np.transpose(np.dot(M, np.transpose(XYZ)))

    sRGB = np.reshape(sRGB, d)
    return sRGB

def get_display_image(hsi, imgType = 'srgb', channel = 150):
    recon = []
    if imgType == 'srgb':        
        [m,n,z] = hsi.shape
        
        if channel == 401:
            filename = os.path.join(get_base_dir(), conf['Directories']['paramDir'], 'displayParam.mat')
        elif channel == 311:
            filename = os.path.join(get_base_dir(), conf['Directories']['paramDir'], 'displayParam_311.mat')
        else:
            raise ValueError('Unknown channel number')

        xyz = load_from_mat(filename, 'xyz')
        illumination = load_from_mat(filename, 'illumination')

        colImage = np.reshape( hsi, (m*n, z) )  
        normConst = np.amax(colImage)
        colImage = colImage / (float(normConst) + 0.0000001)
        colImage =  colImage * illumination
        colXYZ = np.dot(colImage, np.squeeze(xyz))
        
        imXYZ = np.reshape(colXYZ, (m, n, 3))
        imXYZ[imXYZ < 0] = 0
        imXYZ = imXYZ / (np.amax(imXYZ) + 0.0000001)
        dispImage_ = xyz2rgb(imXYZ)
        dispImage_[dispImage_ < 0] = 0
        dispImage_[dispImage_ > 1] = 1
        dispImage_ = dispImage_**0.4
        recon =  dispImage_

    elif imgType =='channel':
        recon = hsi[:,:, channel]

    elif imgType =='grey':
        recon = hsi

    else:
        not_supported(imgType)
        
    return recon

def Average_for_hsiList(hsiList):
    shape = np.shape(hsiList)
    return [np.sum(hsiList[:,:,:,i]) / (shape[0] * shape[1] * shape[2]) for i in range(shape[3])]

def Std_for_hsiList(hsiList):
    shape = np.shape(hsiList)
    ave = Average_for_hsiList(hsiList)
    return [np.sqrt(np.sum(np.square(hsiList[:,:,:,i] - ave[i])) / (shape[0] * shape[1] * shape[2])) for i in range(shape[3])]

######################### Plotting #########################
import skimage.util
import skimage.io

def simple_plot(y, figTitle, xLabel, yLabel, fpath):
    plt.plot(np.arange(len(y))+1, y)
    plt.title(figTitle)
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    pltFname = fpath + figTitle.replace(' ', '_') + '.jpg'
    print("Save figure at: ", pltFname) 
    plt.savefig(pltFname)
    plt.show()

def show_display_image(hsiIm, imgType = 'srgb', channel = 150): 
    show_image(get_display_image(hsiIm, imgType, channel))

def show_image(x, figTitle = None, hasGreyScale = False, fpath = ""):
    if hasGreyScale:
        plt.imshow(x, cmap='gray')
    else:
        plt.imshow(x)
    if figTitle is not None:
        plt.title(figTitle)
        pltFname = os.path.join(fpath, figTitle.replace(' ', '_') + '.jpg')
        plt.savefig(pltFname)
        print("Save figure at:"+ pltFname)
    plt.show()
    
def show_montage(dataList, filename = None, savename = 'montage.jpg', imgType = 'srgb', channel = 150):
    #Needs to have same number of dimensions for each image, type float single
    hsiList = np.array([get_display_image(x, imgType, channel) for x in dataList], dtype='float64')
    # print(hsiList.shape)
    if imgType != 'grey':
        m = skimage.util.montage(hsiList, multichannel=True)
        m = (m * 255).astype(np.uint8)
    else:
        m = skimage.util.montage(hsiList)

    if filename == None: 
        filename = os.path.join(conf['Directories']['outputDir'], 'T20211207-python', savename)
    skimage.io.imsave(filename, m)



