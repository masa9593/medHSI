import numpy as np
import os
import os.path
    
######################### Messages #########################        

def not_supported(varname):
    print('Not supported [', varname, '].')
    return

######################### Config #########################
import configparser

def get_base_dir():
    cwd = os.getcwd()
    parts = cwd.split("/")
    parts = parts[0: parts.index('medHSI-main')+1]
    parts.insert(1, os.sep)
    base_dir = os.path.join(*parts)
    return base_dir

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

######################### Load #########################

from scipy.io import loadmat

def load_from_mat(fname, varname=''):
    val = loadmat(fname)[varname]
    return val

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
    if hsi.shape[0] == channel:
        hsi = np.transpose(hsi, (1,2,0))
    
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
