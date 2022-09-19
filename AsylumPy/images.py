import os
import h5py
import numpy as np
import pandas as pd
import scipy
import pyUSID as usid
import matplotlib.colors as mcolors

#import imutils
#import cv2
#from scipy.optimize import curve_fit

from .ibw import IBW
from .base.datasets import IgorDataset

class Image(IBW):
    def __init__(self, filePath, save=False, verbose=False):
        super().__init__(filePath, save=save, verbose=verbose)
        self.type = 'IgorIBW_Image'
        if self.type != self.get_type(self.file):
            print('Current data type: ', self.get_type(self.file))
            raise ValueError('This object is not a igor image. ')
            
    def load_channel_info(self, channel):
        channel = self.channel2num(channel)
        
        if self.get_attr(self.file[self.grp_name], 'ScanSize') != self.get_attr(self.file[self.grp_name], 'SlowScanSize'):
            ScanSize = [self.get_attr(self.file[self.grp_name], 'SlowScanSize'), self.get_attr(self.file[self.grp_name], 'ScanSize')]
        else:
            ScanSize = self.get_attr(self.file[self.grp_name], 'ScanSize')
        Dict = {
            'Cmap' : self.get_attr(self.file[self.grp_name], 'ColorMap '+str(channel)),
            'Offset' : self.get_attr(self.file[self.grp_name], 'Display Offset '+str(channel)),
            'Range' : self.get_attr(self.file[self.grp_name], 'Display Range '+str(channel)),
            'ScanSize' : ScanSize # 1.2e-05 == 12um
        }
        return Dict
    
    def get_dim(self, channel):
        channel = self.channel2str(channel)
        
        file = self.file[self.grp_name][channel+'/Raw_Data']
        # Loading positional inforamtaion
        
        if isinstance(file, h5py.Dataset):
            inds = usid.hdf_utils.get_auxiliary_datasets(file, ['Position_Indices'])[0]
            pos_dim_sizes = usid.hdf_utils.get_dimensionality(inds)    
        elif isinstance(file, IgorDataset):
            inds = file['Position_Indices']
            pos_dim_sizes = usid.hdf_utils.get_dimensionality(inds.data)

        pos_dim_names = self.get_attr(inds, 'labels')

        dim = dict()
        for name, length in zip(pos_dim_names, pos_dim_sizes):
            dim[name] = length
        units = self.get_attr(file, 'units')
        return dim['X'], dim['Y'], units
    
    def get_data(self, channel, unit=False):
        channel = self.channel2str(channel)
        file = self.file[self.grp_name][channel+'/Raw_Data']
        try:
            img = usid.USIDataset(file)
        except:
            img = file.data

        if unit:
            return img, self.get_attr(file, 'units')
        else:
            return img
    

    def flatten(self, channel, order = -1):
        ## TODO: Magic mask flattening
        img = self.get_data(channel=channel)
        
        scanlines, scanpoints = img.shape
        
        if order == -1:
            return img

        elif order == 0:
            flatten_img = img[0,:] - np.mean(img[0,:])
            for i in range(1, scanlines):
                flatten_img = np.vstack((flatten_img, img[i,:]-np.mean(img[i,:])))

        else:
            x = np.arange(1, scanpoints+1)
            fit = np.polyfit(x, img[0,:], order)
            poly = np.poly1d(fit)
            flatten_img = img[0,:] - poly(x)
            for i in range(1, scanlines):
                fit = np.polyfit(x, img[i,:], order)
                poly = np.poly1d(fit)
                flatten_img = np.vstack((flatten_img, img[i,:] - poly(x)))
        self.flattenImg = flatten_img
        return flatten_img
        
    def add_prefix(self, unit, minval, maxval=None):
        if maxval == None: maxval = minval
        for key, item in self.prefix.items():
            if abs(minval/item) > 1 and abs(maxval/item) < 1000:
                #img = np.divide(img, item)
                unit = key + unit
                break
        return unit, item

    def load_detail(self, channel):
        """cmap = self.get_attr(self.h5, 'ColorMap '+str(channel))
        Offset = self.get_attr(self.h5, 'Display Offset '+str(channel))
        Range = self.get_attr(self.h5, 'Display Range '+str(channel))
        ScanSize = self.get_attr(self.h5, 'ScanSize') # 1.2e-05 == 12um"""
        DetailDict = {
            'Cmap' : self.get_attr(self.h5, 'ColorMap '+str(channel)),
            'Offset' : self.get_attr(self.h5, 'Display Offset '+str(channel)),
            'Range' : self.get_attr(self.h5, 'Display Range '+str(channel)),
            'ScanSize' : self.get_attr(self.h5, 'ScanSize') # 1.2e-05 == 12um
        }
        return DetailDict

    def do_plane_fit(self, channel, order=1):
        img = self.get_data(channel)
        data = img
        X, Y = np.arange(0, data.shape[0]), np.arange(0, data.shape[1])
        X, Y = np.meshgrid(X, Y)
        XX = X.flatten()
        YY = Y.flatten()
        data = np.c_[XX, YY, data.reshape(1,-1).T]
        if order == 1:
            # best-fit linear plane
            A = np.c_[data[:,0], data[:,1], np.ones(data.shape[0])]
            C,_,_,_ = np.linalg.lstsq(A, data[:,2])    # coefficients
            # evaluate it on grid
            Z = C[0]*X + C[1]*Y + C[2]

            # or expressed using matrix/vector product
            #Z = np.dot(np.c_[XX, YY, np.ones(XX.shape)], C).reshape(X.shape)

        elif order == 2:
            # best-fit quadratic curve
            A = np.c_[np.ones(data.shape[0]), data[:,:2], np.prod(data[:,:2], axis=1), data[:,:2]**2]
            C,_,_,_ = scipy.linalg.lstsq(A, data[:,2])

            # evaluate it on a grid
            Z = np.dot(np.c_[np.ones(XX.shape), XX, YY, XX*YY, XX**2, YY**2], C).reshape(X.shape)
        return img - Z

    def color_map(cmap):
        try:
            return cmap_dict[cmap]
        except KeyError:
            print(f'There is not color map named {cmap}.')

    def __getitem__(self, index):
        try:
            return self.get_data(index)
        except:
            return super().__getitem__(index)

def _load_colors(root='color_tables'):
    cmap_dict = {'Grays256':'gray', 'VioletOrangeYellow':'PuOr'}
    current_path = os.path.abspath(os.path.dirname(__file__))
    color_root = os.path.join(current_path, root)
    colors = [f for f in os.listdir(color_root) if os.path.isdir(os.path.join(color_root, f))]
    
    for color in colors:
        color_folder = os.path.join(color_root, color)
        rgb = [f for f in os.listdir(color_folder) if f.endswith('csv') and not f.startswith('.')]
        rgb = {name[0].lower() : np.array(pd.read_csv(path, header=None)[1]) for name, path in zip(rgb, [os.path.join(color_folder, c) for c in rgb])}
        cmap_dict.update({color: _make_colormap(**rgb, name=color)})
    
    return cmap_dict

def _make_colormap(r, g, b, name='Custom'):
    """Return a LinearSegmentedColormap
    r, g, b: a sequence of RGB floats from IgorPro
    """
    r, g, b = r/np.max(r), g/np.max(g), b/np.max(b)
    cdict = {'red': [], 'green': [], 'blue': []}

    for i in range(len(r)):
        index = i/len(r)
        if i == 0:
            r1, g1, b1 = (None, None, None)
            index = 0.0 # index should start from 0.0
        else:
            r1, g1, b1 = r[i - 1], g[i - 1], b[i - 1]
        if i == len(r) -1:
            r2, g2, b2 = (None, None, None)
            index = 1.0 # index should end with 1.0
        else:
            r2, g2, b2 = r[i + 1], g[i + 1], b[i + 1]
        cdict['red'].append([index, r1, r2])
        cdict['green'].append([index, g1, g2])
        cdict['blue'].append([index, b1, b2])
    return mcolors.LinearSegmentedColormap(name, cdict)

## Color Map list
cmaplist = ['gray','viridis', 'plasma', 'inferno', 'magma', 'cividis', 
            'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn',
            'binary', 'gist_yarg', 'gist_gray', 'gray', 'bone', 'pink',
            'spring', 'summer', 'autumn', 'winter', 'cool', 'Wistia',
            'hot', 'afmhot', 'gist_heat', 'copper', 'PiYG', 'PRGn',
            'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn', 'Spectral',
            'coolwarm', 'bwr', 'seismic', 'twilight', 'twilight_shifted', 'hsv',
            'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2', 'Set1', 'Set2',
            'Set3', 'tab10', 'tab20', 'tab20b', 'tab20c', 'flag', 'prism',
            'ocean', 'gist_earth', 'terrain', 'gist_stern',
            'gnuplot', 'gnuplot2', 'CMRmap', 'cubehelix', 'brg',
            'gist_rainbow', 'rainbow', 'jet', 'nipy_spectral', 'gist_ncar']

cmap_dict = _load_colors(root='color_tables')