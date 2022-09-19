import os
import h5py
import sidpy

from .base.datasets import IgorDataset
from .base.file import IgorFile
from .base.groups import IgorGroup
from .base.simple import walk_dict

class IBW(object):
    def __init__(self, filePath, grp_name= 'Measurement_000' ,save = False, verbose = False):
        if isinstance(filePath, str):
            abspath = os.path.abspath(filePath)
            self.folder_path, filename = os.path.split(abspath)
            if filename.split('.')[-1] != 'ibw':
                raise TypeError('Input file should have ibw format not {}.'.format(filename.split('.')[-1]))
            self.filename = filename[:-4]
            
            self.file = self.load_ibw(abspath, save, verbose)
        else:
            self.file = self.load_ibw(filePath, save, verbose)
            
        self.grp_name = grp_name+'/'
        self.channels = self.get_channel_list(self.file[self.grp_name])
        self.prefix = _prefix

    def load_ibw(self, ibwPath, save=False, verbose=False):

        if not save:
            from .base.igorIBW import IgorIBWTranslator as igor
            translator = igor()
            file = translator.translate(ibwPath, verbose)
        else:
            from .base.igor2h5 import IgorIBWTranslator as igor
            translator = igor()
            h5Path =  os.path.join(self.folder_path, self.filename + '.h5')
            if os.path.exists(h5Path):
                file = h5py.File(h5Path, mode='r')
            else:
                file = translator.translate(ibwPath, verbose)
        return file

    def get_type(self, file):
        return self.get_attr(file, 'data_type')
    
    def calc_prefix(self, val):
        for key, item in self.prefix.items():
            if val / item > 0 and val / item < 1000:
                return key, item
    
    def print_tree(self):
        if isinstance(self.file, (h5py.File, h5py.Group)):
            print(sidpy.hdf.hdf_utils.print_tree(self.file))
        elif isinstance(self.file, (IgorFile, IgorGroup)):
            print(walk_dict(self.file))
    
    def print_channels(self):
        for k, v in self.channels.items():
            print(f'{k} : {v}')

    def get_channel_name(self, channel):
        channel = self.channel2str(channel)
        return self.channels[channel]

    def get_channel_list(self, measurment):
        # Get Measurement_000 folder
        channel = dict()
        for group in measurment:
            if group[:5] == 'Chann':
                channel[group] = self.get_attr(measurment[group+'/Raw_Data'], 'quantity')
        return channel
    
    def channel2str(self, channel):
        return 'Channel_'+str(channel).zfill(3) if isinstance(channel, int) else channel

    def channel2num(self, channel):
        if isinstance(channel, int):
            return channel
        elif isinstance(channel, str):
            channel = 0 if not channel[-3:].strip('0') == '0' else channel[-3:].strip('0')
            return channel
        else:
            raise TypeError('Check the types of input. Current input is {}'.format(type(channel)))

    def get_channel_by_name(self, name):
        for k, v in self.channels.items():
            if v == name:
                return k
        return None

    def get_attr(self, file, attr):
        if isinstance(file, (h5py.File, h5py.Group, h5py.Dataset)):
            return sidpy.hdf.hdf_utils.get_attr(file, attr)
        elif isinstance(file, (IgorFile, IgorGroup, IgorDataset)):
            return getattr(file, attr)

    def get_attr_list(self, file):
        ## Return Attributes dictionary
        # Initial ScanSize
        # Initial ScanAngle
        # Initial ScanLines
        # Initial ScanPoints
        # Initial ScanRate
        # ImageNote
        if isinstance(file, (h5py.File, h5py.Group)):
            attrs = {}
            for key, val in sidpy.hdf.hdf_utils.get_attributes(file).items():
                attrs[key] = val
        else:
            attrs = file.__dict__.keys()
            for i in ['data', 'group', 'datasets']:
                try:
                    del attrs[i]
                except:
                    continue
        return attrs

    def __getitem__(self, index):
        return self.file[index]

    def __str__(self):
        return f'<AsylumePy Object: {self.filename}>'

_prefix = {'y': 1e-24,  # yocto
            'z': 1e-21,  # zepto
            'a': 1e-18,  # atto
            'f': 1e-15,  # femto
            'p': 1e-12,  # pico
            'n': 1e-9,   # nano
            'u': 1e-6,   # micro
            'm': 1e-3,   # mili
            #'c': 1e-2,   # centi
            #'d': 1e-1,   # deci
            'k': 1e3,    # kilo
            'M': 1e6,    # mega
            'G': 1e9,    # giga
            'T': 1e12,   # tera
            'P': 1e15,   # peta
            'E': 1e18,   # exa
            'Z': 1e21,   # zetta
            'Y': 1e24,   # yotta
        }
