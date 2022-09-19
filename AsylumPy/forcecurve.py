import pyUSID as usid
from .ibw import IBW

class ForceCurve(IBW):
    def __init__(self, filename, h5):
        super().__init__(filename, h5)
        self.type = 'IgorIBW_ForceCurve'
        if self.type != self.get_type(self.file):
            raise ValueError('This object is not a igor force curve.')
        #self.NumPts = self.get_dim(0)[1] # Number of Points
        self.Delta = 1/self.get_attr(self.h5, 'NumPtsPerSec')
        self.TriggerTime = self.get_attr(self.h5, 'TriggerTime')
        self.Arg2 = self.get_attr(self.h5, 'ARDoIVArg2')
        self.Arg3 = self.get_attr(self.h5, 'ARDoIVArg3') # On/Off cycle time
        self.CycleNumPts = int(self.Arg3/self.Delta) # Number of points in one cycle
        self.Cycles = self.get_attr(self.h5, 'ARDoIVCycles')
        self.IVFrequency = self.get_attr(self.h5, 'ARDoIVFrequency') # Cycles/Frequency == Dwell time
        self.TriggerPoint = int(self.TriggerTime / self.Delta)
        self.EndDwell = int(self.TriggerPoint + self.Cycles/self.IVFrequency/self.Delta)
        
    def get_dim(self, channel):
        channel = self.channel2str(channel)
        h5 = self.h5[self.grp_name][channel+'/Raw_Data']
        #dim, units = self._get_dim(channel)
        spec_inds = usid.hdf_utils.get_auxiliary_datasets(h5, ['Spectroscopic_Indices'])[0]
        spec_dim_sizes = usid.hdf_utils.get_dimensionality(spec_inds)
        spec_dim_names = usid.hdf_utils.get_attr(spec_inds, 'labels')
        spec_dim = dict()
        for name, length in zip(spec_dim_names, spec_dim_sizes):
            spec_dim[name] = length
        units = self.get_attr(h5, 'units')
        return spec_dim['Z'], units
    
    def get_data(self, channel):
        channel = self.channel2str(channel)
        data = usid.USIDataset(self.h5[self.grp_name][channel+'/Raw_Data'])[0]
        return data