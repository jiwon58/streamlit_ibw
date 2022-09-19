class IgorDataset(object):

    def __init__(self, base_name = None, data = None, dtype = None, shape = None, parent=None):
        if parent is not None:
            self.parent_name = parent.name
        if base_name is not None:
            self.name = base_name
        else:
            self.name = 'simple_dataset'
        self.data = data
        if shape is None:
            try:
                self.shape = data.shape
            except:
                self.shape = None
        else:
            self.shape = shape
        self.datasets = {}
        self.dtype = dtype

    def set_parent(self, parent):
        self.parent_name = parent.name

    def keys(self):
        return list(self.datasets.keys())

    def get_attr(self, index):
        return self.attrs[index]
    
    def __getitem__(self, index):
        return self.datasets[index]
    
    def __str__(self):
        return f'<Igor Dataset "{self.name}": shape {self.shape}, type: {self.dtype} >'