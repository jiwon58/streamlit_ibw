from .datasets import IgorDataset

class IgorGroup(object):

    def __init__(self, base_name=None, parent=None):
        self.parent_name = parent.name
        if base_name is not None:
            self.name = base_name
        else:
            self.name = '/'
        self.datasets = {}
        self.groups ={}
        self.type = 'IgorGroup'
        self.shape = None
        
    def create_group(self, base_name):
        group = IgorGroup(base_name = base_name, parent=self)
        self.groups[group.name] = group
        self.names = list(self.groups.keys())
        return self.groups[group.name]

    def create_dataset(self, base_name, data, shape=None, dtype=None, **kwargs):
        self.datasets[base_name] = IgorDataset(base_name, data, shape=shape, dtype=dtype, parent=self, **kwargs)
        return self.datasets[base_name]

    def keys(self):
        return [key.split('/')[-1] for key in self.groups.keys()]

    def __getitem__(self, index):
        if index == '/':
            return self.groups
        if index.endswith('/'):
            index = index[:-1]
        if index.startswith('/'):
            index = index[1:]
        if len(index.split('/')) > 1:
            grp_name = index.split('/')[0]
            return self.groups[grp_name].__getitem__(index[len(grp_name):])
        else:
            try:
                return self.groups[index]
            except:
                return self.datasets[index]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index > len(self.names)-1:
            raise StopIteration
        n = self.names[self.index]
        self.index += 1
        return n

    def __str__(self):
        return '<Igor Dataset "{}": shape {}, type: {} >'.format(self.name, self.shape, self.type)