from .groups import IgorGroup

class IgorFile(IgorGroup):
    
    def __init__(self, name = '/'):
        self.name = name
        self.groups = {}
        self.datasets = {}
        self.type = 'IgorFile'

    def keys(self):
        return self.groups.keys()

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
            return self.groups[index]

    def __str__(self):
        return '<Igor File "{}": has {} groups, type: {} >'.format(self.name, len(self.groups.keys()), self.type)
        