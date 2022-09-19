
class IgorAttr(object):

    def __init__(self, base_name = None, parent=None):
        self.parent_name = parent.name
        if base_name is not None:
            self.name = base_name
        else:
            self.name = 'simple_attributes'
        self.attrs ={}

    def keys(self):
        return self.attrs.keys()

    def __getitem__(self, index):
        if len(self.attrs.keys()) > 1:
            return self.attrs[index]
        else:
            return None