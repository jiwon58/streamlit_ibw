from sidpy.base.num_utils import contains_integers

from .groups import IgorGroup
from .file import IgorFile
from .datasets import IgorDataset

def validate_anc_h5_dsets(inds, vals, main_shape, is_spectroscopic=True):
    """
    Checks ancillary HDF5 datasets against shape of a main dataset.
    Errors in parameters will result in Exceptions

    Parameters
    ----------
    inds : IgorDataset
        HDF5 dataset corresponding to the ancillary Indices dataset
    vals : IgorDataset
        HDF5 dataset corresponding to the ancillary Values dataset
    main_shape : array-like
        Shape of the main dataset expressed as a tuple or similar
    is_spectroscopic : bool, Optional. Default = True
        set to True if ``dims`` correspond to Spectroscopic Dimensions.
        False otherwise.
    """
    if not isinstance(inds, IgorDataset):
        raise TypeError('inds must be a IgorDataset object')
    if not isinstance(vals, IgorDataset):
        raise TypeError('vals must be a IgorDataset object')
    if inds.shape != vals.shape:
        raise ValueError('h5_inds: {} and vals: {} should be of the same '
                         'shape'.format(inds.shape, vals.shape))
    if isinstance(main_shape, (list, tuple)):
        if not contains_integers(main_shape, min_val=1) or \
                len(main_shape) != 2:
            raise ValueError("'main_shape' must be a valid IgorDataset shape")
    else:
        raise TypeError('main_shape should be of the following types:'
                        'h5py.Dataset, tuple, or list. {} provided'
                        ''.format(type(main_shape)))

    if inds.shape[is_spectroscopic] != main_shape[is_spectroscopic]:
        raise ValueError('index {} in shape of h5_inds: {} and main_data: {} '
                         'should be equal'.format(int(is_spectroscopic),
                                                  inds.shape, main_shape))


def walk_dict(obj, depth=1):
    groups = obj.groups
    dsets = obj.datasets
    if isinstance(obj, IgorFile):
        print("IgorFile: {}".format(obj.name))
    if isinstance(obj, IgorGroup):
        for k, v in sorted({**groups, **dsets}.items(), key=lambda x: x[0]):
            if isinstance(v, IgorGroup):
                print(" " * depth + "<{}>: {}".format(v.type, k))
                print(" " * depth + "-" * len(k) * 2)
                walk_dict(v, depth+1)
            elif isinstance(v, IgorDataset):
                print(" " * depth + "├ Dataset: {}".format(k))
            elif k is not None:
                print(" " * depth + "{}".format(k))
            