from enum import Enum
from warnings import warn
import sys
import numpy as np

from pyUSID.io.dimension import Dimension, DimType, validate_dimensions
from pyUSID.io.anc_build_utils import VALUES_DTYPE, INDICES_DTYPE, build_ind_val_matrices
from pyUSID.io.hdf_utils.simple import validate_dims_against_main

from sidpy.base.string_utils import clean_string_att, validate_single_string_arg, validate_string_args
from sidpy.base.num_utils import contains_integers

if sys.version_info.major == 3:
    unicode = str

from .groups import IgorGroup
from .file import IgorFile
from .datasets import IgorDataset

def write_main_dataset(parent, main_data, main_data_name, quantity, units,
                pos_dims, spec_dims, main_dset_attrs=None, pos_inds=None,
                pos_vals=None, spec_inds=None, spec_vals=None,
                aux_spec_prefix='Spectroscopy_', aux_pos_prefix='Position_', verbose=False,
                slow_to_fast=False, **kwargs):
                
    def __check_anc_before_creation(aux_prefix, dim_type='pos'):
        aux_prefix = validate_single_string_arg(aux_prefix, 'aux_' + dim_type + '_prefix')
        if not aux_prefix.endswith('_'):
            aux_prefix += '_'
        if '-' in aux_prefix:
            warn('aux_' + dim_type + ' should not contain the "-" character. Reformatted name from:{} to '
                                     '{}'.format(aux_prefix, aux_prefix.replace('-', '_')))
        aux_prefix = aux_prefix.replace('-', '_')
        for dset_name in [aux_prefix + 'Indices', aux_prefix + 'Values']:
            if dset_name in parent.keys():
                # TODO: What if the contained data was correct?
                raise KeyError('Dataset named: ' + dset_name + ' already exists in group: '
                                                               '{}. Consider passing these datasets using kwargs (if they are correct) instead of providing the pos_dims and spec_dims arguments'.format(parent.name))
        return aux_prefix

    if not isinstance(parent, IgorGroup):
        raise TypeError('Parent data type should be IgorGroup Object.')
    if verbose:
        print('object is ready')

    quantity, units, main_data_name = validate_string_args([quantity, units, main_data_name],
                                                    ['quantity', 'units', 'main_data_name'])
    quantity = quantity.strip()
    units = units.strip()
    main_data_name = main_data_name.strip()

    if '-' in main_data_name:
        main_data_name = main_data_name.replace('-', '_')

    if isinstance(main_data, (list, tuple)):
        if not contains_integers(main_data, min_val=1):
            raise ValueError('main_data if specified as a shape should be a list / tuple of integers >= 1')
        if len(main_data) != 2:
            raise ValueError('main_data if specified as a shape should contain 2 numbers')
        #if 'dtype' not in kwargs:
        #    raise ValueError('dtype must be included as a kwarg when creating an empty dataset')
        #_ = validate_dtype(kwargs.get('dtype'))
        main_shape = main_data
        if verbose:
            print('Selected empty dataset creation. OK so far')
    elif isinstance(main_data, np.ndarray):
        if main_data.ndim != 2:
            raise ValueError('main_data should be a 2D array')
        main_shape = main_data.shape
        if verbose:
            print('Provided numpy or Dask array for main_data OK so far')
    else:
        raise TypeError('main_data should either be a numpy array or a tuple / list with the shape of the data')

    if pos_inds is not None and pos_vals is not None:
        # The provided datasets override fresh building instructions.
 
        #validate_anc_h5_dsets(pos_inds, pos_vals, main_shape, is_spectroscopic=False)
        if verbose:
            print('The shapes of the provided h5 position indices and values are OK')
        #pos_inds, pos_vals = __ensure_anc_in_correct_file(h5_pos_inds, h5_pos_vals, 'Position')
    else:
        aux_pos_prefix = __check_anc_before_creation(aux_pos_prefix, dim_type='pos')
        pos_dims = validate_dimensions(pos_dims, dim_type='Position')
        validate_dims_against_main(main_shape, pos_dims, is_spectroscopic=False)
        if verbose:
            print('Passed all pre-tests for creating position datasets')

        pos_inds, pos_vals = write_ind_val_dsets(parent, pos_dims, is_spectral=False, verbose=verbose,
                                                       slow_to_fast=slow_to_fast, base_name=aux_pos_prefix)

        if verbose:
            print('Created position datasets!')

    if spec_inds is not None and spec_vals is not None:
        # The provided datasets override fresh building instructions.
        #validate_anc_h5_dsets(spec_inds, spec_vals, main_shape, is_spectroscopic=True)
        if verbose:
            print('The shapes of the provided h5 position indices and values '
                  'are OK')
        #h5_spec_inds, h5_spec_vals = __ensure_anc_in_correct_file(spec_inds, spec_vals,
        #                                 'Spectroscopic')
    else:
        aux_spec_prefix = __check_anc_before_creation(aux_spec_prefix, dim_type='spec')
        spec_dims = validate_dimensions(spec_dims, dim_type='Spectroscopic')
        validate_dims_against_main(main_shape, spec_dims, is_spectroscopic=True)
        if verbose:
            print('Passed all pre-tests for creating spectroscopic datasets')
        spec_inds, spec_vals = write_ind_val_dsets(parent, spec_dims, is_spectral=True, verbose=verbose,
                                                         slow_to_fast=slow_to_fast, base_name=aux_spec_prefix)
        if verbose:
            print('Created Spectroscopic datasets')

    if isinstance(main_data, np.ndarray):
        # simple dataset
        main = parent.create_dataset(main_data_name, main_data, **kwargs)
    
        if verbose:
            print('Created main dataset with provided data')
    write_simple_attrs(main, {'quantity': quantity, 'units': units} )
    if verbose:
        print('Wrote quantity and units attributes to main dataset')

    if isinstance(main_dset_attrs, dict):
        write_simple_attrs(main, main_dset_attrs)
        if verbose:
            print('Wrote provided attributes to main dataset')


def write_ind_val_dsets(parent_group, dimensions, is_spectral=True, verbose=False, base_name=None, slow_to_fast=False):
    
    if isinstance(dimensions, Dimension):
        dimensions = [dimensions]
    if not isinstance(dimensions, (list, np.ndarray, tuple)):
        raise TypeError('dimensions should be array-like.')
    if not np.all([isinstance(x, Dimension) for x in dimensions]):
        raise TypeError('dimensions should be a sequence of Dimension objects.')

    if not isinstance(parent_group, (IgorGroup, IgorFile, IgorDataset)):
        raise TypeError('parent should be IgorGroup or IgorFile.')

    if base_name is not None:
        base_name = validate_single_string_arg(base_name, 'base_name')
        if not base_name.endswith('_'):
            base_name += '_'
    else:
        base_name = 'Position_'
        if is_spectral:
            base_name = 'Spectroscopic_'

    for sub_name in ['Indices', 'Values']:
        if base_name + sub_name in list(parent_group.keys()):
            raise KeyError('Dataset: {} already exists in provied group: {}'. format(base_name+sub_name, parent_group.name))

    modes = [dim.mode for dim in dimensions]
    sing_mode = np.unique(modes)

    if sing_mode.size > 1:
        raise NotImplementedError('Cannot yet work on combinations of modes for dimensions. Consider doing manually.')
    
    sing_mode = sing_mode[0]

    if sing_mode == DimType.DEFAULT:
        if slow_to_fast:
            # Ensure that the dimensions are arranged from fast to slow!
            dimensions = dimensions[::-1]
        indices, values = build_ind_val_matrices([dim.values for dim in dimensions],
                                                is_spectral=is_spectral)
        rev_func = np.flipud if is_spectral else np.fliplr
        dimensions = dimensions[::-1]
        indices, values = rev_func(indices), rev_func(values)

    elif sing_mode == DimType.INCOMPLETE:
        lengths = np.unique([len(dim.values) for dim in dimensions])
        if len(lengths) > 1:
            raise ValueError('Values for dimensions not of same lengths.')
        single_dim = np.arange(lengths[0], dtype=INDICES_DTYPE)
        indices = np.tile(single_dim, (2, 1)).T
        values = np.dstack(tuple([dim.values for dim in dimensions])).squeeze()

        if is_spectral:
            indices, values = indices.T, values.T
        
    else:
        raise NotImplementedError('Cannot yet work on Dependent dimensions.')

    if verbose:
        print("Indices: ")
        print(indices)
        print("Values: ")
        print(values)

    # Create object
    indices_obj = IgorDataset(base_name + 'Indices', data=INDICES_DTYPE(indices), dtype=INDICES_DTYPE)
    values_obj = IgorDataset(base_name + 'Values', data=VALUES_DTYPE(values), dtype=VALUES_DTYPE)

    # Push object
    parent_group.datasets[base_name+'Indices'] = indices_obj
    parent_group.datasets[base_name+'Values'] = values_obj

    for dset in [indices_obj, values_obj]:
        
        write_simple_attrs(dset, {
            'units' : [x.units for x in dimensions],
            'labels' : [x.name for x in dimensions],
            'type' : [dim.mode.value for dim in dimensions]
        })
    
    return indices_obj, values_obj

def write_simple_attrs(IgorObj, attrs, force_to_str=True, verbose=False):
    # Works on defined Igor objects, modified from sidpy function
    if not isinstance(attrs, dict):
            raise TypeError('attrs should be a dictionary but is instead of type {}'.format(type(attrs)))

    if not isinstance(IgorObj, (IgorFile, IgorGroup, IgorDataset)):
        raise TypeError('dataset should be IgorDataset object.')
    for key, val in attrs.items():
        if not isinstance(key, (str, unicode)):
            if force_to_str:
                warn('Converted key: {} from type: {} to str'
                     ''.format(key, type(key)))
                key = str(key)
            else:
                warn('Skipping attribute with key: {}. Expected str, got {}'
                     ''.format(key, type(key)))
                continue

        # Get rid of spaces in the key
        key = key.strip()

        if val is None:
            continue
        if isinstance(val, Enum):
            if verbose:
                print('taking the name: {} of Enum: {}'.format(val.name, val))
            val = val.name

        if isinstance(val, list):
            dictionaries = False
            for item in val:
                if isinstance(item, dict):
                    dictionaries = True
                    break
            if dictionaries:
                new_val = {}
                for key, item in enumerate(val):
                    new_val[str(key)] = item
                val = new_val

        if isinstance(val, dict):
            if isinstance(IgorObj, IgorDataset):
                raise ValueError('provided dictionary was nested, not flat. '
                                 'Flatten dictionary using sidpy.base.dict_utils.'
                                 'flatten_dict before calling sidpy.hdf.hdf_utils.'
                                 'write_simple_attrs')
            else:
                new_object = IgorObj.create_group(str(key))
                write_simple_attrs(new_object, val, force_to_str=True, verbose=False)
            
        if verbose:
            print('Writing attribute: {} with value: {}'.format(key, val))
        
        if not (isinstance(val, dict)):  # not sure how this can happen
            if verbose:
                print(key, val)
            try:
                #IgorObj.attrs[key] = clean_val
                setattr(IgorObj, key, val)
            except Exception as excp:
                if verbose:
                    if force_to_str:
                        warn('Casting attribute value: {} of type: {} to str'
                            ''.format(val, type(val)))
                        setattr(IgorObj, key, val)
                        #IgorObj.attrs[key] = str(val)
                    else:
                        raise excp('Could not write attribute value: {} of type: {}'
                                ''.format(val, type(val)))
    if verbose:
        print('Wrote all (simple) attributes to {}: {}'.format(type(IgorObj), IgorObj.name))

def assign_group_index(parent_group, base_name):
        base_name = validate_single_string_arg(base_name, 'base_name')
        if len(base_name) == 0:
            raise ValueError("base_name should not be an empty string.")

        if not base_name.endswith('_'):
            base_name += '_'
        
        temp = [key for key in parent_group.keys()]

        previous_indices = []

        for item_name in temp:
            if item_name.startswith(base_name):
                previous_indices.append(int(item_name.replace(base_name, '')))
        
        previous_indices = np.sort(previous_indices)

        if len(previous_indices) == 0:
            index = 0
        else:
            index = previous_indices[-1] + 1
        
        return base_name + '{:03d}'.format(index)

def create_indexed_group(parent_group, base_name):
    if not isinstance(parent_group, (IgorFile, IgorGroup)):
        raise TypeError('Check the object type.')

    base_name = validate_single_string_arg(base_name, 'base_name')
    group_name = assign_group_index(parent_group, base_name)

    new_group = parent_group.create_group(group_name)
    return new_group