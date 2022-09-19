from warnings import warn
import subprocess
import sys

def install(package):
    subprocess.call([sys.executable, "-m", "pip", "install", package])
        
try:
    import pyUSID as usid
except ImportError:
    warn('pyUSID not found.  Will install with pip.')
    install('pyUSID')
    
try:
    import igor
except ImportError:
    warn('igor not found.  Will install with pip.')
    install('igor')
    
try:
    import pandas
except ImportError:
    warn('pandas not found.  Will install with pip.')
    install('pandas')
    
try:
    import h5py
except ImportError:
    warn('h5py not found.  Will install with pip.')
    install('h5py')

from . import base
from .images import Image
from .forcecurve import ForceCurve