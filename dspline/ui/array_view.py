# -*- coding: utf-8 -*-
"""
Created on Mon Feb 21 18:33:21 2022

@author: jelmer
"""


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True   # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False      # Probably standard Python interpreter


def array_view(img, **kwargs):

    if isnotebook():
        raise RuntimeError('No array viewer implemented for jupyter notebooks')
    
    import dspline.ui.array_viewer_pyqt as pyqt_ui
    return pyqt_ui.array_view(img, **kwargs)
        
