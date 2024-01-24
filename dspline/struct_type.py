# -*- coding: utf-8 -*-
"""
Created on Thu May 26 21:03:08 2022

@author: jelmer
"""
from collections.abc import Mapping

class struct(Mapping):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        
    def __getitem__(self, key):
        return self.__dict__[key]
    
    def __setitem__(self, key, val):
        self.__dict__[key] = val
        
    def __repr__(self):
        return str(self.__dict__)

    def __len__(self):
        return len(self.__dict__)
    
    def __contains__(self, key):
        return key in self.__dict__
    
    def __iter__(self):
        return self.__dict__.__iter__()
        
if __name__ == '__main__':
    b = struct(x=1)

    print(b)    