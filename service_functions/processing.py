from service_functions.loading import load_matlab_matpsf
import pickle
import numpy as np
from dspline.struct_type import struct


def make_new_dict(filename, ref_pickle='psfsim_astig.pickle'):
    '''
    We load Matlab-based PSF and convert it to a pickle file of the same format as we had before.
    '''

    # Load reference pickle
    with open(ref_pickle, 'rb') as f:
        d = pickle.load(f)

    psf, tmp_dict = load_matlab_matpsf(filename)

    zrange = np.linspace(*tmp_dict['zrange'], num=tmp_dict['Npupil'])
    zres = int(abs(zrange[0] - zrange[1]))
    items = [psf,
             None,
             tmp_dict['lambda'],
             tmp_dict['NA'],
             tmp_dict['refcov'],
             tmp_dict['pixelsize'],
             tmp_dict['Mx'],
             zres,
             tmp_dict['Npupil'],
             None,
             zres,
             zrange,
             tmp_dict['aberrations'],
             tmp_dict['Mx']]
    return dict(zip(d.keys(), items))


def convert_mat_to_pickle(matlab_name='exp/base_astig.mat', pickle_name='exp/new_astig.pickle'):
    d = make_new_dict(matlab_name)

    with open(pickle_name, "wb") as f:
        s = struct()
        s.__dict__.update(d)
        pickle.dump(s, f)

    return d