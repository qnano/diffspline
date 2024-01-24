from fastpsf import Context, CSplineMethods
import glob
import numpy as np
from scipy.io import loadmat



def load_psfmodel(fname, roisize):
    ctx = Context(debugMode=False)
    psfmodel = CSplineMethods(ctx).CreatePSFFromFile(roisize, fname)
    return psfmodel


def get_basepsf_data(base_folder, nums='????', pop=None, print_paths=False, specific=None):
    '''
    psf_ind - list of indices to include on the third place of the depth (hundreds, ?x??)
    pop - index of the element to pop
    '''
    # inds = ', '.join(map(str,psf_ind))
    if specific is None:
        psf_paths = glob.glob(f'{base_folder}/astig_PSF_base_sph{nums}.mat')
        psf_paths = sorted(psf_paths, key=lambda x: x[-8:-4])  # sort by depth
        Ds = np.array([int(elem[-8:-4]) for elem in psf_paths])
    else:
        psf_paths = []
        for depth in specific:
            psf_paths += glob.glob(f'{base_folder}/astig_PSF_base_sph{int(depth):04d}.mat')
        Ds = specific

    if pop is not None:
        psf_paths.pop(pop)

    if print_paths:
        print(psf_paths)

    all_psfs = []
    for psf_path in psf_paths:
        psf, _ = load_matlab_matpsf(psf_path)
        all_psfs.append(psf)
    all_psfs = np.array(all_psfs)

    return all_psfs, Ds


def get_smoothpsf_data(base_folder, micron=None, nm_hundreds='?', specific=None, roisize_smap=34):
    if specific is None:
        num = f'{micron}{nm_hundreds}00'
        psf_paths = glob.glob(f'{base_folder}/astig_smooth_sph{num}_3Dcorr.mat')
        psf_paths = sorted(psf_paths, key=lambda x: x[-15:-11])  # sort by depth
    else:
        psf_paths = []
        for depth in specific:
            psf_paths += glob.glob(f'{base_folder}/astig_smooth_sph{depth:04d}_3Dcorr.mat')
    print(psf_paths)

    d_psfs = {}
    for smap_name in psf_paths:
        ctx_smap = Context(debugMode=False)
        smap_psfmodel = CSplineMethods(ctx_smap).CreatePSFFromFile(roisize_smap, smap_name)
        n = int(smap_name.split('_')[-2][3:])
        d_psfs[n] = smap_psfmodel.ExpectedValue(generate_params(smap_psfmodel, I=1, bg=0, roipos=(0,0)))

    all_psfs, all_depths = np.array(list(d_psfs.values())), np.array(list(d_psfs.keys()))
    print("Depths in inital stack: ", *all_depths)
    return all_psfs, all_depths



def load_matlab_matpsf(filename='exp/base_astig.mat'):
    '''
    Load the Matlab PSF file and return the PSF and the parameter dictionary
    '''
    ref = loadmat(filename)  # loadmat('exp/p.mat')

    # Convert the parameter values to a dictionary with scalar values or tuples where applicable
    vals = [elem[0] if len(elem) != 0 else elem for elem in ref['parameters'][0][0]]
    vals = [elem[0] if len(elem) == 1 else tuple(elem) if len(elem) == 2 else [] if len(elem) == 0 else elem for elem in
            vals]

    # Transpose the PSF from x,y,z to z,y,x
    psf = np.transpose(ref['PSF'], (2, 0, 1))

    # Get the final dictionary of simulation parameters
    return psf, dict(zip(ref['parameters'].dtype.names, vals))


def generate_params(psf, I=1, bg=0, roipos=(0,0)):
    """
    Generates parameters with X,Y in the center of the PSF roisize over the whole Z range.
    """
    roisize = psf.calib.coefsx.shape[1]
    params = np.repeat([[roisize / 2 + roipos[0], roisize / 2 + roipos[1], 0, I, bg]], psf.calib.n_voxels_z + 1,
                       axis=0)  # np.zeros((psf.calib.n_voxels_x, psf.calib.n_voxels_y, psf.calib.n_voxels_z, ))
    params[:, 2] = np.round(
        np.linspace(int(psf.calib.z_min * 100) * 1, int(psf.calib.z_max * 100), psf.calib.n_voxels_z + 1) / 100,
        3)  # in NANOMETERS
    return params.astype(float)
