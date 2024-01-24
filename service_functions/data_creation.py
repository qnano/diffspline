import numpy as np
import matlab.engine
import glob
import tifffile

from service_functions.loading import load_matlab_matpsf



def calculate_spherical_aberrations(d, na=1.3, n1=1.33, n2=1.52, wavelength=680):
    '''
    na - numerical aperture
    n1 - sample index (corresponds to refmed parameter in p.mat)
    n2 - oil index (coresponds to refcov parameter in p.mat)
    d - depth in nm
    '''
    B = lambda r, n: (1 - (n - 1) * np.tan(r / 2) ** 4 / (n + 3)) * (np.tan(r / 2) ** (n - 1)) / (
                2 * (n - 1) * np.sqrt(n + 1))

    alpha1 = np.arctan(n1 / n2)
    alpha1n1 = n2 * np.sin(alpha1)
    alpha2 = np.arcsin(alpha1n1 / na)

    A = B(alpha1, 4) - B(alpha2, 4)
    nomial_d = np.tan(alpha2) * d / np.tan(alpha1)

    spherial_nm = nomial_d * n2 * np.sin(alpha1) * A  # spherial aberration in nm
    spherial_ml = spherial_nm * 1000 / wavelength
    return (spherial_ml, 0)


def calculate_spherical_aberrations_improved(d, NA=1.3, n1=1.33, n2=1.52, wavelength=680):
    '''
    Aberration correction as per https://doi.org/10.1111/j.1365-2818.1998.99999.x
    '''
    alpha = np.arcsin(NA / n1)
    beta = np.arcsin(NA / n2)
    # aberration(12,3), aberration(26,3)
    return ((B(4, alpha) - B(4, beta)) * NA * d * 1000 / wavelength,
            (B(6, alpha) - B(6, beta)) * NA * d * 1000 / wavelength)

def B(n, r):
    tr = np.tan(r / 2)
    v = (1 - (n - 1) / (n + 3)) * tr**4 * (tr**(n - 1) / (2 * (n - 1) * np.sqrt(n + 1)))
    return v


def generate_base_PSF(d, base_folder, z_start=-500, z_finish=500, z_steps=101, ROIsize=34, spherical_type='improved', extra_folder=''):
    eng = matlab.engine.start_matlab()
    if spherical_type == 'improved':
        spherical1, spherical2 = calculate_spherical_aberrations_improved(d=d)
    else:
        spherical1, spherical2 = calculate_spherical_aberrations(d=d)
    fname = f'{base_folder}/{extra_folder}astig_PSF_base_sph{int(d):04d}.mat'
    eng.generatePSF(0, 0, float(1), float(0), z_start, z_finish, z_steps, ROIsize, float(spherical1), float(spherical2), fname, nargout=0)
    eng.quit()
    

def generate_shifted_PSFs(depths, base_folder, Irange=(5000.0, 15000.0), bg=20.0, pixelsize=100, n_realizations=10,
                          z_start=-500, z_finish=500, z_steps=101, ROIsize=56, spherical_type='improved', extra_folder=''):
    eng = matlab.engine.start_matlab()
    xyz_shifts = np.hstack([np.random.randint(0, pixelsize-1, (n_realizations, 2))])
    Is = np.arange(Irange[0],Irange[1],1000.0)
    psf_paths = []
    for d in depths:
        for j in range(n_realizations):
            if spherical_type == 'improved':
                spherical1, spherical2 = calculate_spherical_aberrations_improved(d=d)
            else:
                spherical1, spherical2 = calculate_spherical_aberrations(d=d)
            xpos, ypos = xyz_shifts[j,:]
            fname = f'{base_folder}/{extra_folder}astig_PSF_x{xpos:02d}_y{ypos:02d}_sph{int(d):04d}.mat'
            psf_paths.append(fname)
            eng.generatePSF(int(xpos), int(ypos), float(np.random.choice(Is)), float(bg), z_start, z_finish, z_steps, ROIsize, float(spherical1), float(spherical2), fname, nargout=0)
    eng.quit()
    
    for psf_path in psf_paths:
        psf, _ = load_matlab_matpsf(psf_path)
        noised_psf = np.random.poisson(psf)
        tifffile.imwrite(psf_path.split('.')[0]+'.tif', data=noised_psf, dtype='uint32')
    

