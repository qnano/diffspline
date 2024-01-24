from dspline.spline_psf import CSplinePSFEstimator4D
from fastpsf import Context, CSplineCalibration, CSplineMethods
from service_functions.loading import get_smoothpsf_data
import glob
import numpy as np

def calibrate_and_esimate_diffspline(base_folder, microns = [2,5], est_I = 15000.0, est_bg = 50.0, zstepsize = 10, roisize = 34, zplanes = 101, 
                          damp = 10000, iterations = 250, device='cuda'):
    from service_functions.estimation import produce_estimations
    name = f'{base_folder}/astig_PSF'
    d_estims = {}

    for micron in microns:
        all_psfs, all_depths = get_smoothpsf_data(micron, base_folder)
        n_beads = all_psfs.shape[0]
        print("Depths in inital stack: ", *all_depths)


        for depth in range(0+1,9-1):  # extra +1 because of SMAP files starting from 1000 nm depth instead of 0
            print(f"Current depth {all_depths[depth]}")

            estimator = CSplinePSFEstimator4D(stepsize_nm=zstepsize, device=device)

            print_out = ''
            print_out += estimator.init(gt_I=np.ones(all_psfs.shape[0]), gt_bg=np.zeros(all_psfs.shape[0]), gt_PSF=all_psfs, gt_depths=all_depths, positions=[1.5,0,0], n_beads=n_beads)
            print(print_out)

            calib = estimator.to_calib(depth=all_depths[depth], median_padding=False)
            calib.save_mat(name+f'_diffspline_dp{all_depths[depth]:04d}.mat')

            ctx = Context(debugMode=False)
            psfmodel = CSplineMethods(ctx).CreatePSFFromFile(roisize, name + f'_diffspline_dp{all_depths[depth]:04d}' + '.mat')

            estims_exp_, _ = produce_estimations(psfmodel, base_folder, est_I=est_I, est_bg=est_bg, 
                                                 nums=f'{micron}?00', calibration_depth_ind=depth,
                                                 plot_title=f'4D Diffspline interpolating exp SMAP data at {all_depths[depth]}',
                                                 damp=damp, iterations=iterations) #, ylim=(-50,50))
            d_estims[all_depths[depth]] = estims_exp_.copy()



def calibrate_diffspline_at_depth(depth, psfs, depths, base_folder, positions=[-0.5,-0.5,0], median_padding=False, zstepsize=10, name=None, device='cuda'):
    '''
    Calibrate diffspline given a specific depth and smoothpsfs and save it in .mat file
    Function doesn't return anything
    '''
    estimator = CSplinePSFEstimator4D(stepsize_nm=zstepsize, device=device)

    print_out = ''
    print_out += estimator.init(gt_I=np.ones(psfs.shape[0]), gt_bg=np.zeros(psfs.shape[0]), gt_PSF=psfs, gt_depths=depths, positions=[0,0,0], n_beads=psfs.shape[0])
    

    calib_exp_ = estimator.to_calib(depth=depth, median_padding=median_padding, pos=positions)
    if name is None:
        calib_exp_.save_mat(f'{base_folder}/astig_PSF_diffspline_dp{int(depth):04d}.mat')
    else:
        calib_exp_.save_mat(name)


def process_with_SMAP_pipeline(calibration_depths, base_folder, n_realizations=10, return_psfs=False):
    # send the PSFs through the SMAP pipeline
    import matlab.engine
    eng = matlab.engine.start_matlab()
    
    psf_paths = []
    for depth in calibration_depths:
        psf_paths += glob.glob(f'{base_folder}/astig_PSF_x??_y??_sph{int(depth):04d}.tif')
    psf_paths = sorted(psf_paths, key=lambda x: x[-8:-4])  # sort by depth
    each_depth = [int(elem[-8:-4]) for elem in psf_paths]
    print(psf_paths, each_depth)

    '''
    SMAP parameters:

    p.filtersize = 2;
    p.mindistance = 10; 
    p.dz = 10; 
    p.zcorr ='cross-correlation';
    p.zcorrframes = 50; 
    p.ROIxy = 34; 
    p.smoothz = 1;
    '''

    output_filename = '_'.join(psf_paths[0].split('_')[:-4] + ['smooth'] + psf_paths[0].split('_')[-1:]).split('.')[0][:-4]
    print(output_filename, each_depth)

    if return_psfs:
        PSFsmooth, PSF, zstack, shiftedzstack, shifts = eng.test(psf_paths, output_filename, each_depth, n_realizations,
                                                                 nargout=5)
        PSFsmooth_np = np.array(PSFsmooth).transpose(2,0,1)
        PSF_np = np.array(PSF).transpose(2,0,1)
        zstack = np.array(zstack).transpose(3,2,0,1)
        shiftedzstack = np.array(shiftedzstack).transpose(3,2,0,1)
        return PSFsmooth_np, PSF_np, zstack, shiftedzstack, shifts
    else:
        eng.test(psf_paths, output_filename, each_depth, n_realizations, nargout=5)
    eng.quit()