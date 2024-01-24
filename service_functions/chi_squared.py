import numpy as np
import matplotlib.pyplot as plt
from service_functions.loading import generate_params, get_smoothpsf_data, get_basepsf_data
from dspline.spline_psf import CSplinePSFEstimator4D
from fastpsf import Context, CSplineMethods



def get_chisquareds(psfmodel, observed, I, bg, roipos=(0,0), roichange=(None, None), zrange=(None,None)):
    expected = psfmodel.ExpectedValue(generate_params(psfmodel, I=I, bg=bg, roipos=roipos))[None, zrange[0]:zrange[1], roichange[0]:roichange[1], roichange[0]:roichange[1]]
    return chi_squared(observed=observed, expected=expected)


def get_chisq_over_delta(delta_depth1, subdelta, base_folder,
                         num_beads=10, I=10000, bg=50, zstepsize=10, device='cuda'):
    ''' Under construction
    Delta_depth1 & 2 - the interval (d1,d2) to interpolate over, delta = d2 - d1
    Subdelta - subinterval size
    '''
    name = f'{base_folder}/astig_PSF'
    #delta = delta_depth2 - delta_depth1
    specific = [delta_depth1 - subdelta, delta_depth1, delta_depth1+subdelta, delta_depth1 + 2*subdelta]
    # # Load data with the chosen delta
    # Load 4 data points to interpolate over (training/calibration data)
    calibration_psfs, calibration_depths = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?', specific=specific)
    # Load all subinterval data (test data)
    #subinterval_depths = list(np.arange(start=delta_depth1+subdelta, stop=delta_depth2, step=subdelta))
    #test_psfs, _ = get_basepsf_data(base_folder, nums=None, specific=subinterval_depths)

    ## Calibrate the diffspline on data
    estimator = CSplinePSFEstimator4D(stepsize_nm=zstepsize, device=device)

    estimator.init(gt_I=np.ones(calibration_psfs.shape[0]), gt_bg=np.zeros(calibration_psfs.shape[0]),
                        gt_PSF=calibration_psfs, gt_depths=calibration_depths, positions=[-1, -1, 0],
                        n_beads=calibration_psfs.shape[0])

    # Make test data "observed" PSFs
    observed_beads = np.random.poisson(I * calibration_psfs[:, None, ...].repeat(num_beads, axis=1) + bg)  # size = (subinterval_size, num_beads, *PSF.shape)

    chisq = {}
    # For a bunch of subinterval positions
    for i, observed_realizations in enumerate(observed_beads):
        ## Evaluate the diffspline at the position
        print(f'Calibrating on depth={calibration_depths[i]} nm')
        calib = estimator.to_calib(depth=calibration_depths[i], median_padding=False)
        calib.save_mat(name + f'_diffspline_dp{calibration_depths[i]:04d}.mat')

        ctx_dfspl = Context(debugMode=False)
        dfspl_psfmodel = CSplineMethods(ctx_dfspl).CreatePSFFromFile(roisize=calibration_psfs.shape[-1],
                                                                     filename=name + f'_diffspline_dp{calibration_depths[i]:04d}.mat')
        ## Find the chi-squared
        chisq[calibration_depths[i]] = get_chisquareds(dfspl_psfmodel, observed_beads[i], I, bg)
        #expected_beads = dfspl_psfmodel.ExpectedValue(generate_params(dfspl_psfmodel, I=I, bg=bg))
        #chisq[subinterval_depths[i]] = chi_squared(observed=observed_beads[i], expected=expected_beads[None, ...])
    return chisq


def get_chisq_over_delta_old(delta_depth1, delta_depth2, subdelta, base_folder,
                         num_beads=10, I=10000, bg=50, zstepsize=10, device='cuda'):
    '''
    Delta_depth1 & 2 - the interval (d1,d2) to interpolate over, delta = d2 - d1
    Subdelta - subinterval size
    '''
    name = f'{base_folder}/astig_PSF'
    delta = delta_depth2 - delta_depth1
    specific = [delta_depth1 - delta, delta_depth1, delta_depth2, delta_depth2 + delta]
    # # Load data with the chosen delta
    # Load 4 data points to interpolate over (training/calibration data)
    calibration_psfs, calibration_depths = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?', specific=specific)

    # Load all subinterval data (test data)
    subinterval_depths = list(np.arange(start=delta_depth1+subdelta, stop=delta_depth2, step=subdelta))
    test_psfs, _ = get_basepsf_data(base_folder, nums=None, specific=subinterval_depths)

    ## Calibrate the diffspline on data
    estimator = CSplinePSFEstimator4D(stepsize_nm=zstepsize, device=device)

    estimator.init(gt_I=np.ones(calibration_psfs.shape[0]), gt_bg=np.zeros(calibration_psfs.shape[0]),
                        gt_PSF=calibration_psfs, gt_depths=calibration_depths, positions=[-1, -1, 0],
                        n_beads=calibration_psfs.shape[0])

    # Make test data "observed" PSFs
    observed_beads = np.random.poisson(I * test_psfs[:, None, ...].repeat(num_beads, axis=1) + bg)  # size = (subinterval_size, num_beads, *PSF.shape)

    chisq = {}
    # For a bunch of subinterval positions
    for i, observed_realizations in enumerate(observed_beads):
        ## Evaluate the diffspline at the position
        print(f'Calibrating on depth={subinterval_depths[i]} nm')
        calib = estimator.to_calib(depth=subinterval_depths[i], median_padding=False)
        calib.save_mat(name + f'_diffspline_dp{subinterval_depths[i]:04d}.mat')

        ctx_dfspl = Context(debugMode=False)
        dfspl_psfmodel = CSplineMethods(ctx_dfspl).CreatePSFFromFile(roisize=calibration_psfs.shape[-1],
                                                                     filename=name + f'_diffspline_dp{subinterval_depths[i]:04d}.mat')
        expected_beads = dfspl_psfmodel.ExpectedValue(generate_params(dfspl_psfmodel, I=I, bg=bg))

        ## Find the chi-squared
        chisq[subinterval_depths[i]] = chi_squared(observed=observed_beads[i], expected=expected_beads[None, ...])
    return chisq



def get_chisq_over_intensity(diffspline_psfmodel, smap_psfmodel, base_psf,
                             bg=1, num_beads=9, zrange=(None, None),
                             int_start=1000, int_stop=20000, int_nsteps=20, roiposs=[(0,0), (0,0)],
                             roichange=(None, None), plot=False, depth=None, labels=['diffspline', 'cspline']):
    """
    Takes Diffspline PSFmodel, SMAP PSFmodel, base PSF.

    Returns
    dictionaries of chi-squared means and stds across beads for diffspline, SMAP, and expected
    mus
    """
    # for every GT I
    chisq_res_diff = dict()
    chisq_res_smap = dict()
    chisq_res_expected_diff = dict()
    chisq_res_expected_smap = dict()

    smap_mus = dict()
    diff_mus = dict()

    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):  # np.logspace(2,4,20): #
        # Generate observed images (noised GT)
        zstack = np.random.poisson(I * base_psf[None, zrange[0]:zrange[1]].repeat(num_beads, axis=0) + bg)[..., roichange[0]:roichange[1], roichange[0]:roichange[1]]
        expected_mu = np.prod(zstack[0].shape)

        # calculate chi-squared for each bead
        chisq_diff = get_chisquareds(diffspline_psfmodel, zstack, I, bg, roipos=roiposs[0], roichange=roichange, zrange=zrange)
        #diff_mu = diffspline_psfmodel.ExpectedValue(generate_params(diffspline_psfmodel, I=I, bg=bg))[:, roichange[0]:roichange[1], roichange[0]:roichange[1]]
        #chisq_diff = chi_squared(observed=zstack, expected=diff_mu[None, zrange[0]:zrange[1]])  # (n_beads, zsize, roisize, roisize)

        # keep mean and std along the beads
        relative_chisq_diff = chisq_diff.sum((-3, -2, -1)) / expected_mu
        chisq_res_diff[I] = [relative_chisq_diff.mean(0).item(), relative_chisq_diff.std(0).item()]
        ###chisq_res_expected_diff[I] = [expected_mu, np.sqrt(2 * expected_mu + (1 / diff_mu).sum().item())]

        # chisq for SMAP
        chisq_smap = get_chisquareds(smap_psfmodel, zstack, I, bg, roipos=roiposs[1], roichange=roichange, zrange=zrange)
        #smap_mu = smap_psfmodel.ExpectedValue(generate_params(smap_psfmodel, I=I, bg=bg))[:, roichange[0]:roichange[1], roichange[0]:roichange[1]]
        #chisq_smap = chi_squared(observed=zstack, expected=smap_mu[None, zrange[0]:zrange[1]])  # (n_beads, zsize, roisize, roisize)
        relative_chisq_smap = chisq_smap.sum((-3, -2, -1)) / expected_mu
        chisq_res_smap[I] = [relative_chisq_smap.mean(0).item(), relative_chisq_smap.std(0).item()]
        ###chisq_res_expected_smap[I] = [expected_mu, np.sqrt(2 * expected_mu + (1 / smap_mu).sum().item())]

        #diff_mus[I] = diff_mu.cpu().numpy()
        #smap_mus[I] = smap_mu.cpu().numpy()
    if plot:
        plot_chisquareds([chisq_res_diff, chisq_res_smap], labels=labels, depth=depth)
    return chisq_res_diff, chisq_res_smap #, chisq_res_expected_diff, chisq_res_expected_smap, diff_mus, smap_mus


def plot_chisquareds(chisqs:list, labels, depth=''):
    for i,chisq in enumerate(chisqs):
        plt.errorbar(chisq.keys(), np.array(list(chisq.values()))[:,0], yerr=np.array(list(chisq.values()))[:,1], label=labels[i])
    
    plt.title(f'$\chi^2$ fit on depth = {depth} nm')
    plt.xlabel('Intensity, photons')
    plt.ylabel('Relative $\chi^2$')
    plt.legend()
    plt.show()


def plot_chisq_vs_intensities(chisq_diff, chisq_smap, chisq_expected_diff, chisq_expected_smap, observed, bg,
                              num_beads):
    diff_vals = np.array(list(chisq_diff.values()))
    plt.errorbar(chisq_diff.keys(), diff_vals[:, 0], yerr=diff_vals[:, 1], label="diffspline")

    smap_vals = np.array(list(chisq_smap.values()))
    plt.errorbar(chisq_smap.keys(), smap_vals[:, 0], yerr=smap_vals[:, 1], label="SMAP")

    if chisq_expected_diff is not None or chisq_expected_smap is not None:
        expdiff_vals = np.array(list(chisq_expected_diff.values()))
        expsmap_vals = np.array(list(chisq_expected_smap.values()))
        plt.errorbar(chisq_expected_diff.keys(), expdiff_vals[:, 0], yerr=expdiff_vals[:, 1],
                     label='expected diffspline $\chi^2$')
        plt.errorbar(chisq_expected_smap.keys(), expsmap_vals[:, 0], yerr=expsmap_vals[:, 1],
                     label='expected SMAP $\chi^2$')

    obs = {"gt": "GT", "data_new": "noisy"}[observed]
    plt.title(f"$\chi^2$ test for {obs} zstack with I,bg GT on {num_beads} beads (bg={bg})")
    plt.ylabel("$\chi^2$")
    plt.xlabel("Intensity, photons")
    plt.legend()
    plt.show()


def chi_squared(observed, expected):
    return (observed - expected) ** 2 / expected