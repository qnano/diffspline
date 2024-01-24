import torch
from dspline.spline_psf import CSplinePSFEstimator
from view_utils.utils import chi_squared
import matplotlib.pyplot as plt
import numpy as np
from dspline.training import create_zstack


def generate_params(psf, I=1, bg=0):
    """
    Generates parameters with X,Y in the center of the PSF roisize over the whole Z range.
    """
    roisize = psf.calib.coefsx.shape[1]
    params = np.repeat([[roisize/2, roisize/2, 0, I, bg]], psf.calib.n_voxels_z+1, axis=0)  # np.zeros((psf.calib.n_voxels_x, psf.calib.n_voxels_y, psf.calib.n_voxels_z, ))
    params[:, [0,1]] = roisize / 2
    params[:, 2] = np.round(np.linspace(int(psf.calib.z_min*100)*1, int(psf.calib.z_max*100), psf.calib.n_voxels_z+1)/100, 3) # in NANOMETERS
    return params.astype(float)


def get_chisq_over_intensity(estimator, smap_psfmodel, d, observed='data', bg=1, num_beads=9, zrange=(10, -9),
                             int_start=1000, int_stop=20000, int_nsteps=20, device='cuda:0'):
    """
    Takes diffspline model, SMAP PSFmodel, dictionary with GT

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

    sum_inv_mu_diff = []
    sum_inv_mu_smap = []

    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):  # np.logspace(2,4,20): #
        if observed == 'data':
            # extract ROIs from the zstack
            true_zstack = create_zstack(d.psf, d.zsize, num_beads, roisize=d.roisize, psf_zrange=d.zrange,
                                        zrange=zrange, intensity=[I] * num_beads, background=bg, pad=1)
            zstack = torch.Tensor(np.random.poisson(true_zstack)).to(device)
            zstack_estimator = CSplinePSFEstimator(stepsize_nm=d.zres, device=device)
            zstack_estimator.add_zstack(zstack,
                                        threshold=1.2, detection_sigma=3, roisize=d.roisize)

            # smap_mu shape is (80,30,30) instead of (81,30,30) which the GT and diffspline are
            zstack = torch.stack(zstack_estimator.zstacks)[:, :-1]  # zstack shape = (n_beads, zplane, roisize, roisize)
        elif observed == 'data_new':
            zstack = torch.Tensor(
                np.random.poisson(I * d.psf[None, zrange[0]:zrange[1]].repeat(num_beads, axis=0) + bg)).to(device)
        elif observed == 'gt':
            zstack = torch.Tensor(I * d.psf[None, zrange[0]:zrange[1] - 1].repeat(num_beads, axis=0) + bg).to(device)
            chisq_res_expected_diff = None
            chisq_res_expected_smap = None

        expected_mu = np.prod(zstack[0].shape)
        estimator.positions[0, 0] = -0.5
        estimator.positions[0, 1] = -0.5
        estimator.positions[0, 2] = 0
        diff_psf = estimator.eval(d.roisize, d.zsize, I=1, bg=0)[0].detach()  # estimator.psf[:-1]

        # calculate chi-squared for each bead
        diff_mu = I * diff_psf + bg
        chisq_diff = chi_squared(observed=zstack, expected=diff_mu[None, :])  # (n_beads, zsize, roisize, roisize)

        # keep mean and std along the beads
        chisq_res_diff[I] = [chisq_diff.sum((-3, -2, -1)).mean(0).item(), chisq_diff.sum((-3, -2, -1)).std(0).item()]
        if observed != 'gt':
            chisq_res_expected_diff[I] = [expected_mu, np.sqrt(2 * expected_mu + (1 / diff_mu).sum().item())]

        # chisq for SMAP
        smap_mu = torch.Tensor(smap_psfmodel.ExpectedValue(generate_params(smap_psfmodel, I=I, bg=bg))).to(device)
        chisq_smap = chi_squared(observed=zstack, expected=smap_mu[None, :])  # (n_beads, zsize, roisize, roisize)

        chisq_res_smap[I] = [chisq_smap.sum((-3, -2, -1)).mean(0).item(), chisq_smap.sum((-3, -2, -1)).std(0).item()]
        if observed != 'gt':
            chisq_res_expected_smap[I] = [expected_mu, np.sqrt(2 * expected_mu + (1 / smap_mu).sum().item())]

        diff_mus[I] = diff_mu.cpu().numpy()
        smap_mus[I] = smap_mu.cpu().numpy()

    return chisq_res_diff, chisq_res_smap, chisq_res_expected_diff, chisq_res_expected_smap, diff_mus, smap_mus


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


def is_cutoff(vals, exp):
    return vals[:,0] - vals[:,1] < exp[:,0] + exp[:,1]


def find_cutoff(chisq_dict, exp_dict):
    chisq_vals = np.array(list(chisq_dict.values()))
    exp_vals = np.array(list(exp_dict.values()))
    return list(chisq_dict.keys())[np.where(is_cutoff(chisq_vals, exp_vals))[0].max()]