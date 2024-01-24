import torch
import numpy as np
import matplotlib.pyplot as plt


from dspline.spline_psf import CSplinePSFEstimator
from dspline.struct_type import struct

from view_utils.utils import chi_squared, get_stats_along_z, make_psf_figure

import tifffile
import pickle


def run_pipeline_new(psf_src='psfsim.pickle', load_zstacks=False, cuda_index=0, intensity=[5000], background=20, zrange=(10,-9),
                  n_beads=9,
                  detection_thr=1.2, detection_sigma=3,
                  npass=10, iterations_per_pass=50, lr=0.001, opt='Adam', training_plot=True,
                 gt_I=None, gt_bg=None, gt_PSF=None, alt_target=None):
    """
    Run the pipeline for PSF calibration

    psf_src - name of the .pickle file containing struct with the psf and microscope settings, None if done from real data
    load_zstacks - str, path to zstacks (e.g. exp/data/zstack) if using previously generated zstacks, False otherwise
    cuda_index - index for cuda device, cpu if negative
    """
    device = torch.device("cuda:" + str(cuda_index)) if torch.cuda.is_available() and cuda_index >= 0 else torch.device(
        "cpu")
    # load psf
    with open(psf_src, 'rb') as f:
        d = pickle.load(f)
        true_psf = d.psf / d.psf.sum((-2, -1), keepdims=True)  # normalize


    # true_psf_model = tifffile.imread(f'{load_zstacks}_true.tif')
    true_psf_model = None  # for now
    noised_zstack = tifffile.imread(f'{load_zstacks}.tif')[None, :]  # just read in just one zstack for now
    noised_zstack_tensor = torch.FloatTensor(noised_zstack).to(device)

    print_out = ''
    # spline initialization and training
    estimator = CSplinePSFEstimator(stepsize_nm=d.zres, device=device)
    for i in range(noised_zstack_tensor.shape[0]):
        print_out += estimator.add_zstack(noised_zstack_tensor[i], threshold=detection_thr, detection_sigma=detection_sigma,
                                          roisize=d.roisize)


    print_out += estimator.init(gt_I=gt_I, gt_bg=gt_bg, gt_PSF=gt_PSF, n_beads=n_beads)

    init_psf = estimator.eval(roisize=d.roisize, zplanes=d.zsize).detach().cpu().numpy()  # outputs eval for each bead

    opt_params = (['theta'] if gt_I is None else []) + (['spline']) #if gt_PSF is None else [])
    estimator.optimize(npass=npass, iterations_per_pass=iterations_per_pass, params=opt_params, lr=lr, opt=opt, plot=training_plot,
                       loss_type='mse', figures=figures, alt_target=alt_target)
    res = estimator.eval(roisize=d.roisize, zplanes=d.zsize).detach().cpu().numpy()

    print(print_out)

    return true_psf_model, noised_zstack, init_psf, res, estimator, d



def run_pipeline(psf_src='psfsim.pickle', load_zstacks=False, cuda_index=0, intensity=[10000], background=100,
                 num_zstacks=1,
                 num_beads=3, zrange=(10,-9), detection_thr=1.2, detection_sigma=3, plots=False, npass=10, iterations_per_pass=50, lr=0.001, opt='Adam', training_plot=True,
                 training_figures=False, params={"roisize": 24, "zres": 10}, gt_I=None, gt_bg=None, gt_PSF=None, alt_target=None):
    """
    Run the pipeline for PSF calibration

    psf_src - name of the .pickle file containing struct with the psf and microscope settings, None if done from real data
    load_zstacks - str, path to zstacks (e.g. exp/data/zstack) if using previously generated zstacks, False otherwise
    cuda_index - index for cuda device, cpu if negative
    """
    device = torch.device("cuda:" + str(cuda_index)) if torch.cuda.is_available() and cuda_index >= 0 else torch.device(
        "cpu")
    # load psf
    if psf_src is not None:
        with open(psf_src, 'rb') as f:
            d = pickle.load(f)
            true_psf = d.psf / d.psf.sum((-2, -1), keepdims=True)  # normalize


        if load_zstacks:
            # true_psf_model = tifffile.imread(f'{load_zstacks}_true.tif')
            true_psf_model = None  # for now
            noised_zstack = tifffile.imread(f'{load_zstacks}.tif')[None, :]  # just read in just one zstack for now
        else:
            # creating just for one zstack with multiple beads
            true_psf_model = intensity[:, None, None, None] * true_psf + background
            true_zstack = create_zstack(true_psf, d.zsize, num_beads, roisize=d.roisize, psf_zrange=d.zrange,
                                        zrange=zrange, intensity=intensity, background=background, pad=1)
            noised_zstack = np.random.poisson(true_zstack)[None, :] #true_zstack[None, :]
            plt.imshow(true_zstack[0])
            plt.show()
            # true_image = intensity * true_psf + background
            # noised_image = np.random.poisson(true_image, size=(num_beads,)+true_image.shape)
        noised_zstack_tensor = torch.FloatTensor(noised_zstack).to(device)

        print_out = ''
        # spline initialization and training
        estimator = CSplinePSFEstimator(stepsize_nm=d.zres, device=device)
        for i in range(noised_zstack_tensor.shape[0]):
            print_out += estimator.add_zstack(noised_zstack_tensor[i], threshold=detection_thr, detection_sigma=detection_sigma,
                                              roisize=d.roisize)
    else:
        true_psf_model = None
        noised_zstack = []  # list because the zstacks can be of different x,y sizes
        for i in range(num_zstacks):
            noised_zstack.append(tifffile.imread(f'{load_zstacks}{i}.tif').astype(np.float32))

        assert len(np.unique([zst.shape[0] for zst in noised_zstack])) == 1  # zrange is the same for all zstacks

        d = struct(**params)
        d["zsize"] = noised_zstack[0].shape[0]
        d["zrange"] = (np.arange(d.zsize) - (
                    d.zsize + 1) // 2) * d.zres  # zrange from HanserPSF _get_zrange() in otf.py package

        print_out = ''
        # spline initialization and training
        estimator = CSplinePSFEstimator(stepsize_nm=d.zres, device=device)
        for i in range(num_zstacks):
            print(noised_zstack[i])
            print_out += estimator.add_zstack(torch.FloatTensor(noised_zstack[i]), threshold=detection_thr, detection_sigma=detection_sigma,
                                              roisize=d.roisize)

    print_out += estimator.init(gt_I=gt_I, gt_bg=gt_bg, gt_PSF=gt_PSF)

    init_psf = estimator.eval(roisize=d.roisize, zplanes=d.zsize).detach().cpu().numpy()  # outputs eval for each bead

    if training_figures:
        figures = ([true_image, noised_image.mean(0), init_psf.mean(0)], d.zrange)
    else:
        figures = None

    opt_params = (['theta'] if gt_I is None else []) + (['spline']) #if gt_PSF is None else [])
    estimator.optimize(npass=npass, iterations_per_pass=iterations_per_pass, params=opt_params, lr=lr, opt=opt, plot=training_plot,
                       loss_type='mse', figures=figures, alt_target=alt_target)
    res = estimator.eval(roisize=d.roisize, zplanes=d.zsize).detach().cpu().numpy()

    print(print_out)

    # figures
    if plots:
        make_psf_figure([true_image, noised_image.mean(0), init_psf.mean(0), res.mean(0)], d.zrange,
                        labels=['Ground truth', 'Ground truth image', 'Initial diffspline', 'Trained diffspline'])

        chi_init = chi_squared(noised_image, init_psf).mean(0)
        chi_trained = chi_squared(noised_image, res).mean(0)

        make_psf_figure([chi_init, chi_trained], d.zrange,
                        labels=['$\chi^{2}$(GT, Initial diffspline)', '$\chi^{2}$(GT, Trained diffspline)'])

        get_stats_along_z([chi_init, chi_trained], d.zrange,
                          labels=['$\chi^{2}$(GT, Initial diffspline)', '$\chi^{2}$(GT, Trained diffspline)'])

    return true_psf_model, noised_zstack, init_psf, res, estimator, d


def create_zstack_from_multiple_psfs(psfs: list, zsize, roisize, psf_zrange, zrange=None, intensity=[10000], background=100, pad=1):
    ## psfs with subpixel shift
    ## zrange is a pair of indices (i,j), the psf will be cut from i-th to j-th z position
    
    
    n_beads = len(psfs)
    if len(intensity) == 1:
        intensity *= n_beads

    if type(n_beads) != tuple:
        n_columns = int(np.sqrt(n_beads))
        n_rows = int(np.ceil(n_beads / n_columns))  # take one more non-full row
    else:
        n_rows, n_columns = n_beads

    if zrange is None:
        zrange = (None, None)
    else:
        zsize = psf_zrange[zrange[0]:zrange[1]].shape[0]

    zstack = np.zeros((zsize, n_rows * (roisize + 2 * pad) + roisize, n_columns * (roisize + 2 * pad) + roisize))
    zstack += background

    k = 0
    for i in range(roisize // 2 + 1, n_rows * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
        for j in range(roisize // 2 + 1, n_columns * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
            zstack[:, i:i + roisize, j:j + roisize] += psfs[k][zrange[0]:zrange[1]] * intensity[k]
            k += 1

    return zstack


def create_zstack(psf, zsize, n_beads, roisize, psf_zrange, zrange=None, intensity=[10000], background=100, pad=1):
    ## no subpixel shift

    ## zrange is in indices

    if type(n_beads) != tuple:
        n_columns = int(np.sqrt(n_beads))
        n_rows = int(np.ceil(n_beads / n_columns))  # take one more non-full row
    else:
        n_rows, n_columns = n_beads

    if zrange is None:
        zrange = (None, None)
    else:
        zsize = psf_zrange[zrange[0]:zrange[1]].shape[0]

    zstack = np.zeros((zsize, n_rows * (roisize + 2 * pad) + roisize, n_columns * (roisize + 2 * pad) + roisize))
    zstack += background

    k = 0
    for i in range(roisize // 2 + 1, n_rows * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
        for j in range(roisize // 2 + 1, n_columns * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
            zstack[:, i:i + roisize, j:j + roisize] += psf[zrange[0]:zrange[1]] * intensity[k]
            k += 1

    return zstack


def _create_zstack(psf, xysize, xyres, zres, zsize, n_beads, roisize, intensity=10000, background=100, pad=1):
    ## no subpixel shift
    ## keeping but not using the parameters xysize, xyres, zres for backwards compatibility with Step 1 notebook
    if type(n_beads) != tuple:
        n_columns = int(np.sqrt(n_beads))
        n_rows = int(np.ceil(n_beads / n_columns))  # take one more non-full row
    else:
        n_rows, n_columns = n_beads

    zstack = np.zeros((zsize, n_rows * (roisize + 2 * pad) + roisize, n_columns * (roisize + 2 * pad) + roisize))
    zstack += background

    for i in range(roisize // 2 + 1, n_rows * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
        for j in range(roisize // 2 + 1, n_columns * (roisize + 2 * pad) + roisize // 2, roisize + 2 * pad):
            zstack[:, i:i + roisize, j:j + roisize] += psf * intensity

    return zstack