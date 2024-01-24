from service_functions.data_creation import generate_base_PSF, generate_shifted_PSFs
from service_functions.loading import get_basepsf_data, get_smoothpsf_data, generate_params, load_psfmodel, load_matlab_matpsf

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from dspline.spline_psf import CSplinePSFEstimator4D
from fastpsf import Context, CSplineMethods



def produce_estimations(psfmodel, base_folder, est_I=5000, est_bg=20, roipos=(0,0),
                        specific=None, nums='????', calibration_depth_ind=0,
                        plot_title='4D Diffspline interpolating expected SMAP data at 3000 vs. Diffspline 3D GT in Z',
                        n_samples=100, damp=0, iterations=50,
                        ylim=None, plot=True, return_errors=False):
    '''
    Estimate PSFmodel on the given depth on ROIs produced by the provided PSF. Options "specific" and "nums + calibraiton_depth_ind" are equivalent.
    '''
    if specific is not None:
        all_psfs, all_depths = get_basepsf_data(base_folder, pop=None, specific=[specific] if type(specific) is not list else specific)  # 0 to take the psfs, 2 - for depth 2000
    else:
        all_psfs, all_depths = get_basepsf_data(base_folder, pop=None, nums=nums)
    gt_psf = all_psfs[calibration_depth_ind]
    #print(all_depths)

    gt_params = generate_params(psfmodel, I=est_I, bg=est_bg, roipos=roipos)
    zrange = np.linspace(*(psfmodel.calib.zrange), psfmodel.calib.n_voxels_z + 1) * 1000

    psfmodel.SetLevMarParams(stepCoeffs=np.array([damp, damp, damp, damp, damp]), normalizeWeights=True,
                             iterations=iterations)

    rois = np.vstack(np.random.poisson(est_I * gt_psf + est_bg, size=(n_samples, *gt_psf.shape)))
    initial_guess = gt_params[None, ...].repeat(n_samples, axis=0).reshape(-1, 5)
    estim, _, traces = psfmodel.Estimate(rois, initial=initial_guess)
    estims = estim.reshape(n_samples, -1, 5).transpose(1, 2, 0) #[:, :, 1:]  # why cut one?

    if plot:
        plt.figure(figsize=(5,3))
        plt.plot(zrange[10:-10], gt_params[:, 2, None][10:-10] * 1000, label='z gt')
        plt.plot(zrange[10:-10], estims.mean(-1)[10:-10, 2] * 1000, label='z spline preds')
        plt.legend()
        plt.title(f'{plot_title} vs. 3D GT in Z for eval depth={all_depths[calibration_depth_ind]}nm')
        plt.ylabel('GT z, nm')
        plt.xlabel('Predicted z, nm')
        plt.show()

        plt.figure(figsize=(5, 3))
        bias = ((estims[:, 2, :] * 1000) - gt_params[:, 2, None] * 1000).mean(1)
        bias_err = ((estims[:, 2, :] * 1000) - gt_params[:, 2, None] * 1000).std(1)
        plt.errorbar(zrange, bias, yerr=bias_err)

        # errors = (estims[:,2,:]*1000) - gt_params[:,2,None]*1000
        # bias = errors.mean(-1)
        # quant_down = bias - np.quantile(a=errors, q=0.25, axis=-1)
        # quant_up = np.quantile(a=errors, q=0.75, axis=-1) - bias
        # plt.errorbar(zrange, bias, yerr=np.vstack([quant_down, quant_up]))

        plt.title(f"{plot_title} bias in z for depth={all_depths[calibration_depth_ind]}nm")
        plt.xlabel("Z position, nm")
        plt.ylabel("Bias, nm")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.show()

        plt.figure(figsize=(5, 3))
        plt.errorbar(zrange[10:-10], bias[10:-10], yerr=bias_err[10:-10])
        plt.title(f"{plot_title} bias in z (closer ranges)")
        plt.xlabel("Z position, nm")
        plt.ylabel("Bias, nm")
        if ylim is not None:
            plt.ylim(*ylim)
        plt.show()

    if return_errors:
        return estims, traces, estims - gt_params[..., None]
    return estims, traces


def plot_estims(estims: list, psfmodels: list, labels=[''], depth=3000, est_I=15000, est_bg=50, roipos=(0,0), errortype='std'):
    gt_params_list = []
    for psfmodel in psfmodels:
        gt_params_list.append(generate_params(psfmodel, I=est_I, bg=est_bg, roipos=roipos))

    zrange = np.linspace(*(psfmodels[0].calib.zrange), psfmodels[0].calib.n_voxels_z + 1) * 1000

    plt.figure(figsize=(5, 3))
    plt.plot(zrange[10:-10], gt_params_list[0][:, 2, None][10:-10] * 1000, label='z gt')
    for i, estim in enumerate(estims):
        plt.plot(zrange[10:-10], estim.mean(-1)[10:-10, 2] * 1000, label=f'{labels[i]} z-preds')
    plt.legend()
    plt.title(f'Splines vs. 3D GT in Z for eval depth={depth} nm')
    plt.ylabel('GT z, nm')
    plt.xlabel('Predicted z, nm')
    plt.show()

    plt.figure(figsize=(5, 3))
    for i, estim in enumerate(estims):
        errors = (estim[:, 2, :] * 1000) - gt_params_list[i][:, 2, None] * 1000
        bias = errors.mean(1)
        if errortype == 'quantile':
            quant_down = bias - np.quantile(a=errors, q=0.25, axis=1)
            quant_up = np.quantile(a=errors, q=0.75, axis=1) - bias
            plt.errorbar(zrange, bias, yerr=np.vstack([quant_down, quant_up]), label=labels[i])
        elif errortype == 'std':
            plt.errorbar(zrange, bias, yerr=errors.std(1), label=labels[i])

    plt.title(f"Spline bias in z for depth={depth} nm")
    plt.xlabel("Z position, nm")
    plt.ylabel("Bias, nm")
    plt.legend()
    # plt.ylim(*ylim)
    plt.show()

    plt.figure(figsize=(5, 3))
    for i, estim in enumerate(estims):
        errors = (estim[:, 2, :] * 1000) - gt_params_list[i][:, 2, None] * 1000
        bias = errors.mean(1)
        if errortype == 'quantile':
            quant_down = bias - np.quantile(a=errors, q=0.25, axis=1)
            quant_up = np.quantile(a=errors, q=0.75, axis=1) - bias
            plt.errorbar(zrange[10:-10], bias[10:-10], yerr=np.vstack([quant_down, quant_up])[:, 10:-10], label=labels[i])
        elif errortype == 'std':
            plt.errorbar(zrange[10:-10], bias[10:-10], yerr=errors.std(1)[10:-10], label=labels[i])

    plt.title(f"Spline bias in z (closer ranges)")
    plt.xlabel("Z position, nm")
    plt.ylabel("Bias, nm")
    plt.legend()
    # plt.ylim(*ylim)
    plt.show()


def plot_bias(psfmodels: list, base_folder, depth, I, bg, labels, damp=10000, iterations=250, n_samples=100, roiposs=[(0,0),(0,0)],
              translations=[100, 100, 1000], colors=['r','g','b'], axis=['x', 'y', 'z'], figsize=(15, 12),
              zrange_lim=(None, None), plot_estimations=False, plot_rmse=False, savefig_name=''):
    plt.figure(figsize=figsize)
    errors = []
    for i, psfmodel in enumerate(psfmodels):
        errors.append(produce_estimations(psfmodel, base_folder, est_I=I, est_bg=bg, roipos=roiposs[i], damp=damp, iterations=iterations,
                                            specific=[depth], n_samples=n_samples,
                                          plot=plot_estimations, plot_title=labels[i], return_errors=True)[-1])  # use only errors
    biases = [[] for _ in errors]
    for j in [2]:#range(3):
        #plt.subplot2grid((3, 1), (j, 0))
        for i, error in enumerate(errors):
            zrange = np.linspace(*psfmodels[i].calib.zrange, num=error.shape[0]) * translations[2]
            bias = (abs(error[:, j, :]) * translations[j]).mean(-1)
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], bias[zrange_lim[0]:zrange_lim[1]], label=f'{labels[i]} bias', marker='o', linewidth=0, c=colors[i])
            biases[i].append(bias[zrange_lim[0]:zrange_lim[1]])
            if plot_rmse:
                plt.plot(np.linspace(*psfmodels[i].calib.zrange, num=error.shape[0])*translations[2],
                     np.sqrt(((error[:, j, :] * translations[j]) ** 2).mean(-1)), label=f'{labels[i]} rmse', marker='x',
                     linewidth=0, c=colors[i])
        if j == 2:
            #plt.title(' vs. '.join(labels) + ' bias')
            plt.legend()
        plt.ylabel(f'bias in {axis[j]}, nm')
    plt.xlabel('z-position, nm')
    plt.savefig(f'z-bias{savefig_name}.svg')
    plt.show()
    return biases


def calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels, damp=10000, iterations=250,
                                  n_samples=100, roiposs=[(0,0),(0,0)], return_estims=False, plot_estimations=False):
    errors = []
    crlbs = []
    estims = []
    for i, psfmodel in enumerate(psfmodels):
            crlbs.append(psfmodel.CRLB(generate_params(psfmodel, I=I, bg=bg, roipos=roiposs[i])))
            specific = depth[i] if type(depth)==list else [depth]
            estim, _, error = produce_estimations(psfmodel, base_folder, est_I=I, est_bg=bg, roipos=roiposs[i], damp=damp, iterations=iterations,
                                    specific=specific, n_samples=n_samples,
                                    plot=plot_estimations, plot_title=labels[i], return_errors=True)
            estims.append(estim)
            errors.append(error)
    if return_estims:
        return errors, crlbs, estims
    return errors, crlbs


def plot_precision_with_crlb(psfmodels: list, base_folder, depth, I, bg, labels, damp=10000, iterations=250, n_samples=100, roiposs=[(0,0),(0,0)],
                             translations=[100, 100, 1000], colors=['r', 'g', 'b'], axis=['x', 'y', 'z'], figsize=(15,12),
                             zlimit=None, zrange_lim=(None, None), plot_estimations=False, ddof=0, savefig_name=''):
    plt.figure(figsize=figsize)
    errors, crlbs = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                                  damp=damp, iterations=iterations, n_samples=n_samples,
                                                  roiposs=roiposs, plot_estimations=plot_estimations)
    res_precs = [[] for _ in errors]
    res_crlbs = [[] for _ in errors]
    for j in [2]: #range(3):
        # plt.subplot2grid((3, 1), (j, 0))
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            precision = (error[:, j, :] * translations[j]).std(-1, ddof=ddof)
            zrange = np.linspace(*psfmodels[i].calib.zrange, num=error.shape[0]) * translations[2]
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], crlb[zrange_lim[0]:zrange_lim[1]], label=f'{labels[i]} CRLB', c=colors[i])
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], precision[zrange_lim[0]:zrange_lim[1]], marker='x', linewidth=0, label=f'{labels[i]} precision', c=colors[i])
            res_precs[i].append(precision[zrange_lim[0]:zrange_lim[1]])
            res_crlbs[i].append(crlb[zrange_lim[0]:zrange_lim[1]])
            if zlimit is not None and j == 2:
                plt.ylim(zlimit)
        if j == 2:
            #plt.title(' vs. '.join(labels) + ' precision (std)')
            plt.legend()
        if j == 2:
            plt.xlabel('z-position, nm')
        plt.ylabel(f'precision in {axis[j]}, nm')
        '''
        plt.subplot2grid((3, 2), (j, 1))
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            precision = (error[:, j, :] * translations[j]).std(-1)
            zrange = np.linspace(*psfmodels[i].calib.zrange, num=error.shape[0]) * translations[2]
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], abs(precision - crlb)[zrange_lim[0]:zrange_lim[1]], label=f'{labels[i]} gap with total={round((abs(precision - crlb)[zrange_lim[0]:zrange_lim[1]]).sum())}', c=colors[i])
            plt.legend()
        if j == 0:
            plt.title(' vs. '.join(labels) + ' Precision-CRLB gap')
        if j == 2:
            plt.xlabel('z-position, nm')
        plt.ylabel(f'Precision-CRLB gap in {axis[j]}, nm')
        '''

    plt.savefig(f'z-precision{savefig_name}.svg')
    plt.show()
    return res_precs, res_crlbs
    

def calculate_axial_gap_over_intensity(psfmodels, base_folder, depth, bg, labels, damp=100, iterations=100, n_samples=100,
                                 roiposs=[(0,0),(0,0)], translations=[100, 100, 1000], zrange_lim=(None,None),
                                 int_start=1000, int_stop=20000, int_nsteps=20,
                                 plot_estimations=False, plot_gaps=True):
    gaps = [{} for _ in psfmodels]
    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):
        errors, crlbs = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                                      damp=damp, iterations=iterations, n_samples=n_samples,
                                                      roiposs=roiposs, plot_estimations=plot_estimations)
        j = 2  ## only show results in Z
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            precision = (error[:, j, :] * translations[j]).std(-1)
            gaps[i][I] = (abs(precision - crlb)[zrange_lim[0]:zrange_lim[1]]).mean()  # iI?

    if plot_gaps:
        for i, gap in enumerate(gaps):
            plt.plot(gap.keys(), np.array(list(gap.values())), label=labels[i])

        plt.title(f'|Precision-CRLB| averaged in Z-axis')
        plt.xlabel('Intensity, photons')
        plt.ylabel('Average |Precision-CRLB|, nm')
        plt.legend()
        plt.show()
    return gaps


def show_estims_for_depths(psfmodel, depthrange, base_folder, I, bg, title='', damp=10000, iterations=250, figsize=(7,4),
                                  n_samples=100, translation=1000, roisize=34, plot_estimations=False, savefig_name=''):
    if title == 'Diffspline':
        psfmodel = load_psfmodel(f'{base_folder}/astig_PSF_diffspline_dp{depthrange[0]:04d}.mat', roisize=roisize)
    if title == 'Cspline recalibrated':
        psfmodel = load_psfmodel(f'{base_folder}/astig_smooth_sph{depthrange[0]:04d}_3Dcorr.mat', roisize=roisize)
    zrange = np.linspace(*psfmodel.calib.zrange, num=psfmodel.calib.n_voxels_z + 1) * translation
    roiposs = [(0,0) for _ in depthrange]
    plt.figure(figsize=figsize)
    cmap = plt.colormaps['brg'](np.linspace(0, 1, num=len(depthrange)))  #'brg' 'gnuplot2' 'rainbow'
    for i, depth in enumerate(depthrange):
        if title == 'Diffspline':
            psfmodel = load_psfmodel(f'{base_folder}/astig_PSF_diffspline_dp{depth:04d}.mat', roisize=roisize)
        if title == 'Cspline recalibrated':
            psfmodel = load_psfmodel(f'{base_folder}/astig_smooth_sph{depth:04d}_3Dcorr.mat', roisize=roisize)
        preds, _ = produce_estimations(psfmodel, base_folder, est_I=I, est_bg=bg, roipos=roiposs[i], damp=damp,
                                      iterations=iterations, specific=[depth], n_samples=n_samples,
                                      plot=plot_estimations, return_errors=False)
        plt.scatter(depth+zrange, preds[:, 2, :].mean(-1)*translation, color=cmap[i], s=5)
    plt.xlabel('focal plane position, nm')
    plt.ylabel('predicted z-position, nm')
    plt.title(f'{title} z-estimations')
    plt.savefig(f'estims{title.split(" ")[0] if savefig_name == "" else savefig_name}.svg')
    plt.show()
    return


def plot_rmse(psfmodels: list, base_folder, depth, I, bg, labels, damp=10000, iterations=250, n_samples=100,
              roiposs=[(0, 0), (0, 0)],
              translations=[100, 100, 1000], colors=['r', 'g', 'b'], axis=['x', 'y', 'z'], figsize=(15, 12),
              zrange_lim=(None, None), plot_estimations=False, ):
    plt.figure(figsize=figsize)
    errors, crlbs = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                               damp=damp, iterations=iterations, n_samples=n_samples,
                                               roiposs=roiposs, plot_estimations=plot_estimations)
    for j in range(3):
        plt.subplot2grid((3, 1), (j, 0))
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            rmse = np.sqrt(((error[:, j, :] * translations[j]) ** 2).mean(-1))
            zrange = np.linspace(*psfmodels[i].calib.zrange, num=error.shape[0]) * translations[2]
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], crlb[zrange_lim[0]:zrange_lim[1]], label=f'{labels[i]} CRLB',
                     c=colors[i])
            plt.plot(zrange[zrange_lim[0]:zrange_lim[1]], rmse[zrange_lim[0]:zrange_lim[1]],
                     label=f'{labels[i]} rmse', marker='x', linewidth=0, c=colors[i])
        if j == 0:
            plt.title(' vs. '.join(labels) + ' RMSE')
            plt.legend()
        plt.ylabel(f'RMSE in {axis[j]}, nm')
    plt.xlabel('z-position, nm')
    plt.show()


def calculate_axial_rmsebycrlb_over_intensity(psfmodels, base_folder, depth, bg, labels, damp=100, iterations=100, n_samples=100,
                                 roiposs=[(0,0),(0,0)], translations=[100, 100, 1000], zrange_lim=(None,None),
                                 int_start=1000, int_stop=20000, int_nsteps=20,
                                 plot_estimations=False, plot_gaps=True):
    res = [{} for _ in psfmodels]
    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):
        errors, crlbs = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                                      damp=damp, iterations=iterations, n_samples=n_samples,
                                                      roiposs=roiposs, plot_estimations=plot_estimations)
        j = 2  ## only show results in Z
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            rmse = np.sqrt(((error[:, j, :] * translations[j]) ** 2).mean(-1))
            res[i][I] = ((rmse/crlb)[zrange_lim[0]:zrange_lim[1]]).mean()

    if plot_gaps:
        for i, elem in enumerate(res):
            plt.plot(elem.keys(), np.array(list(elem.values())), label=labels[i])

        plt.title(f'RMSE/CRLB ratio averaged in Z-axis')
        plt.xlabel('Intensity, photons')
        plt.ylabel('Average RMSE/CRLB')
        plt.legend()
        plt.show()
    return res


def calculate_axial_rmsemincrlb_over_intensity(psfmodels, base_folder, depth, bg, labels, damp=100, iterations=100, n_samples=100,
                                 roiposs=[(0,0),(0,0)], translations=[100, 100, 1000], zrange_lim=(None,None),
                                 int_start=1000, int_stop=20000, int_nsteps=20,
                                 plot_estimations=False, plot_gaps=True):
    res = [{} for _ in psfmodels]
    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):
        errors, crlbs = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                                      damp=damp, iterations=iterations, n_samples=n_samples,
                                                      roiposs=roiposs, plot_estimations=plot_estimations)
        j = 2  ## only show results in Z
        for i, error in enumerate(errors):
            crlb = crlbs[i][:, j] * translations[j]
            rmse = np.sqrt(((error[:, j, :] * translations[j]) ** 2).mean(-1))
            res[i][I] = (abs(rmse-crlb)[zrange_lim[0]:zrange_lim[1]]).mean()

    if plot_gaps:
        for i, elem in enumerate(res):
            plt.plot(elem.keys(), np.array(list(elem.values())), label=labels[i])

        plt.title(f'|RMSE-CRLB| averaged in Z-axis')
        plt.xlabel('Intensity, photons')
        plt.ylabel('Average |RMSE-CRLB|, nm')
        plt.legend()
        plt.show()
    return res


def calculate_precision_over_intensity(psfmodels, base_folder, depth, bg, labels, damp=100, iterations=100, n_samples=100,
                                 roiposs=[(0,0),(0,0)], translations=[100, 100, 1000], zrange_lim=(None,None),
                                 int_start=1000, int_stop=20000, int_nsteps=20,
                                 plot_estimations=False, plot_gaps=True):
    prec = [{} for _ in psfmodels]
    for I in np.linspace(int_start, int_stop, int_nsteps, dtype=int):
        errors, _ = calculate_errors_with_crlb(psfmodels, base_folder, depth, I, bg, labels,
                                                      damp=damp, iterations=iterations, n_samples=n_samples,
                                                      roiposs=roiposs, plot_estimations=plot_estimations)
        j = 2  ## only show results in Z
        for i, error in enumerate(errors):
            precision = (error[:, j, :] * translations[j]).std(-1)
            prec[i][I] = (precision[zrange_lim[0]:zrange_lim[1]]).mean()  # iI?

    if plot_gaps:
        for i, val in enumerate(prec):
            plt.plot(val.keys(), np.array(list(val.values())), label=labels[i])

        plt.title(f'Precision averaged in Z-axis')
        plt.xlabel('Intensity, photons')
        plt.ylabel('Average precision, nm')
        plt.legend()
        plt.show()
    return prec


def plot_precisions_over_intensity_for_depths(psfmodels, base_folder, depths, bg, basic_labels, damp=100, iterations=100, n_samples=100,
                                 roiposs=[(0,0),(0,0)], roisize=34, translations=[100, 100, 1000], zrange_lim=(None,None),
                                 int_start=1000, int_stop=20000, int_nsteps=20,
                                 plot_estimations=False, plot_gaps=True, figsize=(7,4), savefig_name=''):
    markers = ['o', '^', 'x']

    # smap_psfmodel = load_psfmodel(f'{base_folder}/astig_PSF_smooth_sph{smap_depth:04d}_3Dcorr.mat', roisize=roisize)

    cmap_cspl = plt.colormaps['Greens'](np.linspace(0.3, 0.8, num=len(depths)))
    cmap_dfspl = plt.colormaps['Reds'](np.linspace(0.3, 0.8, num=len(depths)))
    cmaps = [cmap_dfspl, cmap_cspl]

    plt.figure(figsize=figsize)

    precs_over_ints = []
    for j, depth in enumerate(depths):
        labels = [f'{basic_labels[0].split(" ")[0]} {["on coverslip" if depth == 0 else "at " + str(depth) + " nm"][0]}',
                        f'{basic_labels[1].split(" ")[0]} {["on coverslip" if depth == 0 else "at " + str(depth) + " nm"][0]}']
        base_psf, _ = load_matlab_matpsf(f'{base_folder}/astig_PSF_base_sph{depth:04d}.mat')
        dfspl_name = f'{base_folder}/astig_PSF_diffspline_dp{depth:04d}.mat'
        diffspline_psfmodel = load_psfmodel(dfspl_name, roisize=roisize)
        psfmodels[0] = diffspline_psfmodel
        prec = calculate_precision_over_intensity(psfmodels, base_folder, depth, bg,
                                                  labels=labels, damp=damp, iterations=iterations, n_samples=n_samples,
                                                  roiposs=roiposs, translations=translations, zrange_lim=zrange_lim,
                                                  int_start=int_start, int_stop=int_stop, int_nsteps=int_nsteps,
                                                  plot_estimations=plot_estimations, plot_gaps=plot_gaps)
        for i, val in enumerate(prec):
            plt.scatter(val.keys(), np.array(list(val.values())), marker=markers[j], label=labels[i],
                        color=cmaps[i][j])
        precs_over_ints.append(prec)
    # plt.title(f'Precision averaged in Z-axis')
    plt.xlabel('intensity, photons')
    plt.ylabel('precision (averaged over z), nm')
    plt.legend()
    plt.savefig(f'precisions{savefig_name}.svg')
    plt.show()
    return precs_over_ints

def get_metrics_over_delta_for_depth(depth, delta, num_evals, base_folder, I=10000, bg=50, roisize=34, 
                          damp=10, iterations=100, n_samples=100, zrange_lim=(None,None),
                          roiposs=[(0,0)], translations=(100,100,1000), device='cuda'):
    '''
    Delta_depth1 & 2 - the interval (d1,d2) to interpolate over, delta = d2 - d1
    Subdelta - subinterval size
    num_evals - number of uniformly distributed evaluation points within the delta
    '''
    from service_functions.calibration import calibrate_diffspline_at_depth
    
    # # Load data with the chosen delta
    # Load 4 data points to interpolate over (training/calibration data)
    specific = [depth - delta, depth, depth + delta, depth + 2 * delta]
    calibration_psfs, calibration_depths = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?', specific=specific)
    
    # Load all subinterval data (test data)
    subinterval_depths = list(np.linspace(depth, depth + delta, num_evals+2)[1:-1])
    os.makedirs(base_folder+'/subdeltas/', exist_ok=True)
    for d in subinterval_depths:
        generate_base_PSF(d, base_folder, extra_folder='subdeltas/')
    #subinterval_depths = [int(elem) for elem in subinterval_depths]
    
    # Make test data
    #test_psfs, _ = get_basepsf_data(base_folder, nums=None, specific=subinterval_depths)

    # Make test data "observed" PSFs
    #observed_beads = np.random.poisson(I * calibration_psfs[:, None, ...].repeat(num_beads, axis=1) + bg)  # size = (subinterval_size, num_beads, *PSF.shape)

    diffspline_psfmodels = []
    # For a bunch of subinterval positions
    for i, subdepth in enumerate(subinterval_depths):
        ## Evaluate the diffspline at the position
        #print(f'Calibrating on depth={subdepth} nm')
        dfspl_name = f'{base_folder}/subdeltas/astig_PSF_diffspline_dp{subdepth:07.2f}.mat'
        positions = [-0.5, -0.5, 0]
        calibrate_diffspline_at_depth(subdepth, calibration_psfs, calibration_depths, base_folder, positions=positions,
                                      median_padding=False, name=dfspl_name, device=device)
        diffspline_psfmodels.append(load_psfmodel(dfspl_name, roisize=roisize))

        
    ## Find the metrics
    errors, crlbs, estims = calculate_errors_with_crlb(diffspline_psfmodels, base_folder, depth, I, bg, labels=[None]*len(diffspline_psfmodels), 
                                                       return_estims=True, plot_estimations=False, damp=damp, iterations=iterations, 
                                                       n_samples=n_samples, roiposs=[(0,0) for _ in range(len(diffspline_psfmodels))])
    res_biases = []
    res_precisions = []
    #res_crlbs = []
    res_estims = []
    derrors = []
    for i, error in enumerate(errors):
        derrors.append((error[:, 2, :]) * translations[2])  # then derrors becomes num_evals,101,100
        #bias = (abs(error[:, 2, :]) * translations[2]).mean(-1)
        #average_bias = (bias[zrange_lim[0]:zrange_lim[1]]).mean()
        #res_biases.append(average_bias)
        #precision = (error[:, 2, :] * translations[2]).std(-1)
        #average_precision = (precision[zrange_lim[0]:zrange_lim[1]]).mean()
        #res_precisions.append(average_precision)
        #crlb = crlbs[i][:, 2, :] * translations[2]
        #average_crlb = (crlb[zrange_lim[0]:zrange_lim[1]]).mean()
        #res_crlbs.append(average_crlb)
        zrange = depth + np.linspace(*diffspline_psfmodels[0].calib.zrange, num=diffspline_psfmodels[0].calib.n_voxels_z + 1) * translations[2]
        res_estims.append(estims[i][:, 2, :].mean(-1)*translations[2])
        #plt.scatter(depth+zrange, preds[:, 2, :].mean(-1)*translation, color=cmap[i], s=5)
    return res_biases, res_precisions, zrange, res_estims, np.array(derrors).flatten() #, res_crlbs



def plot_metrics_over_deltas(depths:list, deltas:list, num_evals, base_folder, I=10000, bg=50, roisize=34, 
                             damp=10, iterations=100, n_samples=100, zrange_lim=(None,None), figsize=(7,4),
                             translations=(100,100,1000), device='cuda', plot_preds=False, savefig_name=''):
    biases = []
    precs = []
    estims = []
    zranges = []
    derrors = {}
    plt.figure(figsize=figsize)
    for depth in depths:
        bias = {}
        prec = {}
        estim = {}
        errors = {}
        for delta in deltas:
            res_biases, res_precisions, zrange, res_estims, error = get_metrics_over_delta_for_depth(depth, delta, num_evals, base_folder, I, bg, roisize,
                                                                     damp, iterations, n_samples, zrange_lim, 
                                                                     translations=translations, device=device)
            #bias[delta] = np.mean(res_biases)
            #prec[delta] = np.mean(res_precisions)
            estim[delta] = np.array(res_estims)
            errors[delta] = error.flatten()
        #biases.append(bias)
        #precs.append(prec)
        estims.append(estim)
        zranges.append(zrange)
        derrors[depth] = errors

    arr = np.empty(shape=(1, 3))
    for j, depth in enumerate(derrors.keys()):
        for i, beaddists in enumerate(derrors[depth].keys()):
            tmp = np.vstack([[int(depth)] * n_samples*num_evals*len(zrange), [int(beaddists)] * n_samples*num_evals*len(zrange), derrors[depth][beaddists]]).T
            arr = np.vstack([arr, tmp])
    arr = arr[1:]
    df = pd.DataFrame(arr, columns=['depth', 'bead distance, nm', 'errors, nm'])
    sns.boxplot(df, x='bead distance, nm', y='errors, nm', hue='depth', showfliers=False)
    #sns.violinplot(df, x='bead distance, nm', y='errors, nm', hue='Depth')
    #for i in range(len(biases)):
    #    plt.errorbar(biases[i].keys(), np.array(list(biases[i].values())), yerr=np.array(list(precs[i].values())), label=f'depth = {depths[i]} nm')
    #plt.title('Diffspline bias and precision over varying bead distances measured at different depths')
    #plt.xlabel('distance between beads, nm')
    #plt.ylabel('average bias +- precision, nm')
    plt.legend()
    plt.savefig(f'beaddistance{savefig_name}.svg')
    plt.show()

    if plot_preds:
        cmap = plt.colormaps['brg'](np.linspace(0, 1, num=num_evals))
        for i in range(len(depths)):
            for delta in deltas:
                subinterval_delta = delta / (num_evals+1)
                subinterval_depths = list(np.arange(0, delta+subinterval_delta, subinterval_delta))
                for subdelta_i in range(num_evals+2):
                    if subdelta_i == 0 or subdelta_i == num_evals+1:
                        plt.scatter(zranges[i]+subinterval_depths[subdelta_i], zranges[i]-depths[i], color='black', s=5)
                    else:
                        plt.scatter(zranges[i]+subinterval_depths[subdelta_i], estims[i][delta][subdelta_i-1], color=cmap[subdelta_i-1], s=5)
                plt.xlabel('Focal plane position, nm')
                plt.ylabel('Predicted z-position, nm')
                plt.title(f'Diffspline z-estimations over depths')
                plt.show()
    return df


def plot_subdelta_errors_over_deltas_for_depth_boxplot(depth, delta, num_evals, base_folder, I=10000, bg=50, roisize=34,
                                     damp=10, iterations=100, n_samples=100, zrange_lim=(None, None),
                                     roiposs=[(0, 0)], translations=(100, 100, 1000), device='cuda',
                                     savefig_name=''):
    '''
    Delta_depth1 & 2 - the interval (d1,d2) to interpolate over, delta = d2 - d1
    Subdelta - subinterval size
    num_evals - number of uniformly distributed evaluation points within the delta
    '''
    from service_functions.calibration import calibrate_diffspline_at_depth

    # # Load data with the chosen delta
    # Load 4 data points to interpolate over (training/calibration data)
    specific = [depth - delta, depth, depth + delta, depth + 2 * delta]
    calibration_psfs, calibration_depths = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?',
                                                              specific=specific)

    # Load all subinterval data (test data)
    subinterval_depths = list(np.linspace(depth, depth + delta, num_evals + 2)[:-1])
    for d in subinterval_depths:
        generate_base_PSF(d, base_folder, extra_folder='subdeltas/')
    # subinterval_depths = [int(elem) for elem in subinterval_depths]

    # Make test data
    # test_psfs, _ = get_basepsf_data(base_folder, nums=None, specific=subinterval_depths)

    # Make test data "observed" PSFs
    # observed_beads = np.random.poisson(I * calibration_psfs[:, None, ...].repeat(num_beads, axis=1) + bg)  # size = (subinterval_size, num_beads, *PSF.shape)


    # For a bunch of subinterval positions
    arr = np.empty(shape=(1, 3))
    for i, subdepth in enumerate(subinterval_depths):
        ## Evaluate the diffspline at the position
        print(f'Calibrating on depth={subdepth} nm')
        dfspl_name = f'{base_folder}/subdeltas/astig_PSF_diffspline_dp{subdepth:07.2f}.mat'
        positions = [-0.5, -0.5, 0]
        calibrate_diffspline_at_depth(subdepth, calibration_psfs, calibration_depths, base_folder, positions=positions,
                                      median_padding=False, name=dfspl_name, device=device)
        diffspline_psfmodel = load_psfmodel(dfspl_name, roisize=roisize)

        smap_psfmodel_lower = load_psfmodel(f'{base_folder}/astig_smooth_sph{depth:04d}_3Dcorr.mat', roisize=roisize)
        smap_psfmodel_higher = load_psfmodel(f'{base_folder}/astig_smooth_sph{(depth+delta):04d}_3Dcorr.mat', roisize=roisize)

        ## Find the metrics
        errors, crlbs, estims = calculate_errors_with_crlb([diffspline_psfmodel, smap_psfmodel_lower, smap_psfmodel_higher], base_folder+'/subdeltas', 
                                                           subdepth, I, bg, labels=[None, None, None], return_estims=True, plot_estimations=False,
                                                           damp=damp, iterations=iterations, n_samples=n_samples, roiposs=[(0, 0), (0, 0), (0,0)])

        for i, error in enumerate(errors):
            converted_errors = (error[:, 2, :]) * translations[2]  # then derrors becomes num_evals,101,100
            #print(converted_errors.shape)
            zrange = depth + np.linspace(*diffspline_psfmodel.calib.zrange,
                                         num=diffspline_psfmodel.calib.n_voxels_z + 1) * translations[2]
            #res_estims.append(estims[i][:, 2, :].mean(-1) * translations[2])

            # type (smap or diffspline) and subdepth with errors on x axis
            splinetype = ['diffspline', f'cspline at {depth} nm', f'cspline at {int(depth + delta)} nm'][i]
            tmp = np.vstack([[splinetype] * n_samples * len(zrange),
                             [int(subdepth)] * n_samples * len(zrange),
                             converted_errors.flatten()]).T
            arr = np.vstack([arr, tmp])
    arr = arr[1:]
    df = pd.DataFrame(arr, columns=['splinetype', 'depth, nm', 'errors, nm'])
    df['errors, nm'] = df['errors, nm'].astype(float)
    df['depth, nm'] = df['depth, nm'].astype(float)
    sns.boxplot(df, x='depth, nm', y='errors, nm', hue='splinetype', showfliers=False)
    # sns.violinplot(df, x='bead distance, nm', y='errors, nm', hue='Depth')
    # for i in range(len(biases)):
    #    plt.errorbar(biases[i].keys(), np.array(list(biases[i].values())), yerr=np.array(list(precs[i].values())), label=f'depth = {depths[i]} nm')
    # plt.title('Diffspline bias and precision over varying bead distances measured at different depths')
    # plt.xlabel('distance between beads, nm')
    # plt.ylabel('average bias +- precision, nm')
    plt.legend()
    plt.savefig(f'beaddistance{savefig_name}.svg')
    plt.show()

    return df


def plot_subdelta_errors_over_deltas_for_depth(depth, delta, num_evals, base_folder, csplines=2, I=10000, bg=50, roisize=34,
                                     damp=10, iterations=100, n_samples=100, zrange_lim=(None, None),
                                     roiposs=[(0, 0)], translations=(100, 100, 1000), device='cuda',
                                     savefig_name=''):
    '''
    Delta_depth1 & 2 - the interval (d1,d2) to interpolate over, delta = d2 - d1
    Subdelta - subinterval size
    num_evals - number of uniformly distributed evaluation points within the delta
    '''
    from service_functions.calibration import calibrate_diffspline_at_depth

    # # Load data with the chosen delta
    # Load 4 data points to interpolate over (training/calibration data)
    specific = [depth - delta, depth, depth + delta, depth + 2 * delta]
    calibration_psfs, calibration_depths = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?',
                                                              specific=specific)

    # Load all subinterval data (test data)
    subinterval_depths = list(np.linspace(depth, depth + delta, num_evals + 2))
    subinterval_depths[-1] -= 1  # to avoid interpolation over another interval, we take max-epsilon, epsilon=1 nm
    for d in subinterval_depths:
        generate_base_PSF(d, base_folder, extra_folder='subdeltas/')
    
    # For several subinterval positions
    res_errors = [[],[],[]]  # for diffspline, cspline_lower, and cspline_higher
    splinetypes = ['diffspline', f'cspline at {depth} nm', f'cspline at {int(depth + delta)} nm']
    alphas = [0.5, 0.2, 0.2]
    for i, subdepth in enumerate(subinterval_depths):
        ## Evaluate the diffspline at the position
        print(f'Calibrating on depth={subdepth} nm')
        dfspl_name = f'{base_folder}/subdeltas/astig_PSF_diffspline_dp{subdepth:07.2f}.mat'
        positions = [-0.5, -0.5, 0]
        calibrate_diffspline_at_depth(subdepth, calibration_psfs, calibration_depths, base_folder, positions=positions,
                                      median_padding=False, name=dfspl_name, device=device)
        diffspline_psfmodel = load_psfmodel(dfspl_name, roisize=roisize)

        if csplines == 2:
            smap_psfmodel_lower = load_psfmodel(f'{base_folder}/astig_smooth_sph{depth:04d}_3Dcorr.mat', roisize=roisize)
            smap_psfmodel_higher = load_psfmodel(f'{base_folder}/astig_smooth_sph{(depth+delta):04d}_3Dcorr.mat', roisize=roisize)
            psfmodels = [diffspline_psfmodel, smap_psfmodel_lower, smap_psfmodel_higher]
        elif csplines == 1:
            smap_depth = depth + (subdepth > delta / 2) * delta
            smap_psfmodel = load_psfmodel(f'{base_folder}/astig_smooth_sph{smap_depth:04d}_3Dcorr.mat',
                                                roisize=roisize)
            psfmodels = [diffspline_psfmodel, smap_psfmodel]
        else:
            raise ValueError()

        ## Find the metrics

        errors, crlbs, estims = calculate_errors_with_crlb(psfmodels, base_folder+'/subdeltas',
                                                           subdepth, I, bg, labels=[None for _ in range(len(psfmodels))],
                                                           return_estims=True, plot_estimations=False, damp=damp,
                                                           iterations=iterations, n_samples=n_samples,
                                                           roiposs=[(0, 0) for _ in range(len(psfmodels))])
        
        for i, error in enumerate(errors):
            res_errors[i].append((error[:, 2, :]) * translations[2])  # then derrors becomes num_evals,101,100
            zrange = depth + np.linspace(*diffspline_psfmodel.calib.zrange,
                                         num=diffspline_psfmodel.calib.n_voxels_z + 1) * translations[2]
    
    for i, error in enumerate(res_errors):
        plt.plot(subinterval_depths, [elem.mean() for elem in error], label=splinetypes[i])
        plt.fill_between(subinterval_depths, [elem.mean() - elem.std() for elem in error], [elem.mean() + elem.std() for elem in error], alpha=alphas[i])
    # sns.violinplot(df, x='bead distance, nm', y='errors, nm', hue='Depth')
    # for i in range(len(biases)):
    #    plt.errorbar(biases[i].keys(), np.array(list(biases[i].values())), yerr=np.array(list(precs[i].values())), label=f'depth = {depths[i]} nm')
    # plt.title('Diffspline bias and precision over varying bead distances measured at different depths')
    # plt.xlabel('distance between beads, nm')
    # plt.ylabel('average bias +- precision, nm')
    plt.legend()
    plt.savefig(f'beaddistance{savefig_name}.svg')
    plt.show()

    return res_errors


def plot_nonuniform_beads_estimations(calibration_depths: np.array, base_folder,  load=False, ROIsize = 34, shifted_ROIsize = 56, Irange=(5000.0, 15000.0),
                                      pixelsize=100, n_realizations=10, translations=[100,100,1000], I=10000, bg=50, roisize=34,
                                     damp=10, iterations=100, n_samples=100, device='cuda', spherical_type='improved', savefig_name=''):
    '''
    Best to make base_folder a subfolder
    '''
    from service_functions.calibration import calibrate_diffspline_at_depth, process_with_SMAP_pipeline
    
    if not load:
        # Generate GT and averaged aligned PSFs 
        for depth in calibration_depths:
            generate_base_PSF(depth, base_folder, ROIsize=ROIsize, spherical_type=spherical_type)
        
        generate_shifted_PSFs(calibration_depths, base_folder, Irange=Irange, bg=bg, pixelsize=pixelsize,
                              n_realizations=n_realizations, ROIsize=shifted_ROIsize, spherical_type=spherical_type)
        
        process_with_SMAP_pipeline(calibration_depths, n_realizations=n_realizations, base_folder=base_folder)
    
    calibration_depths = np.floor(calibration_depths).astype(int)  # shorten for diffspline interpolaion
    
    # Load calibration PSFs
    calibration_psfs, _ = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?',
                                                              specific=calibration_depths)

    # Find evaluation depths
    bead_distances = calibration_depths[1:] - calibration_depths[:-1]
    n_eval_depths = np.clip(np.floor(bead_distances / 100).astype(int)[1:-1], a_min=None, a_max=5)
    shortened_calibration_depths = calibration_depths[1:-1]
    eval_depths = np.hstack([np.linspace(shortened_calibration_depths[i], shortened_calibration_depths[i + 1],
                                         n_eval_depths[i] + 2)[:-1] for i in range(len(n_eval_depths))])
    eval_depths = np.floor(eval_depths).astype(int)
    cspline_nearest_depth = np.array([min(calibration_depths, key=lambda calib_depth: abs(eval_depth - calib_depth)) for eval_depth in eval_depths])
    
    splinetypes = ['diffspline'] + [f'cspline at {d} nm' for d in shortened_calibration_depths]
    cmap = plt.colormaps['brg'](np.linspace(0, 1, num=len(shortened_calibration_depths)))
    alphas = [0.75] + [0.2 for _ in range(len(shortened_calibration_depths))]
    res_errors = [[] for _ in range(len(shortened_calibration_depths) + 1)]
    res_estims = [[] for _ in range(len(shortened_calibration_depths) + 1)]
    
    cspline_psfmodels = [load_psfmodel(f'{base_folder}/astig_smooth_sph{int(cspl_depth):04d}_3Dcorr.mat', roisize=roisize) for cspl_depth in shortened_calibration_depths]
    for j, eval_depth in enumerate(eval_depths):
        print(f'Current eval depth: {eval_depth}')
        # Take the closest cspline calibration to the current evaluation depth
        #cspline_psfmodel = load_psfmodel(f'{base_folder}/astig_smooth_sph{int(cspline_nearest_depth[j]):04d}_3Dcorr.mat', roisize=roisize)

        # Calibrate diffspline 
        calibrate_diffspline_at_depth(eval_depth, calibration_psfs, calibration_depths, base_folder+'/subdeltas',
                                          median_padding=False, device=device)
        
        diffspline_psfmodel = load_psfmodel(f'{base_folder}/subdeltas/astig_PSF_diffspline_dp{eval_depth:04d}.mat', roisize=roisize)
        
        generate_base_PSF(eval_depth, base_folder, extra_folder='subdeltas/')
        psfmodels = [diffspline_psfmodel] + cspline_psfmodels
        ## Find the metrics
        errors, crlbs, estims = calculate_errors_with_crlb(psfmodels, base_folder+'/subdeltas',
                                                           eval_depth, I, bg, labels=[None for _ in range(len(psfmodels))],
                                                           return_estims=True, plot_estimations=False, damp=damp,
                                                           iterations=iterations, n_samples=n_samples,
                                                           roiposs=[(0, 0) for _ in range(len(psfmodels))])
        for i, error in enumerate(errors):
            res_errors[i].append(abs(error[:, 2, :]) * translations[2])  # then derrors becomes num_evals,101,100
            res_estims[i].append(abs(estims[i][:, 2, :]) * translations[2])
            zrange = eval_depth + np.linspace(*diffspline_psfmodel.calib.zrange,
                                         num=diffspline_psfmodel.calib.n_voxels_z + 1) * translations[2]
    for i, error in enumerate(res_errors):
        plt.plot(eval_depths, [np.nanmean(abs(elem)) for elem in error], label=splinetypes[i], color=cmap[i-1] if i!=0 else 'black')
        plt.fill_between(eval_depths, [np.nanmean(abs(elem)) - np.nanmean(np.nanstd(elem, axis=-1)) for elem in error], [np.nanmean(abs(elem)) + np.nanmean(np.nanstd(elem, axis=-1)) for elem in error], alpha=alphas[i], color=cmap[i-1] if i!=0 else 'black')
    
    # plt.title('Diffspline bias and precision over varying bead distances measured at different depths')
    # plt.xlabel('distance between beads, nm')
    # plt.ylabel('average bias +- precision, nm')
    plt.xlabel('depth, nm')
    plt.ylabel('bias +- std in z, nm')
    plt.legend()
    plt.savefig(f'nonuniform{savefig_name}.svg')
    plt.show()

    return res_errors, res_estims


def plot_nonuniform_beads_estimations_vs_coverslip(calibration_depths: np.array, base_folder,  load=False, ROIsize = 34, shifted_ROIsize = 56, Irange=(5000.0, 15000.0),
                                      pixelsize=100, n_realizations=10, translations=[100,100,1000], I=10000, bg=50, roisize=34,
                                     damp=10, iterations=100, n_samples=100, device='cuda', spherical_type='improved', label_addition='on non-uniform depths', savefig_name=''):
    '''
    Best to make base_folder a subfolder
    '''
    from service_functions.calibration import calibrate_diffspline_at_depth, process_with_SMAP_pipeline
    
    if not load:
        # Process cspline
        generate_base_PSF(0, base_folder, ROIsize=ROIsize, spherical_type=spherical_type)
        generate_shifted_PSFs([0], base_folder, Irange=Irange, bg=bg, pixelsize=pixelsize,
                              n_realizations=n_realizations, ROIsize=shifted_ROIsize, spherical_type=spherical_type)
        process_with_SMAP_pipeline([0], n_realizations=n_realizations, base_folder=base_folder)        

        # Generate GT and averaged aligned PSFs 
        for depth in calibration_depths:
            generate_base_PSF(depth, base_folder, ROIsize=ROIsize, spherical_type=spherical_type)
        
        generate_shifted_PSFs(calibration_depths, base_folder, Irange=Irange, bg=bg, pixelsize=pixelsize,
                              n_realizations=n_realizations, ROIsize=shifted_ROIsize, spherical_type=spherical_type)
        
        process_with_SMAP_pipeline(calibration_depths, n_realizations=n_realizations, base_folder=base_folder)
    
    calibration_depths = np.floor(calibration_depths).astype(int)  # shorten for diffspline interpolaion
    
    # Load calibration PSFs
    calibration_psfs, _ = get_smoothpsf_data(micron=None, base_folder=base_folder, nm_hundreds='?',
                                                              specific=calibration_depths)

    # Find evaluation depths
    bead_distances = calibration_depths[1:] - calibration_depths[:-1]
    n_eval_depths = np.clip(np.floor(bead_distances / 100).astype(int)[1:-1], a_min=None, a_max=5)
    shortened_calibration_depths = calibration_depths[1:-1]
    eval_depths = np.hstack([np.linspace(shortened_calibration_depths[i], shortened_calibration_depths[i + 1],
                                         n_eval_depths[i] + 2)[:-1] for i in range(len(n_eval_depths))])
    eval_depths = np.floor(eval_depths).astype(int)
    
    splinetypes = [f'diffspline {label_addition}', 'cspline on coverslip']
    #cmap = plt.colormaps['brg'](np.linspace(0, 1, num=len(shortened_calibration_depths)))
    colors = ['r', 'g']
    alphas = [0.2, 0.2]
    res_errors = [[], []]
    res_estims = [[], []]
    
    cspline_psfmodels = [load_psfmodel(f'{base_folder}/astig_smooth_sph{0:04d}_3Dcorr.mat', roisize=roisize)]
    
    plt.figure(figsize=(7,4))
    for j, eval_depth in enumerate(eval_depths):
        print(f'Current eval depth: {eval_depth}')
        # Take the closest cspline calibration to the current evaluation depth
        #cspline_psfmodel = load_psfmodel(f'{base_folder}/astig_smooth_sph{int(cspline_nearest_depth[j]):04d}_3Dcorr.mat', roisize=roisize)

        # Calibrate diffspline 
        calibrate_diffspline_at_depth(eval_depth, calibration_psfs, calibration_depths, base_folder+'/subdeltas',
                                          median_padding=False, device=device)
        
        diffspline_psfmodel = load_psfmodel(f'{base_folder}/subdeltas/astig_PSF_diffspline_dp{eval_depth:04d}.mat', roisize=roisize)
        
        generate_base_PSF(eval_depth, base_folder, extra_folder='subdeltas/')
        psfmodels = [diffspline_psfmodel] + cspline_psfmodels
        ## Find the metrics
        errors, crlbs, estims = calculate_errors_with_crlb(psfmodels, base_folder+'/subdeltas',
                                                           eval_depth, I, bg, labels=[None for _ in range(len(psfmodels))],
                                                           return_estims=True, plot_estimations=False, damp=damp,
                                                           iterations=iterations, n_samples=n_samples,
                                                           roiposs=[(0, 0) for _ in range(len(psfmodels))])
        for i, error in enumerate(errors):
            res_errors[i].append(error[:, 2, :] * translations[2])  # then derrors becomes num_evals,101,100
            res_estims[i].append(estims[i][:, 2, :] * translations[2])
            zrange = eval_depth + np.linspace(*diffspline_psfmodel.calib.zrange,
                                         num=diffspline_psfmodel.calib.n_voxels_z + 1) * translations[2]
    for i, error in enumerate(res_errors):
        plt.plot(eval_depths, [np.nanmean(abs(elem)) for elem in error], label=splinetypes[i], color=colors[i])
        plt.fill_between(eval_depths, [np.nanmean(abs(elem)) - np.nanmean(np.nanstd(elem, axis=-1)) for elem in error], [np.nanmean(abs(elem)) + np.nanmean(np.nanstd(elem, axis=-1)) for elem in error], alpha=alphas[i], color=colors[i])
    
    # plt.title('Diffspline bias and precision over varying bead distances measured at different depths')
    plt.xlabel('depth, nm')
    plt.ylabel('bias +- std in z, nm')
    plt.legend()
    plt.savefig(f'nonuniform{savefig_name}.svg')
    plt.show()

    return res_errors, res_estims