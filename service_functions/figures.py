import matplotlib.pyplot as plt
import numpy as np
from fastpsf.cspline import CSplinePSF
from service_functions.loading import generate_params



def make_psf_figure_old(psfmodels: list, I=15000, bg=50, difference_base=None,
                    labels=['Diffspline', 'Cspline', 'Difference'],
                    plt_range=None, roichange_psfmodel=(None, None)):
    '''
    Generate 5 rows (z=-500, 0, 500, ZX and ZY views) and 3 rows (psfmodel1, psfmodel2, difference)
    difference_base - index of "ground truth" PSF among psfmodels, don't plot difference if None
    '''
    psfs = []
    for psfmodel in psfmodels:
        # Generate images of the PSFs
        if type(psfmodel) == CSplinePSF:
            psf = psfmodel.ExpectedValue(generate_params(psfmodel, I=I, bg=bg))[:, roichange_psfmodel[0]:roichange_psfmodel[1], roichange_psfmodel[0]:roichange_psfmodel[1]]
        else:
            psf = (I*psfmodel + bg)[:, roichange_psfmodel[0]:roichange_psfmodel[1], roichange_psfmodel[0]:roichange_psfmodel[1]]
        psfs += [psf]
    if difference_base is not None:
        diffs = [100*(psfs[i] - psfs[difference_base])/psfs[difference_base] for i in range(len(psfs)) if i != difference_base]
    
    
    imgs = psfs + diffs

    nplanes = psfs[0].shape[0]
    zval_min = int(psfmodels[0].calib.z_min*1000)
    zval_max = int(psfmodels[0].calib.z_max*1000)

    if plt_range == 'auto':
        plt_min = np.min(psfs)
        plt_max = np.max(psfs)
    elif plt_range is None:
        plt_min, plt_max = (None, None)
    else:
        plt_min, plt_max = plt_range

    # Plot PSFs
    plt.figure(figsize=(15, 12))
    rows = 3
    cols = len(imgs) + 2
    for row in range(rows):
        for col in range(cols):
            if col in [len(psfs), len(imgs)]:
                plt.colorbar()
                continue
            plt.subplot2grid((rows, cols), (row, col))
            img = imgs[col][row*(nplanes//2)]  # 0, center, or nplanes z-positions
            if col > 2: 
                plt.imshow(img, vmin=None, vmax=None)
            else:
                plt.imshow(img, vmin=plt_min, vmax=plt_max)
            if row == 0:
                plt.title(labels[col], fontsize=15)
            if col == 0:
                plt.ylabel(f'Z = {[zval_min, 0, zval_max][row]} nm', fontsize=15)
            

    plt.figure(figsize=(15, 12))
    rows = 2
    cols = len(imgs)
    # Plot last row separately
    for row in range(rows):
        for col in range(cols):
            plt.subplot2grid((rows, cols), (row, col), colspan=1)
            img = imgs[col].sum(row-rows)  # -2 or -1
            plt.imshow(img, vmin=plt_min, vmax=plt_max, interpolation='nearest')
            if col == 0:
                plt.ylabel(["ZY view", "ZX view"][row], fontsize=15)
            plt.colorbar()

    # plt.legend()
    plt.show()
    



def make_psf_figure(psfmodels: list, I=15000, bg=50, roiposs=[(0,0),(0,0),(0,0)], difference_base=None,
                labels=['Diffspline', 'Cspline', 'Difference'], figsize1=(15, 15), figsize2=(15, 15), usetex=False,
                plt_range=None, roichange_psfmodel=(None, None), savefig_name=''):
    '''
    Generate 5 rows (z=-500, 0, 500, ZX and ZY views) and 3 rows (psfmodel1, psfmodel2, difference)
    difference_base - index of "ground truth" PSF among psfmodels, don't plot difference if None
    '''
    plt.rc('text', usetex=usetex)
    #plt.rc('font', size=12)

    psfs = []
    for i, psfmodel in enumerate(psfmodels):
        # Generate images of the PSFs
        if type(psfmodel) != np.ndarray:
            psf = psfmodel.ExpectedValue(generate_params(psfmodel, I=I, bg=bg, roipos=roiposs[i]))[:, roichange_psfmodel[0]:roichange_psfmodel[1], roichange_psfmodel[0]:roichange_psfmodel[1]]
        else:
            psf = (I*psfmodel + bg)[:, roichange_psfmodel[0]:roichange_psfmodel[1], roichange_psfmodel[0]:roichange_psfmodel[1]]
        psfs += [psf]
    if difference_base is not None:
        diffs = [100*(psfs[i] - psfs[difference_base])/psfs[difference_base] for i in range(len(psfs)) if i != difference_base]
    
    imgs = psfs + diffs

    zyx_psfs = np.array([[psf.sum(ind) for psf in psfs] for ind in [-2,-1]])  # size=(2,3,101,34)
    zyx_diffs = []
    for zyx_row in zyx_psfs:
        zyx_diffs.append([100*(zyx_row[i] - zyx_row[difference_base])/zyx_row[difference_base] for i in range(len(zyx_row)) if i != difference_base])
    zyx_imgs = np.hstack([zyx_psfs, zyx_diffs])

    non_differencebase_ind = (difference_base + 1)%len(psfs)
    nplanes = psfs[non_differencebase_ind].shape[0]
    zval_min = int(psfmodels[non_differencebase_ind].calib.z_min*1000)
    zval_max = int(psfmodels[non_differencebase_ind].calib.z_max*1000)

    if plt_range == 'auto':
        plt_min_pixels = np.min(psfs)
        plt_max_pixels = np.max(psfs)
        plt_min_diff = np.min(diffs)
        plt_max_diff = np.max(diffs)
    
        plt_min_zyxpixels = np.min(zyx_imgs)
        plt_max_zyxpixels = np.max(zyx_imgs)
        plt_min_zyxdiff = np.min(zyx_diffs)
        plt_max_zyxdiff = np.max(zyx_diffs)
    elif plt_range is None:
        plt_min_pixels, plt_max_pixels = (None, None)
        plt_min_diff, plt_max_diff = (None, None)
        plt_min_zyxpixels, plt_max_zyxpixels = (None, None)
        plt_min_zyxdiff, plt_max_zyxdiff = (None, None)
    else:
        plt_min_pixels, plt_max_pixels = plt_range[0]
        plt_min_diff, plt_max_diff = plt_range[1]
        plt_min_zyxpixels, plt_max_zyxpixels = plt_range[2]
        plt_min_zyxdiff, plt_max_zyxdiff = plt_range[3]

    fig, axs = plt.subplots(3, len(imgs), figsize=figsize1, layout="constrained")
    for row in range(len(axs)):
        for col in range(len(axs[row])):
            img = imgs[col][row*(nplanes//2)]  # 0, center, or nplanes z-positions
            print(f'row={row*(nplanes//2)}, col={col}\tmin={np.min(img)}, max={np.max(img)}, avg={np.mean(abs(img))}')
            if col < 3:
                vmin,vmax = (plt_min_pixels,plt_max_pixels)
                im_pixels = axs[row, col].imshow(img, vmin=vmin, vmax=vmax)
            else:
                vmin,vmax = (plt_min_diff, plt_max_diff)
                im_diff = axs[row, col].imshow(img, vmin=vmin, vmax=vmax, interpolation='nearest')
        
            if col == 0:
                axs[row, col].set_ylabel(f'Z = {[zval_min, 0, zval_max][row]} nm'+'\n\n X axis, pixels', fontsize=9)
            if row == len(axs)-1:
                axs[row, col].set_xlabel('{Y axis, pixels}', fontsize=9)
        
            if row == 0:
                axs[row, col].set_title('{'+labels[col]+'}', fontsize=12)

    
    cbar = fig.colorbar(im_pixels, ax=axs[:, 2], pad=0.2)
    cbar.ax.set_xlabel('intensity,\nphotons', fontsize=9)  #, rotation=270)
    cbar = fig.colorbar(im_diff, ax=axs[:, -1], pad=0.3)
    cbar.ax.set_xlabel('relative difference,\%', fontsize=9)  #, rotation=270)

    plt.savefig(f'visual_xy{savefig_name}.svg')
    plt.show()

    fig, axs = plt.subplots(1, 2*len(imgs), figsize=figsize2, layout="constrained")
    for col in range(len(imgs)):
        for subcol in range(2):

            img = zyx_imgs[subcol][col]
            print(f'subcol={subcol}, col={col}\tmin={np.min(img)}, max={np.max(img)}, avg={np.mean(abs(img))}')
            if col < 3:
                vmin,vmax = (plt_min_zyxpixels,plt_max_zyxpixels)
                im_pixels = axs[2*col+subcol].imshow(img, vmin=vmin, vmax=vmax)
            else:
                vmin,vmax = (plt_min_zyxdiff,plt_max_zyxdiff)
                im_diff = axs[2*col+subcol].imshow(img, vmin=vmin, vmax=vmax, interpolation='nearest')
            axs[2*col+subcol].set_yticks((0, nplanes//2, nplanes-1))
            axs[2*col+subcol].set_yticklabels((zval_min, 0, zval_max))
            if col == 0 and subcol == 0:
                axs[2*col+subcol].set_ylabel("Z axis, nm", fontsize=9)
            
            axs[2*col+subcol].set_xlabel('{'+f'{["Y", "X"][subcol]}'+' axis, pixels}', fontsize=9)

    
    cbar = fig.colorbar(im_pixels, ax=axs[2*2+1], pad=0.2)
    cbar.ax.set_xlabel('intensity,\nphotons', fontsize=9)#, rotation=270)
    cbar = fig.colorbar(im_diff, ax=axs[-1], pad=0.2)
    cbar.ax.set_xlabel('relative difference,\%', fontsize=9)#, rotation=270)

    plt.savefig(f'visual_zxy{savefig_name}.svg')
    plt.show()
    plt.rc('text', usetex=False)
    return imgs