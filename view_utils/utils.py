from IPython.display import clear_output
import matplotlib.pyplot as plt
from time import sleep
import numpy as np


def show_psf(psf, zrange=[], titles=[], pause=0.01, plt_range=None):
    """
    A function for jupyter notebook to show one or several PSFs in a single row

    """
    if type(psf) != list:
        psf = [psf]
    l = len(psf)

    if plt_range == 'auto':
        search = np.hstack([elem.ravel() for elem in psf])
        plt_min = np.min(search)
        plt_max = np.max(search)
    elif plt_range is None:
        plt_min, plt_max = (None, None)
    else:
        plt_min, plt_max = plt_range

    for j in range(min([psf[i].shape[0] for i in range(l)])):
        plt.figure(figsize=(15, 5))

        for i in range(l):
            plt.subplot2grid((1, l), (0, i))
            plt.imshow(psf[i][j], vmin=plt_min, vmax=plt_max)
            if len(titles) != 0 or len(zrange) != 0:
                title = titles[i] if len(titles) != 0 else 'Z-stack'
                extra = f'at z={zrange[i][j]}' if len(zrange) != 0 else ''
                plt.title(f'{title} {extra}')

            plt.colorbar()

        # plt.legend()
        plt.show()

        sleep(pause)
        clear_output(True)


def make_psf_figure_old(psfs: list, zrange, maxdist=400, plt_range=None, labels=["Ground truth", "SMAP", "Diffspline"]):
    """
    psfs - PSF arrays of the same shape (z_size, roisize, roisize) (gt, smap, diffspline)
    zrange - explains to which position the PSF correspond to
    plt_range - range of values on the images, 'auto' to pick min/max across all given images
    """
    check = [psfs[i].shape == psfs[i + 1].shape for i in range(len(psfs) - 1) if len(psfs) > 1]
    assert sum(check) == len(check)

    if plt_range == 'auto':
        plt_min = np.min(psfs)
        plt_max = np.max(psfs)
    elif plt_range is None:
        plt_min, plt_max = (None, None)
    else:
        plt_min, plt_max = plt_range

    cols = 3
    rows = len(psfs)

    inds = [np.where(zrange == -maxdist)[0][0], np.where(zrange == 0)[0][0], np.where(zrange == maxdist)[0][0]]

    plt.figure(figsize=(15, 12))

    for row in range(rows):
        for col in range(cols):
            plt.subplot2grid((rows, cols), (row, col))
            plt.imshow(psfs[row][inds[col]], vmin=plt_min, vmax=plt_max)
            if row == 0:
                plt.title(f'Z = {[-maxdist, 0, maxdist][col]} nm', fontsize=15)
            if col == 0:
                plt.ylabel(labels[row], fontsize=15)

            plt.colorbar()

    # plt.legend()
    plt.show()


def chi_squared(observed, expected):
    return (observed - expected) ** 2 / expected


def get_stats_along_z(psfs: list, zrange, figsize=(20, 4),
                      labels=['$\chi^{2}$(GT, SMAP)', '$\chi^{2}$(GT, Diffspline)']):
    assert type(psfs) == list

    rows = len(psfs)

    plt.figure(figsize=figsize)
    for row in range(rows):
        plt.subplot2grid((rows, 2), (row, 0))
        plt.plot(zrange, psfs[row].sum(-1).sum(-1))
        if row == rows - 1:  # only label the last row
            plt.xlabel('z position (nm)')
        plt.ylabel(labels[row])
        if row == 0:
            plt.title('Cumulative')

        plt.subplot2grid((rows, 2), (row, 1))
        plt.plot(zrange, psfs[row].max(-1).max(-1))
        if row == rows - 1:
            plt.xlabel('z position (nm)')
        if row == 0:
            plt.title('Maximum')
    plt.show()

    plt.figure(figsize=figsize)

    plt.subplot2grid((1, 2), (0, 0))
    for row in range(rows):
        plt.plot(zrange, psfs[row].sum(-1).sum(-1), label=labels[row])
    plt.xlabel('z position (nm)')
    plt.ylabel(labels[row])
    plt.title('Cumulative')
    plt.legend()

    plt.subplot2grid((1, 2), (0, 1))
    for row in range(rows):
        plt.plot(zrange, psfs[row].max(-1).max(-1), label=labels[row])
    plt.xlabel('z position (nm)')
    plt.title('Maximum')
    plt.legend()
    plt.show()



def get_stats_along(psfs: list, zranges, figsize=(20, 4),
                    labels=['$\chi^{2}$(GT, SMAP)', '$\chi^{2}$(GT, Diffspline)']):
    assert type(psfs) == list

    rows = len(psfs)

    plt.figure(figsize=figsize)

    plt.subplot2grid((1, 2), (0, 0))
    for row in range(rows):
        plt.plot(zranges[row], psfs[row].sum(-1).sum(-1), label=labels[row])
    plt.xlabel('z position (nm)')
    plt.ylabel(labels[row])
    plt.title('Cumulative')
    plt.legend()

    plt.subplot2grid((1, 2), (0, 1))
    for row in range(rows):
        plt.plot(zranges[row], psfs[row].max(-1).max(-1), label=labels[row])
    plt.xlabel('z position (nm)')
    plt.title('Maximum')
    plt.legend()
    plt.show()



def make_figures(true_psf, zstack, init_psf, res, zrange, plt_range=None, labels=[['Ground truth', 'Ground truth image', 'Initial diffspline', 'Trained diffspline'], ['$\chi^{2}$(GT, Initial diffspline)', '$\chi^{2}$(GT, Trained diffspline)'], ['$\chi^{2}$(GT, Initial diffspline)', '$\chi^{2}$(GT, Trained diffspline)']]):
    # figures
    make_psf_figure([true_psf, zstack.mean(0), init_psf.mean(0), res.mean(0)], zrange, maxdist=zrange[-1], plt_range=plt_range, labels=labels[0])

    chi_init = chi_squared(zstack, init_psf).mean(0)
    chi_trained = chi_squared(zstack, res).mean(0)

    make_psf_figure([chi_init, chi_trained], zrange, maxdist=zrange[-1], plt_range=plt_range, labels=labels[1])

    get_stats_along_z([chi_init, chi_trained], zrange, labels=labels[2])

