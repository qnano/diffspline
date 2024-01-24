# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 23:20:06 2021

@author: jelmer
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import math
import time
from typing import List, Tuple, Union, Optional, Final
from torch import Tensor

#from smlmtorch.crlb import poisson_crlb, poisson_crlb_select_axes
#from smlmtorch.levmar import LM_MLE

import ctypes
import numpy as np
import numpy.ctypeslib as ctl
import os,yaml

#from smlmtorch.grad_descent import BatchAdaptiveGradientDescent, PoissonLikelihoodLoss

Theta = ctypes.c_float * 4
FisherMatrix = ctypes.c_float * 16



class GaussAstigPSFCalib:
    def __init__(self, x=[1.3,2,3,0], y=[1.3,-2,3,0], zrange=[-3, 3]):
        self.x = x
        self.y = y
        self.zrange = zrange

    @property
    def params(self):
        return np.array([self.x,self.y])

    @classmethod
    def from_file(cls, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            calibration = np.load(filename, allow_pickle=True).item()
        else:
            with open(filename, "r") as f:
                calibration = yaml.safe_load(f)
        return cls(calibration.get("x"), calibration.get("y"))

    def save(self, filename):
        ext = os.path.splitext(filename)[1]
        if ext == '.npy':
            np.save(filename, self.__dict__())
        elif ext == '.yaml':
            with open(filename, "w") as f:
                yaml.dump(self.__dict__(),f)

@torch.jit.script
def gauss_psf_2D(theta, numpixels: int):
    """
    theta: [x,y,N,bg,sigma_x, sigma_y].T
    """
    pi = 3.141592653589793  # torch needs to include pi
    
    x,y,N,bg,sigma_x, sigma_y = theta[:,0], theta[:,1], theta[:,2], theta[:,3], theta[:,4], theta[:,5]
    pixelpos = torch.arange(0, numpixels, device=theta.device)

    OneOverSqrt2PiSigmaX = (1.0 / (torch.sqrt(2 * pi) * sigma_x))[:,None,None]
    OneOverSqrt2SigmaX = (1.0 / (torch.sqrt(2) * sigma_x))[:,None,None]
    OneOverSqrt2PiSigmaY = (1.0 / (torch.sqrt(2 * pi) * sigma_y))[:,None,None]
    OneOverSqrt2SigmaY = (1.0 / (torch.sqrt(2) * sigma_y))[:,None,None]

    # Pixel centers
    Xc = pixelpos[None,None,:]
    Yc = pixelpos[None,:,None]
    Xexp0 = (Xc-x[:,None,None]+0.5) * OneOverSqrt2SigmaX
    Xexp1 = (Xc-x[:,None,None]-0.5) * OneOverSqrt2SigmaX
    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigmaX * (torch.exp(-Xexp1**2) - torch.exp(-Xexp0**2))
    dEx_dSigma = (torch.exp(-Xexp1**2) * Xexp1 - torch.exp(-Xexp0**2)* Xexp0) / torch.sqrt(pi)

    Yexp0 = (Yc-y[:,None,None]+0.5) * OneOverSqrt2SigmaY
    Yexp1 = (Yc-y[:,None,None]-0.5) * OneOverSqrt2SigmaY
    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigmaY * (torch.exp(-Yexp1**2) - torch.exp(-Yexp0**2))
    dEy_dSigma = (torch.exp(-Yexp1**2) * Yexp1 - torch.exp(-Yexp0**2)* Yexp0) / torch.sqrt(pi)
    
    mu = N[:,None,None] * Ex*Ey + bg[:,None,None]
    dmu_x = N[:,None,None] * Ey * dEx
    dmu_y = N[:,None,None] * Ex * dEy    
    dmu_I = Ex*Ey
    dmu_bg = 1 + mu*0
    dmu_sigma = N[:,None,None] * ( Ex * dEy_dSigma + dEx_dSigma * Ey )
    
    deriv = torch.stack((dmu_x, dmu_y,dmu_I, dmu_bg, dmu_sigma),-1)
    return mu,  deriv

@torch.jit.script
def gauss_psf_2D_fixed_sigma(theta, roisize: int, sigma_x: float, sigma_y: float):
    
    sigma_ = torch.ones((len(theta),2), device=theta.device) * torch.tensor([sigma_x,sigma_y],device=theta.device)[None]
    theta_ = torch.cat((theta, sigma_),-1)
    
    mu, jac = gauss_psf_2D(theta_, roisize)
    return mu, jac[...,:-1]

class Gaussian2DFixedSigmaPSF(torch.nn.Module):
    def __init__(self, roisize, sigma):
        super().__init__()
        self.roisize = roisize
        if type(sigma) == float:
            sigma = [sigma, sigma]
        self.sigma = sigma

    def forward(self, x, const_:Optional[Tensor]=None):
        return gauss_psf_2D_fixed_sigma(x, self.roisize, self.sigma[0], self.sigma[1])


@torch.jit.script
def gauss_psf_2D_astig(theta: Tensor, numpixels:int, calib: Tensor):
    tx = theta[:,0,None,None]
    ty = theta[:,1,None,None]
    tz = theta[:,2,None,None]
    tI = theta[:,3,None,None]
    tbg = theta[:,4,None,None]
    
    sqrt2pi = 2.5066282746310002
    sqrt2 = 1.4142135623730951

    # ugly but torch script is complaining    
    s0_x, gamma_x, d_x, A_x = calib[0,0], calib[0,1], calib[0,2], calib[0,3]
    s0_y, gamma_y, d_y, A_y = calib[1,0], calib[1,1], calib[1,2], calib[1,3]
        
    tzx = tz - gamma_x
    tzy = tz - gamma_y
    sigma_x = s0_x * torch.sqrt(1 + tzx**2 / d_x**2 + A_x * tzx**3 / d_x**3)
    sigma_y = s0_y * torch.sqrt(1 + tzy**2 / d_y**2 + A_y * tzy**3 / d_y**3)
    
    OneOverSqrt2PiSigma_x = 1 / (sqrt2pi * sigma_x)
    OneOverSqrt2Sigma_x = 1 / (sqrt2  * sigma_x)
    OneOverSqrt2PiSigma_y = 1 / (sqrt2pi * sigma_y)
    OneOverSqrt2Sigma_y = 1 / (sqrt2  * sigma_y)
        
    pixelpos = torch.arange(0, numpixels,device=theta.device)
    Xc = pixelpos[None,None,:]
    Yc = pixelpos[None,:,None]

    # Pixel centers
    Xexp0 = (Xc-tx+0.5) * OneOverSqrt2Sigma_x
    Xexp1 = (Xc-tx-0.5) * OneOverSqrt2Sigma_x 
    Ex = 0.5 * torch.erf(Xexp0) - 0.5 * torch.erf(Xexp1)
    dEx = OneOverSqrt2PiSigma_x * (torch.exp(-Xexp1**2) - torch.exp(-Xexp0**2))
        
    Yexp0 = (Yc-ty+0.5) * OneOverSqrt2Sigma_y
    Yexp1 = (Yc-ty-0.5) * OneOverSqrt2Sigma_y
    Ey = 0.5 * torch.erf(Yexp0) - 0.5 * torch.erf(Yexp1)
    dEy = OneOverSqrt2PiSigma_y * (torch.exp(-Yexp1**2) - torch.exp(-Yexp0**2))

    G21y = 1 / (sqrt2pi * sigma_y * sigma_y) * (
        (Yc - ty - 0.5) * torch.exp(-(Yc - ty - 0.5)*(Yc - ty - 0.5) / (2.0 * sigma_y * sigma_y)) -
        (Yc - ty + 0.5) * torch.exp(-(Yc - ty + 0.5)*(Yc - ty + 0.5) / (2.0 * sigma_y * sigma_y)))

    mu = tbg + tI * Ex * Ey
    dmu_dx = tI * dEx * Ey
    dmu_dy = tI * dEy * Ex
    
    G21x = 1 / (sqrt2pi * sigma_x * sigma_x) * (
        (Xc - tx - 0.5) * torch.exp(-(Xc - tx - 0.5)*(Xc - tx - 0.5) / (2 * sigma_x * sigma_x)) -
        (Xc - tx + 0.5) * torch.exp(-(Xc - tx + 0.5)*(Xc - tx + 0.5) / (2 * sigma_x * sigma_x)))
    
    dMuSigmaX = tI * Ey * G21x
    dMuSigmaY = tI * Ex * G21y
    
    dSigmaXThetaZ = (s0_x * (2 * tzx / d_x**2 + A_x * 3 * tzx**2 / d_x**2) /
        (2 * torch.sqrt(1 + tzx**2 / d_x**2 + A_x * tzx**3 / d_x**3)))
    dSigmaYThetaZ = (s0_y * (2 * tzy / d_y**2 + A_y * 3 * tzy**2 / d_y**2) /
        (2 * torch.sqrt(1 + tzy**2 / d_y**2 + A_y * tzy**3 / d_y**3)))
    
    dmu_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ
    
    dmu_dI0 = Ex * Ey
    dmu_dIbg = dmu_dx*0 + 1

    return mu, torch.stack( ( dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg ), -1 )

class Gaussian2DAstigmaticPSF(torch.nn.Module):
    def __init__(self, roisize:int, calib):
        super().__init__()
        self.roisize = roisize
        self.calib = calib

    def forward(self,x, const_:Optional[Tensor]=None):
        return gauss_psf_2D_astig(x, self.roisize, self.calib)



        

def estimate_precision(x, photoncounts, phot_ix, psf_model, mle, plot_axes=None, 
                       axes_scaling=None, axes_units=None, skip_axes=[],
                       jit=True):
    crlb = []
    prec = []
    rmsd = []
    runtime = []
       
    if jit:
        mle = torch.jit.script(mle)
    
    for i, phot in enumerate(photoncounts):
        x_ = x*1
        x_[:,phot_ix] = phot
        mu, deriv = psf_model(x_)
        smp = torch.poisson(mu)
        
        initial = x_*(torch.rand(x_.shape,device=x.device) * 0.2 + 0.9) 
        #initial = x*1
        
        t0 = time.time()
        estim = mle(smp, initial)
                
        errors = x_ - estim

        t1 = time.time()
        runtime.append( len(x) / (t1-t0+1e-10)) # 
        
        prec.append(errors.std(0))
        rmsd.append(torch.sqrt((errors**2).mean(0)))
        
        crlb_i = poisson_crlb_select_axes(mu, deriv, skip_axes=skip_axes)
        crlb.append(crlb_i.mean(0))
        
    print(runtime)

    crlb = torch.stack(crlb).cpu()
    prec = torch.stack(prec).cpu()
    rmsd = torch.stack(rmsd).cpu()
        
    if plot_axes is not None:
        figs = []
        for i, ax_ix in enumerate(plot_axes):
            fig,ax = plt.subplots()
            figs.append(fig)
            ax.loglog(photoncounts, axes_scaling[i] * prec[:, ax_ix], label='Precision')
            ax.loglog(photoncounts, axes_scaling[i] * crlb[:, ax_ix],'--',  label='CRLB')
            ax.loglog(photoncounts, axes_scaling[i] * rmsd[:, ax_ix],':k', label='RMSD')
            ax.legend()
            ax.set_title(f'Estimation precision for axis {ax_ix}')
            ax.set_xlabel('Photon count [photons]')
            ax.set_ylabel(f'Precision [{axes_units[i]}]')
            
        return crlb, prec, rmsd, figs
        
    return crlb, prec, rmsd

def test_gauss_psf_2D():
    N = 2000
    roisize = 12
    sigma = 1.5
    thetas = np.zeros((N, 4))
    thetas[:,:2] = roisize/2 + np.random.uniform(-roisize/8,roisize/8, size=(N,2))
    thetas[:,2] = 1000#np.random.uniform(200, 2000, size=N)
    thetas[:,3] = np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')
    
    param_range = torch.Tensor([
        [0, roisize-1],
        [0, roisize-1],
        [1, 1e9],
        [1e-6, 1e6],
    ]).to(dev)

    thetas = torch.Tensor(thetas).to(dev)
    mu, jac = gauss_psf_2D_fixed_sigma(thetas, roisize, sigma)    
    
    smp = torch.poisson(mu)
    plt.figure()
    plt.imshow(smp[0].cpu().numpy())
    
    crlb = poisson_crlb(mu,jac)
    
    model = Gaussian2DFixedSigmaPSF(roisize, sigma)
    initial = thetas*(torch.rand(thetas.shape, device=dev) * 0.2 + 0.9) 
    mle = LM_MLE(model, lambda_=1e2, iterations=30, param_range_min_max=param_range)
    mle = torch.jit.script(mle)

    for i in range(10):
        t0 = time.time()
        mle(smp, initial, None) 
        t1 = time.time()
        print( N/(t1-t0))
    
    photoncounts = np.logspace(2, 4, 10)
    estimate_precision(thetas, photoncounts, phot_ix=2, psf_model=model, mle=mle,
                       plot_axes=[0,1,2],
                       axes_scaling=[100,100,1],
                       axes_units=['nm', 'nm', 'photons'])

    


#%%
if __name__ == '__main__':
    #test_gauss_psf_2D()
   

    gauss3D_calib = [
        [1.0,-0.12,0.2,0.1],
         [1.05,0.15,0.19,0]]
    

    np.random.seed(0)

    N = 500
    roisize = 9
    thetas = np.zeros((N, 5))
    thetas[:,:2] = roisize/2 + np.random.uniform(-roisize/8,roisize/8, size=(N,2))
    thetas[:,2] = np.random.uniform(-0.3,0.3,size=N)
    thetas[:,3] = 1000#np.random.uniform(200, 2000, size=N)
    thetas[:,4] = 50# np.random.uniform(1, 10, size=N)

    dev = torch.device('cuda')
    
    param_range = torch.Tensor([
        [0, roisize-1],
        [0, roisize-1],
        [-1, 1],
        [1, 1e9],
        [1.0, 1e6],
    ]).to(dev)

    gauss3D_calib = torch.tensor(gauss3D_calib).to(dev)
    thetas = torch.Tensor(thetas).to(dev)
    mu, jac = gauss_psf_2D_astig(thetas, roisize, gauss3D_calib)
    
    smp = torch.poisson(mu)
    plt.figure()
    plt.imshow(smp[0].cpu().numpy())
    
    #%%
    
    psf_model = Gaussian2DAstigmaticPSF(roisize, gauss3D_calib)
        
    lm_estimator = LM_MLE(psf_model, param_range, iterations=50, lambda_=1e2)
    
    loss = PoissonLikelihoodLoss(psf_model)
    loss = torch.jit.script(loss)
    gd_estimator = BatchAdaptiveGradientDescent(loss, param_range.T, 
                                                initial_step=0.1,
                                                param_step_factor=torch.tensor([1,1,10,1000000,1000], device=dev))

    def gd_estim(smp, initial):
        return gd_estimator.forward(smp,initial)[0]

    crlb = poisson_crlb(mu,jac)
    print(f"SMLM CRLB: {crlb.mean(0)} ")
    
    photoncounts = np.logspace(2, 4, 10)
    
 #   astig_crlb, astig_prec_lm, astig_rmsd_lm = estimate_precision(thetas, photoncounts, phot_ix=3, 
 #                      psf_model = psf_model, mle = gd_estimator, jit=False)

    _, astig_prec_gd, astig_rmsd_gd = estimate_precision(thetas, photoncounts, phot_ix=3, 
                        psf_model = psf_model, mle=gd_estim, jit=False)

    astig_crlb, astig_prec_lm, astig_rmsd_lm = estimate_precision(thetas, photoncounts, phot_ix=3, 
                       psf_model = psf_model,  mle=lm_estimator,jit=False)


#%%    
    plot_axes=[0,1,2,3,4]
    axes_scaling=[100,100,1000,1,1]
    axes_units=['nm', 'nm', 'nm', 'photons', 'ph/px']

    for i, ax_ix in enumerate(plot_axes):
        fig,ax = plt.subplots()
        ax.loglog(photoncounts, axes_scaling[i] * astig_crlb[:, ax_ix], '--g',  label='CRLB (Astig.)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_rmsd_lm[:, ax_ix], 'og', ms=4,label='RMSD (Astig, LevMar)')
        ax.loglog(photoncounts, axes_scaling[i] * astig_rmsd_gd[:, ax_ix], 'ok', ms=4,label='RMSD (Astig, GD)')
  #      ax.loglog(photoncounts, axes_scaling[i] * astig_rmsd_gd[:, ax_ix], 'om', ms=4,label='RMSD (Astig, GD)')
        #ax.loglog(photoncounts, axes_scaling[i] * astig_prec[:, ax_ix], '+g', ms=4,label='Prec.(Astig.)')
        ax.legend()
        ax.set_title(f'Estimation precision for axis {ax_ix}')
        ax.set_xlabel('Photon count [photons]')
        ax.set_ylabel(f'Precision [{axes_units[i]}]')

    plt.show()

