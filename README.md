# Using splines for point spread function calibration at non-uniform depths in localization microscopy

## Introduction

This repository contains the source code for the paper titled "Using splines for point spread function calibration at non-uniform depths in localization microscopy". The paper preprint can be found on [Biorxiv](https://www.biorxiv.org/content/10.1101/2024.01.24.577007v1). In this document, we describe the structure of the repository and how it can be used.

## Repository setup

The repository consists of helper functions (in Python and MATLAB), diffspline code (Python) and the example notebook (Python). 
- Helper functions contain all the necessary functions for data generation, processing, spline calibration and estimation. They include ```matlab data generation```, ```service_functions```, ```view_utils``` directories, and ```generatePSF.m``` file for data generation, ```test.m``` with ```calibrate.m``` files for spline calibration (see [original code](https://github.com/jries/SMAP)). 
- The diffspline implementation is done in Python and can be found in ```dspline``` folder.
- An example of all the stages with the data generation, processing, spline calibration and estimation can be found in the [example Jupyter notebook ](https://github.com/qnano/diffspline/blob/main/Example%20notebook.ipynb).

## Dependencies

PSF localization process extensively employs ```fastPSF``` package. The installation procedure can be found on [GitLab](https://gitlab.com/jcnossen/fastpsf).

Moreover, the code generation and spline estimation code leverages MATLAB code, therefore, it is recommended to use MATLAB of the latest version. Since we call the MATLAB-implemented functions from Python interface, we require the installation of the [MATLAB engine for Python](https://nl.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).
