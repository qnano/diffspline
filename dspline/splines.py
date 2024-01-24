# -*- coding: utf-8 -*-
"""
Implementation of 1D,2D and 3D catmull-rom splines and hermite splines in pytorch.
Copyright Jelmer Cnossen 2021/2022
"""
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.nn.parameter import Parameter
import numpy as np
import matplotlib.pyplot as plt
import tqdm
import itertools


def _S(X):
    return X.detach().cpu().numpy()


class SplineBase(torch.nn.Module):
    def __init__(self, knots, catmull: bool, device=None):
        super(SplineBase, self).__init__()

        if type(knots) is not torch.Tensor:
            knots = torch.tensor(knots, dtype=torch.float32, device=device)

        self.knots = Parameter(knots)

        if catmull:
            hermiteBasis = np.array([[2, -2, 1, 1],
                                     [-3, 3, -2, -1],
                                     [0, 0, 1, 0],
                                     [1, 0, 0, 0]])

            catmullRom = np.array([[0, 1, 0, 0],
                                   [0, 0, 1, 0],
                                   [-0.5, 0, 0.5, 0],
                                   [0, -0.5, 0, 0.5]])

            A = ((hermiteBasis @ catmullRom)[::-1]).copy()
        else:
            hermiteBasis = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [-3, -2, 3, -1],
                [2, 1, -2, 1]
            ])

            A = hermiteBasis.copy()  # #(hermiteBasis[::-1]).copy()

        self.spline_basis = torch.tensor(A, dtype=torch.float32, device=self.knots.device)
        self.register_buffer('spline_basis_', self.spline_basis)
        self.catmull = catmull

    @property
    def shape(self):
        return self.knots.shape

    @torch.jit.export
    def __len__(self):
        return len(self.knots)

    def _spline_idx(self, coord, axis: int):
        """
        Compute interpolation weights that, multiplied with the spline knots, 
        give you the interpolated values along the spline.
        """

        assert coord.dtype == self.spline_basis.dtype

        idx = coord.to(torch.int32)
        s = torch.clip(coord - idx, 0, 1)

        if self.catmull:
            idx = idx - 1
        else:
            idx = idx * 2

        idx = torch.clip((idx[:, None] + torch.arange(4, device=idx.device)[None, :]), 0, self.knots.shape[axis] - 1)
        c = (s[:, None] ** (torch.arange(4, device=idx.device)[None]))  # [len(pts), 4]
        #print(c)

        return idx, c @ self.spline_basis.to(idx.device)

    def _spline_idx_deriv(self, coord, axis: int):
        """
        like _spline_idx, but also returns weights to compute the derivatives w.r.t the specified axis
        """
        idx = coord.to(torch.int64)
        s = torch.clip(coord - idx, 0, 1)

        if self.catmull:
            idx = idx - 1
        else:
            idx = idx * 2

        exp = torch.arange(4, device=idx.device)[None]
        # exp = [0,1,2,3]

        deriv_exp = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32, device=idx.device)

        idx = torch.clip((idx[:, None] + torch.arange(4, device=idx.device)[None, :]), 0, self.knots.shape[axis] - 1)
        c = (s[:, None] ** exp) @ self.spline_basis_
        deriv_c = (exp * s[:, None] ** deriv_exp) @ self.spline_basis_
        return idx, c, deriv_c

    def __str__(self):
        if self.catmull:
            return f'Catmull Rom Spline with knots ({self.knots.shape})'
        else:
            return f'Hermite Spline with knots ({self.knots.shape})'


class Spline3D(SplineBase):
    """
    Spline that maps a 3D coordinate onto a n-d output
    """

    def __init__(self, knots, catmull, device=None):
        super(Spline3D, self).__init__(knots, catmull=catmull, device=device)

        assert (len(knots.shape) == 4)

    @torch.jit.export
    def get_knots(self):
        return self.knots

    def forward(self, idx):
        iz, cz = self._spline_idx(idx[:, 0], 0)
        iy, cy = self._spline_idx(idx[:, 1], 1)
        ix, cx = self._spline_idx(idx[:, 2], 2)

        # pre-reduction tensor shape is [evalpts, z-index, y-index, x-index, dims]
        sel_knots = self.knots[
            iz[:, :, None, None],
            iy[:, None, :, None],
            ix[:, None, None, :]]

        return (sel_knots *
                cz[:, :, None, None, None] *
                cy[:, None, :, None, None] *
                cx[:, None, None, :, None]).sum((1, 2, 3))

    @torch.jit.export
    def deriv(self, pts):
        """
        returns value at pts, but also derivatives w.r.t. xyz:
            
        tuple: 
            values: [ len(pts) ]
            derivs: [ len(pts), 3 ]
        """
        iz, cz, dcz = self._spline_idx_deriv(pts[:, 0], 0)
        iy, cy, dcy = self._spline_idx_deriv(pts[:, 1], 1)
        ix, cx, dcx = self._spline_idx_deriv(pts[:, 2], 2)

        # pre-reduction tensor shape is [evalpts, z-index, y-index, x-index, dims]
        sel_knots = self.knots[
            iz[:, :, None, None],
            iy[:, None, :, None],
            ix[:, None, None, :]]

        values = (sel_knots *
                  cz[:, :, None, None, None] *
                  cy[:, None, :, None, None] *
                  cx[:, None, None, :, None]).sum((1, 2, 3))

        deriv_z = (sel_knots *
                   dcz[:, :, None, None, None] *
                   cy[:, None, :, None, None] *
                   cx[:, None, None, :, None]).sum((1, 2, 3))

        deriv_y = (sel_knots *
                   cz[:, :, None, None, None] *
                   dcy[:, None, :, None, None] *
                   cx[:, None, None, :, None]).sum((1, 2, 3))

        deriv_x = (sel_knots *
                   cz[:, :, None, None, None] *
                   cy[:, None, :, None, None] *
                   dcx[:, None, None, :, None]).sum((1, 2, 3))

        return values, torch.stack((deriv_z, deriv_y, deriv_x))

    def _segment_integral_idx(self, cs, ce, axis: int):
        idx = cs.to(torch.int64)
        cs = torch.clip(cs - idx, 0, 1)
        ce = torch.clip(ce - idx, 0, 1)

        if self.catmull:
            idx = idx - 1
        else:
            idx = idx * 2

        idx = torch.clip((idx[:, None] + torch.arange(4)[None, :]), 0, self.knots.shape[axis] - 1)

        a = torch.arange(4)[None, :]
        c = ((ce[:, None] ** (a + 1) / (a + 1) - cs[:, None] ** (a + 1) / (a + 1)) @ self.spline_basis)

        return idx, c

    def _segment_integral(self, xstart, xend, ystart, yend, z):
        ix, cx = self._segment_integral_idx(xstart, xend, 2)
        iy, cy = self._segment_integral_idx(ystart, yend, 1)
        iz, cz = self._spline_idx(z, 0)

        # pre-reduction tensor shape is [evalpts, z-index, y-index, x-index, dims]
        sel_knots = self.knots[
            iz[:, :, None, None],
            iy[:, None, :, None],
            ix[:, None, None, :]]

        return (sel_knots *
                cz[:, :, None, None, None] *
                cy[:, None, :, None, None] *
                cx[:, None, None, :, None]).sum((1, 2, 3))

    @torch.jit.export
    def unit_area_integral(self, pts):
        """
        Integrate XY over an area (y-0.5 .. y+0.5, x-0.5 .. x+0.5)
        """

        x = pts[:, 2]
        y = pts[:, 1]
        z = pts[:, 0]

        x0 = x - 0.5
        x1 = torch.ceil(x0)
        x2 = x + 0.5

        y0 = y - 0.5
        y1 = torch.ceil(y0)
        y2 = y + 0.5

        return (
                self._segment_integral(x0, x1, y0, y1, z) +
                self._segment_integral(x1, x2, y0, y1, z) +
                self._segment_integral(x0, x1, y1, y2, z) +
                self._segment_integral(x1, x2, y1, y2, z)
        )


class CatmullRomSpline3D(Spline3D):
    def __init__(self, knots, device=None):
        super(CatmullRomSpline3D, self).__init__(knots, catmull=True, device=device)


class CatmullRomSpline4D(torch.nn.Module):
    def __init__(self, knots, depths, device=None):
        '''
        knots - 4D array, list of 3D PSFs over each depth
        depths - array of size len(knots[0])
        '''
        super(CatmullRomSpline4D, self).__init__()

        if type(knots) is not torch.Tensor:
            self.knots = torch.tensor(knots, dtype=torch.float32, device=device)
        else:
            self.knots = knots
        #self.knots = Parameter(knots)

        self.depths = depths
        self.time_diffs = np.diff(self.depths, prepend=0)  # time_diff[i] = time[i] - time[i-1]

        self.hermiteBasis = np.array([[2, -2, 1, 1],
                                     [-3, 3, -2, -1],
                                     [0, 0, 1, 0],
                                     [1, 0, 0, 0]])
        self.uniform_basis = torch.tensor(self.hermiteBasis @ np.array([[0, 1, 0, 0],
                                                                        [0, 0, 1, 0],
                                                                        [-0.5, 0, 0.5, 0],
                                                                        [0, -0.5, 0, 0.5]]), dtype=torch.float32, device=device)

    def get_tangents(self, delta_0, delta_1):
        return np.array([[1, 0], [-1, 1], [0, -1]]) @ np.array([delta_0 / delta_1, delta_1 / delta_0]) * 1 / (
                    delta_0 + delta_1)

    
    def get_spline_basis(self, delta_t1, delta_t2, delta_t3):
        if delta_t1 == delta_t2 == delta_t3 == 1:
            return self.uniform_basis
        s1 = delta_t2 * self.get_tangents(delta_t1, delta_t2)[::-1]
        s2 = delta_t2 * self.get_tangents(delta_t2, delta_t3)[::-1]
        catmullRom = np.array([[0, 1, 0, 0],
                               [0, 0, 1, 0],
                               [*s1, 0],
                               [0, *s2]])
        return torch.tensor(self.hermiteBasis @ catmullRom, dtype=torch.float32, device=self.knots.device)

    def get_depth_coords(self, input_depth):
        '''
        Returns the index of the depth present in the data,
        and the position "r" it has within the interval (used as input into the spline)
        '''
        ind = np.searchsorted(self.depths, input_depth, side='right')  ## depths[ind-1] <= input_depth < depths[ind] for right
        # don't use 'left' because then ind-1 breaks

        norm = (input_depth - self.depths[ind - 1]) / (self.depths[ind] - self.depths[ind - 1])
        return ind - 1, norm

    def forward(self, idx, depth):
        '''
        idx - array of shape [num_points, 3], where the triplets consist of the zyx coords
        depth - float or int, stands for the depth needed
        '''
        depth_ind, depth_r = self.get_depth_coords(depth)
        depth_idx = idx[:,0].clone()
        depth_idx[:] = depth_ind + depth_r

        idp, cd = self._spline_idx(depth_idx, 0, deltas=self.time_diffs[depth_ind:depth_ind+3])
        iz, cz = self._spline_idx(idx[:, 0], 1)
        iy, cy = self._spline_idx(idx[:, 1], 2)
        ix, cx = self._spline_idx(idx[:, 2], 3)

        sel_knots = self.knots[
            idp[:, :, None, None, None],
            iz[:, None, :, None, None],
            iy[:, None, None, :, None],
            ix[:, None, None, None, :]]

        return (sel_knots *
                cd[:, :, None, None, None, None] *
                cz[:, None, :, None, None, None] *
                cy[:, None, None, :, None, None] *
                cx[:, None, None, None, :, None]).sum((1, 2, 3, 4))


    def _spline_idx(self, coord, axis: int, deltas=None):
        """
        Compute interpolation weights that, multiplied with the spline knots,
        give you the interpolated values along the spline.
        """

        #assert coord.dtype == self.spline_basis.dtype

        idx = coord.to(torch.int32)
        s = torch.clip(coord - idx, 0, 1)

        idx = idx - 1

        idx = torch.clip((idx[:, None] + torch.arange(4, device=idx.device)[None, :]), 0, self.knots.shape[axis] - 1)
        c = (s[:, None] ** (torch.arange(4, device=idx.device).flip(0)[None]))  # [len(pts), 4]
        #print(c)
        if deltas is None:
            return idx, c @ self.get_spline_basis(1,1,1)
        else:
            return idx, c @ self.get_spline_basis(*deltas)



class CatmullRomSpline4DUniform(torch.nn.Module):
    def __init__(self, knots, depths, device=None):
        '''
        knots - 4D array, list of 3D PSFs over each depth
        depths - array of size len(knots[0])
        '''
        super(CatmullRomSpline4DUniform, self).__init__()

        if type(knots) is not torch.Tensor:
            self.knots = torch.tensor(knots, dtype=torch.float32, device=device)
        else:
            self.knots = knots
        # self.knots = Parameter(knots)

        self.depths = depths
        self.time_diffs = np.diff(self.depths, prepend=0)  # time_diff[i] = time[i] - time[i-1]

        self.hermiteBasis = np.array([[2, -2, 1, 1],
                                      [-3, 3, -2, -1],
                                      [0, 0, 1, 0],
                                      [1, 0, 0, 0]])
        self.uniform_basis = torch.tensor(self.hermiteBasis @ np.array([[0, 1, 0, 0],
                                                                        [0, 0, 1, 0],
                                                                        [-0.5, 0, 0.5, 0],
                                                                        [0, -0.5, 0, 0.5]]),
                                          dtype=torch.float32, device=device)

    def get_depth_coords(self, input_depth):
        '''
        Returns the index of the depth present in the data,
        and the position "r" it has within the interval (used as input into the spline)
        '''
        ind = np.searchsorted(self.depths, input_depth,
                              side='right')  ## depths[ind-1] <= input_depth < depths[ind] for right
        # don't use 'left' because then ind-1 breaks

        norm = (input_depth - self.depths[ind - 1]) / (self.depths[ind] - self.depths[ind - 1])
        return ind - 1, norm

    def forward(self, idx, depth):
        '''
        idx - array of shape [num_points, 3], where the triplets consist of the zyx coords
        depth - float or int, stands for the depth needed
        '''
        depth_ind, depth_r = self.get_depth_coords(depth)
        depth_idx = idx[:, 0].clone()
        depth_idx[:] = depth_ind + depth_r

        idp, cd = self._spline_idx(depth_idx, 0)
        iz, cz = self._spline_idx(idx[:, 0], 1)
        iy, cy = self._spline_idx(idx[:, 1], 2)
        ix, cx = self._spline_idx(idx[:, 2], 3)

        sel_knots = self.knots[
            idp[:, :, None, None, None],
            iz[:, None, :, None, None],
            iy[:, None, None, :, None],
            ix[:, None, None, None, :]]

        return (sel_knots *
                cd[:, :, None, None, None, None] *
                cz[:, None, :, None, None, None] *
                cy[:, None, None, :, None, None] *
                cx[:, None, None, None, :, None]).sum((1, 2, 3, 4))

    def _spline_idx(self, coord, axis: int):
        """
        Compute interpolation weights that, multiplied with the spline knots,
        give you the interpolated values along the spline.
        """

        # assert coord.dtype == self.spline_basis.dtype

        idx = coord.to(torch.int32)
        s = torch.clip(coord - idx, 0, 1)

        idx = idx - 1

        idx = torch.clip((idx[:, None] + torch.arange(4, device=idx.device)[None, :]), 0, self.knots.shape[axis] - 1)
        c = (s[:, None] ** (torch.arange(4, device=idx.device).flip(0)[None]))  # [len(pts), 4]
        # print(c)
        return idx, c @ self.uniform_basis

class FixedSpline3D(torch.nn.Module):

    @staticmethod
    def from_catmull_rom_median_padding(spl: CatmullRomSpline3D, device=None):
        """
        Convert a catmull-rom spline defined in terms of node positions, 
        to one defined in terms of precomputed cubic coefficients.
        """
        dfspl_knots = spl.knots.detach()

        if device is None:
            device = dfspl_knots.device

        dfspl_knots = dfspl_knots.to(device)
        roisize = dfspl_knots.shape[1]

        border_pixel_median = torch.median(torch.hstack(
            [dfspl_knots[:, :, 0].squeeze(), dfspl_knots[:, :, -1].squeeze(), dfspl_knots[:, 0, 1:-1].squeeze(),
             dfspl_knots[:, -1, 1:-1].squeeze()]), axis=-1).values
        padded_knots = border_pixel_median.reshape(-1, 1, 1).repeat((1, roisize+2, roisize+2))
        padded_knots[:, 1:-1, 1:-1] = dfspl_knots.squeeze()
        knots = padded_knots[..., None]

        ix = torch.arange(dfspl_knots.shape[2] - 1, device=device)
        ix = torch.clip((ix[:, None] + torch.arange(4, device=device)[None, :]), 0, knots.shape[2] - 1)

        iy = torch.arange(dfspl_knots.shape[1] - 1, device=device)
        iy = torch.clip((iy[:, None] + torch.arange(4, device=device)[None, :]), 0, knots.shape[1] - 1)

        iz = torch.arange(dfspl_knots.shape[0] - 1, device=device)
        iz = torch.clip((iz[:, None] + torch.arange(4, device=device)[None, :]), 0, knots.shape[0] - 1)

        cz = (knots[iz[:, None]] * spl.spline_basis[None, :, :, None, None, None]).sum(2)
        cyz = (cz[:, :, iy[:, None]] * spl.spline_basis[None, None, None, :, :, None, None]).sum(4)
        cxyz = (cyz[:, :, :, :, ix[:, None]] * spl.spline_basis[None, None, None, None, None, :, :, None]).sum(6)
        coeff = cxyz.permute((0, 2, 4, 1, 3, 5, 6))

        return FixedSpline3D(coeff)

    @staticmethod
    def from_catmull_rom(spl: CatmullRomSpline3D, device=None):
        """
        Convert a catmull-rom spline defined in terms of node positions,
        to one defined in terms of precomputed cubic coefficients.
        """
        knots = spl.knots.detach()

        if device is None:
            device = knots.device

        orig_knots = knots.to(device)

        medians = torch.median(torch.hstack([knots[:, :, 0].squeeze(), knots[:, :, -1].squeeze(), knots[:, 0, 1:-1].squeeze(),
             knots[:, -1, 1:-1].squeeze()]), axis=-1).values[:, None, None, None]  # (n_zplanes,1,1,1) medians
        knots = torch.concat([knots, medians.repeat((1, knots.shape[1], 1, 1), 1)], axis=2)
        knots = torch.concat([knots, medians.repeat((1, 1, knots.shape[2], 1), 1)], axis=1)

        ix = torch.arange(knots.shape[2] - 1, device=device) - 1
        ix = torch.clip((ix[:, None] + torch.arange(4, device=device)[None, :]), 0, orig_knots.shape[2] - 1)

        iy = torch.arange(knots.shape[1] - 1, device=device) - 1
        iy = torch.clip((iy[:, None] + torch.arange(4, device=device)[None, :]), 0, orig_knots.shape[1] - 1)

        iz = torch.arange(knots.shape[0] - 1, device=device) - 1
        iz = torch.clip((iz[:, None] + torch.arange(4, device=device)[None, :]), 0, orig_knots.shape[0] - 1)

        cz = (orig_knots[iz[:, None]] * spl.spline_basis[None, :, :, None, None, None]).sum(2)
        cyz = (cz[:, :, iy[:, None]] * spl.spline_basis[None, None, None, :, :, None, None]).sum(4)
        cxyz = (cyz[:, :, :, :, ix[:, None]] * spl.spline_basis[None, None, None, None, None, :, :, None]).sum(6)
        coeff = cxyz.permute((0, 2, 4, 1, 3, 5, 6))

        return FixedSpline3D(coeff)

    def __init__(self, coeffs):
        super(FixedSpline3D, self).__init__()

        self.coeffs = coeffs
        self.shape = coeffs.shape

    def _spline_idx(self, x, axis: int):
        idx = x.to(torch.int64)
        idx = torch.clip(idx, 0, self.coeffs.shape[axis] - 1)
        s = torch.clip(x - idx, 0, 1)

        c = (s[:, None] ** (torch.arange(4, device=idx.device)[None]))
        return idx, c

    def forward(self, pts):
        iz, cz = self._spline_idx(pts[:, 0], 0)
        iy, cy = self._spline_idx(pts[:, 1], 1)
        ix, cx = self._spline_idx(pts[:, 2], 2)

        print(iz.shape)
        sel_knots = self.coeffs[
            iz[None, :, None, None],
            iy[None, None, :, None],
            ix[None, None, None, :]]
        print(sel_knots.shape, iz.shape)

        # Two options. Second one does 3900 frames/s, first one 3200 frames/s
        # value = (((sel_knots * cz[:,:,None,None]).sum(1) * cy[:,:,None]).sum(1) * cx).sum(1)

        value = sel_knots * cz[:, :, None, None] * cy[:, None, :, None] * cx[:, None, None, :]
        value = value.sum((1, 2, 3))
        return value

    def _spline_idx_deriv(self, coord, axis: int):
        """
        like _spline_idx, but also returns weights to compute the derivatives w.r.t the specified axis
        """
        idx = coord.to(torch.int64)
        idx = torch.clip(idx, 0, self.coeffs.shape[axis] - 1)
        s = torch.clip(coord - idx, 0, 1)

        exp = torch.arange(4, device=idx.device)[None]
        deriv_exp = torch.tensor([[0, 0, 1, 2]], dtype=torch.int32, device=idx.device)

        # idx = torch.clip((idx[:,None] + torch.arange(4, device=idx.device)[None,:]), 0, self.coeffs.shape[axis]-1)
        c = s[:, None] ** exp
        deriv_c = exp * s[:, None] ** deriv_exp
        return idx, c, deriv_c

    @torch.jit.export
    def deriv(self, pts):
        iz, cz, dcz = self._spline_idx_deriv(pts[:, 0], 0)
        iy, cy, dcy = self._spline_idx_deriv(pts[:, 1], 1)
        ix, cx, dcx = self._spline_idx_deriv(pts[:, 2], 2)

        sel_knots = self.coeffs[
            iz[:, :, None, None],
            iy[:, None, :, None],
            ix[:, None, None, :]]

        value = (sel_knots * cz[:, :, None, None, None] * cy[:, None, :, None, None] * cx[:, None, None, :, None]).sum(
            (1, 2, 3))
        deriv_z = (sel_knots * dcz[:, :, None, None, None] * cy[:, None, :, None, None] * cx[:, None, None, :,
                                                                                          None]).sum((1, 2, 3))
        deriv_y = (sel_knots * cz[:, :, None, None, None] * dcy[:, None, :, None, None] * cx[:, None, None, :,
                                                                                          None]).sum((1, 2, 3))
        deriv_x = (sel_knots * cz[:, :, None, None, None] * cy[:, None, :, None, None] * dcx[:, None, None, :,
                                                                                         None]).sum((1, 2, 3))

        return value, torch.stack((deriv_z, deriv_y, deriv_x))


class HermiteSpline3D(Spline3D):
    def __init__(self, knots):
        super(HermiteSpline3D, self).__init__(knots, catmull=False)


def integratedSpline1D(x, weights):
    """
    x is integrated from x-0.5 to x+0.5
    
    this means adding two integrals: from x-0.5 to ceil(x-0.5), and ceil(x-0.5) to x+0.5
    """
    hermiteBasis = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-3, -2, 3, -1],
        [2, 1, -2, 1]
    ])

    def evalSegment(xs, xe):
        idx = xs.astype(np.int32)
        xs = xs - idx
        xe = xe - idx

        idx_ = np.clip(((idx * 2)[:, None] + np.arange(4)[None, :]), 0, len(weights) - 1)

        val = np.zeros((len(x), weights.shape[1]))
        for a in range(4):
            for i in range(4):
                val += (xe ** (a + 1) / (a + 1) - xs ** (a + 1) / (a + 1)) * hermiteBasis[a, i] * weights[idx * 2 + i]

        a = np.arange(4)[None, :]
        w = ((xe[:, None] ** (a + 1) / (a + 1) - xs[:, None] ** (a + 1) / (a + 1)) @ hermiteBasis)
        # print(f"np w={w}")
        sel_weights = weights[idx_]
        # print(f"np sel w={sel_weights}")
        val2 = (w[:, :, None] * sel_weights).sum(1)

        """
        a = np.arange(4)[None,:,None]
        i = np.arange(4)[None,None,:]
        w = (xe**(a+1)/(a+1) - xs**(a+1)/(a+1)) * hermiteBasis[a,i] * weights[idx*2+i]
        tmp2 = ((xe**(a+1)/(a+1) - xs**(a+1)/(a+1)))

        val3=w.sum((1,2))
        """

        print(f"true: {val}, test:{val2}")
        return val2

    x0 = x - 0.5
    x1 = np.ceil(x0)
    x2 = x + 0.5

    return evalSegment(x0, x1) + evalSegment(x1, x2)


def evalHermiteSpline1D(x, weights):
    hermiteBasis = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-3, -2, 3, -1],
        [2, 1, -2, 1]
    ])

    idx = x.astype(np.int32)
    sx = x - idx

    idx_ = np.clip(((idx * 2)[:, None] + np.arange(4)[None, :]), 0, len(weights) - 1)

    tmp = (sx[:, None] ** (np.arange(4)[None]))
    w = tmp @ hermiteBasis

    val2 = (weights[idx_] * w[:, :, None]).sum(1)

    return val2


def example1D_hermite():
    weights = np.zeros((20))
    weights[::2] = np.random.uniform(size=10)
    weights = weights[:, None]

    xfull = np.linspace(1, 8, 100)
    spline = HermiteSpline1D(weights)
    yfull = _S(spline(xfull))  # hermiteSpline1D(xfull, weights)
    yfull2 = evalHermiteSpline1D(xfull, weights)

    plt.clf()
    plt.plot(xfull, yfull2, 'k')
    plt.plot(xfull, yfull + 0.1, 'r--')
    plt.plot(weights[::2], 'o')


def example1D_hermite_unit_integral():
    pos = np.random.uniform(1, 8)

    weights = np.zeros((20))
    weights[::2] = np.random.uniform(size=10)
    weights = weights[:, None]

    x = np.linspace(pos - 0.5, pos + 0.5, 200)
    y = evalHermiteSpline1D(x, weights)

    xfull = np.linspace(1, 8, 100)
    spline = torch.jit.script(HermiteSpline1D(weights))
    yfull = _S(spline(torch.tensor(xfull).float()))  # hermiteSpline1D(xfull, weights)
    yfull2 = evalHermiteSpline1D(xfull, weights)

    plt.clf()
    plt.plot(xfull, yfull2, 'k')
    plt.plot(xfull, yfull, 'r--')
    plt.plot(x, y, label='y [1::2]=0')
    plt.plot(weights[::2], 'o')

    sum2 = integratedSpline1D(np.array([pos]), weights)[0, 0]

    torch_val = _S(spline.unit_integral(Tensor([pos])))

    ni = y.sum() / len(x)

    print(f"Integrated ({sum2}) vs midpoint ({y[len(y) // 2]}), torch model: {torch_val}, numerically: {ni} ")
    assert abs(torch_val - ni) < 0.01
    assert abs(sum2 - ni) < 0.01


def example_catmull_1D():
    x = np.linspace(0, 5, num=100)
    weights = np.random.uniform(size=5)

    plt.figure()
    plt.plot(weights, 'o')

    spline = CatmullRomSpline1D(weights[:, None])
    y2 = _S(spline.test(torch.tensor(x)))
    plt.plot(x, y2[:, 0], label='original')

    fixed_spline = spline.fixed_spline()
    y3 = _S(fixed_spline(torch.tensor(x)))
    plt.plot(x, y3[:, 0] + 0.1, label='fixed spline')
    plt.title('Fixed-coefficients spline test')
    plt.legend()

    plt.figure()
    nd = (y2[2:] - y2[:-2]) / (2 * (x[1] - x[0]))
    plt.plot(x[1:-1], nd + 0.1, label='numerical deriv. (shifted)')

    deriv = _S(spline.deriv(torch.Tensor(x))[1])
    plt.plot(x, deriv[:, 0], label='spline.deriv.')
    plt.legend()


def example_catmull_2D():
    x = np.arange(6)
    X, Y = np.meshgrid(x, x)
    knots = np.random.uniform(-2, 2, size=(len(x), len(x), 2))
    knots[:, :, 0] += X * 10
    knots[:, :, 1] += Y * 10

    evalpts = []
    N = 400
    L = len(x) - 1
    for y in np.linspace(0, L, 6):
        evalpts.append(np.array([np.linspace(0, L, N), np.ones(N) * y]).T)
        evalpts.append(np.array([np.ones(N) * y, np.linspace(0, L, N)]).T)

    evalpts = np.concatenate(evalpts)

    evalpts2 = np.zeros((20, 2))
    evalpts2[:, 0] = 0.5
    evalpts2[:, 1] = np.linspace(0, L, len(evalpts2))

    plt.figure()
    plt.scatter(knots[:, :, 0].flatten(), knots[:, :, 1].flatten())

    spline = CatmullRomSpline2D(knots)
    so = _S(spline(torch.tensor(evalpts, dtype=torch.float32)))

    # spline.test(torch.Tensor(evalpts2))

    plt.scatter(so[:, 0], so[:, 1], s=1)

    fs = spline.fixed_spline()
    so2 = _S(fs(torch.tensor(evalpts, dtype=torch.float32)))
    plt.scatter(so2[:, 0] + 1, so2[:, 1] + 1, s=1, label='Precomputed')
    plt.title('Example mapping a 2D coordinate to a 2D output')


def example_catmull_3D():
    x = np.arange(6)
    X, Y, Z = np.meshgrid(x, x, x)
    knots = np.random.uniform(-2, 2, size=(len(x), len(x), len(x), 1))

    N = 100
    evalpts = np.ones((N, 3)) * 3
    evalpts[:, 0] = np.linspace(0, 5, N)

    spline = CatmullRomSpline3D(knots)
    so = _S(spline.deriv(torch.tensor(evalpts, dtype=torch.float32))[1])
    # so2 = _S(spline(torch.tensor(evalpts+0.1, dtype=torch.float32)))

    val = _S(spline(torch.tensor(evalpts, dtype=torch.float32)))

    # spline.test(torch.Tensor(evalpts2))

    fig, axes = plt.subplots(2)
    axes[0].plot(val)
    axes[1].plot(so[0], label='diff')
    # plt.plot(so2)

    fs = FixedSpline3D.from_catmull_rom(spline)
    so2 = _S(fs.deriv(torch.tensor(evalpts, dtype=torch.float32))[1])
    axes[1].plot(so2[0] + 0.1, label='fixed')
    axes[1].legend()
    # plt.scatter(so2[:,0]+1, so2[:,1]+1, s=1, label='Precomputed')


def test_spline_lsq():
    # random data
    x = np.random.uniform(0, 10, size=100)
    y = np.random.poisson(10 + (x - 5) ** 2)

    fig = plt.figure()
    plt.scatter(x, y)

    spline = CatmullRomSpline1D(np.zeros((6, 1)))
    optimizer = torch.optim.SGD(spline.parameters(), lr=1, momentum=0)

    x = Tensor(x)
    y = Tensor(y)

    epochs = 50
    line = None
    for i in range(epochs):
        optimizer.zero_grad()

        output = spline(x * 0.5)[:, 0]

        loss = torch.mean((output - y) ** 2)
        loss.backward()
        optimizer.step()

        xl = np.linspace(0, 10)
        result = _S(spline(xl * 0.5)[:, 0])

        if line is None:
            line = plt.plot(xl, result, 'k', label='result')[0]
            plt.legend()
        else:
            line.set_ydata(result)

        fig.gca().set_title(f"epoch {i}")
        fig.canvas.draw()
        fig.canvas.flush_events()

    xl = np.arange(0, 6)
    result = _S(spline(xl)[:, 0])
    plt.plot(xl * 2, result, 'or')
    fig.canvas.draw()

    return spline


class AdaptiveOptimizer(Optimizer):
    def __init__(self):
        ...


def test_estimator_1D():
    import scipy.special as sps
    """
    Generate positions
    Generate samples
    Estimate 1D-PSF at position + error
    """

    N = 1000
    roisize = 16

    # xpos = np.array([roisize/2, roisize/2+3, roisize/2-4])
    xpos = np.random.normal(roisize / 2, 0.5, size=N).astype(np.float32)
    sigma = 1.5
    intensity = 100

    OneOverSqrt2Sigma = (1.0 / (np.sqrt(2) * sigma))

    true_bg = 0.5
    Xc = np.arange(roisize)
    Xexp0 = (Xc[None] - xpos[:, None] + 0.5) * OneOverSqrt2Sigma
    Xexp1 = (Xc[None] - xpos[:, None] - 0.5) * OneOverSqrt2Sigma
    Ex = 0.5 * sps.erf(Xexp0) - 0.5 * sps.erf(Xexp1)
    mu = Ex * intensity + true_bg

    # plt.plot(mu[0])

    smp = np.random.poisson(mu)
    for i in range(3):
        plt.figure()
        x = np.linspace(0, roisize, 100)
        y = np.exp(-((xpos[i] - x) / sigma) ** 2 / 2) / (sigma * np.sqrt(2 * np.pi)) / (len(x) / roisize) * intensity

        plt.bar(np.arange(roisize), smp[i])
        plt.plot(x, y * (len(x) / roisize), 'k')

    plt.gcf().canvas.flush_events()

    spline = HermiteSpline1D(np.ones((roisize * 2, 1)))
    bg = Tensor(1)

    params = itertools.chain(spline.parameters(), (bg,))
    optimizer = torch.optim.SGD(params, lr=0.01, momentum=0)
    # optimizer = torch.optim.Adam(params, lr=1e-4)
    line = None

    smp = Tensor(smp.astype(np.float32))
    xpos = Tensor(xpos.astype(np.float32))
    evalpos = ((torch.arange(roisize)[None, :] - xpos[:, None] + roisize / 2).flatten()).to(torch.float32)

    fig = plt.figure()

    n_epochs = 50000
    with tqdm.tqdm(total=n_epochs) as pb:
        for epoch in range(n_epochs):

            optimizer.zero_grad()

            # evaluate spline at all xpos
            model_eval = spline(evalpos).reshape((len(xpos), -1))
            # model_eval = spline.unit_integral(evalpos).reshape((len(xpos),-1))
            evals = torch.clip(model_eval + bg, 1e-5)

            #    likelihood = p(smp | model)
            # ll = smp * log(mu) - mu

            bg_penalty = torch.mean(model_eval)

            ll = -torch.mean(smp * torch.log(evals) - evals) + bg_penalty
            # print(_S(ll))

            ll.backward()
            optimizer.step()

            with torch.no_grad():
                spline.knots[::2, 0] = torch.maximum(spline.knots[::2, 0], torch.zeros(1))

            if False:
                xl = np.linspace(0, roisize - 1, 100)
                result = _S(spline(xl)[:, 0])

                if line is None:
                    line = plt.plot(xl, result, 'k', label='result')[0]
                    plt.legend()
                else:
                    line.set_ydata(result)

                fig.gca().set_title(f"epoch {epoch}")
                fig.canvas.draw()
                fig.canvas.flush_events()

            pb.set_description(f"ll={_S(ll)}. bg={_S(bg)}")
            pb.update(1)

    xl = np.linspace(0, roisize - 1, 100)
    result = _S(spline(xl)[:, 0])
    line = plt.plot(xl, result, 'k', label='result')[0]
    y = np.exp(-((xl - roisize / 2) / sigma) ** 2 / 2) / (sigma * np.sqrt(2 * np.pi)) / (len(x) / roisize) * intensity
    plt.plot(xl, y / np.max(y) * np.max(result), label='original gaussian')
    plt.legend()

    return spline


def evalHermiteSpline2D(pts, weights):
    spline_basis = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-3, -2, 3, -1],
        [2, 1, -2, 1]
    ])

    """    
    idx = x.astype(np.int32)
    sx = x-idx 
    
    idx_ = np.clip(( (idx*2)[:,None] + np.arange(4)[None,:]), 0, len(weights)-1)

    tmp = (sx[:,None]**(np.arange(4)[None]))
    w = tmp @ hermiteBasis

    val2 = (weights[idx_]*w[:,:,None]).sum(1)

    """

    x = pts[:, 1]
    y = pts[:, 0]

    ix = x.astype(np.int32)
    sx = np.clip(x - ix, 0, 1)

    iy = y.astype(np.int32)
    sy = np.clip(y - iy, 0, 1)

    iy = np.clip(((iy * 2)[:, None] + np.arange(4)[None, :]), 0, weights.shape[0] - 1)
    ix = np.clip(((ix * 2)[:, None] + np.arange(4)[None, :]), 0, weights.shape[1] - 1)

    # compute sx^a * A
    cx = (sx[:, None] ** (np.arange(4)[None])) @ spline_basis
    cy = (sy[:, None] ** (np.arange(4)[None])) @ spline_basis

    sel_weights = weights[iy[:, :, None], ix[:, None, :]]

    # pre-reduction tensor shape is [evalpts, y-index, x-index, dims]
    return (sel_weights * cy[:, :, None, None] * cx[:, None, :, None]).sum((1, 2))


def evalHermiteSpline2D_(pts, weights):
    spline_basis = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [-3, -2, 3, -1],
        [2, 1, -2, 1]
    ])

    x = pts[:, 1]
    y = pts[:, 0]

    ix = x.astype(np.int32)
    sx = np.clip(x - ix, 0, 1)

    iy = y.astype(np.int32)
    sy = np.clip(y - iy, 0, 1)

    iy = np.clip(((iy * 2)[:, None] + np.arange(4)[None, :]), 0, weights.shape[0] - 1)
    ix = np.clip(((ix * 2)[:, None] + np.arange(4)[None, :]), 0, weights.shape[1] - 1)

    # compute sx^a * A
    cx = (sx[:, None] ** (np.arange(4)[None])) @ spline_basis
    cy = (sy[:, None] ** (np.arange(4)[None])) @ spline_basis

    sel_weights = weights[iy[:, :, None], ix[:, None, :]]

    # pre-reduction tensor shape is [evalpts, y-index, x-index, dims]
    return (sel_weights * cy[:, :, None, None] * cx[:, None, :, None]).sum((1, 2))


def example_hermite_3D_unit_area_integral():
    x = np.arange(8)
    knots = np.zeros((len(x) * 2, len(x) * 2, len(x) * 2, 1))
    knots[::2, ::2, ::2] = np.random.uniform(-0.1, 0.1, size=knots[::2, ::2, ::2].shape)
    spl = HermiteSpline3D(knots)

    W = 100
    L = len(x) - 1

    evalX, evalY = np.meshgrid(np.linspace(0, L, W), np.linspace(0, L, W))
    evalX = evalX.flatten()
    evalY = evalY.flatten()
    Zplane = 2.5
    evalpts = np.array([np.ones((W * W)) * Zplane, evalY, evalX])

    plt.figure()
    plt.imshow(_S(spl(Tensor(evalpts.T))).reshape((W, W)))

    if False:
        zrange = np.linspace(0, L, 100)
        imgs = []
        for z in range(len(zrange)):
            evalpts = np.array([np.ones((W * W)) * zrange[z], evalY, evalX])
            imgs.append(_S(spl(Tensor(evalpts.T))).reshape((W, W)))

        zstack = np.array(imgs)
        from imshow_to_gif import images_to_mp4
        images_to_mp4(zstack, 'interp3D.mp4')


def example_hermite_2D_image():
    x = np.arange(6)
    X, Y = np.meshgrid(x, x)
    knots = np.zeros((len(x) * 2, len(x) * 2, 1))
    knots[::2, ::2] = np.random.uniform(-0.1, 0.1, size=(len(x), len(x), 1))

    W = 100
    L = len(x) - 1

    evalX, evalY = np.meshgrid(np.linspace(0, L, W), np.linspace(0, L, W))
    evalX = evalX.flatten()
    evalY = evalY.flatten()
    evalpts = Tensor(np.array([evalX, evalY]).T)

    spl = HermiteSpline2D(knots)

    plt.figure()
    plt.imshow(_S(spl(evalpts)).reshape((W, W)))


def example_hermite_2D():
    x = np.arange(3)
    X, Y = np.meshgrid(x, x)
    knots = np.zeros((len(x) * 2, len(x) * 2, 2))
    # knots[1::2,::2] = np.random.uniform(-0.5,0.5,size=knots[::2,0].shape)
    knots[::2, ::2] = np.random.uniform(-0.1, 0.1, size=(len(x), len(x), 2))
    knots[::2, ::2, 1] += X
    knots[::2, ::2, 0] += Y

    evalpts = []

    N = 400
    L = len(x) - 1
    for y in np.linspace(0, L, 6):
        evalpts.append(np.array([np.linspace(0, L, N), np.ones(N) * y]).T)
        evalpts.append(np.array([np.ones(N) * y, np.linspace(0, L, N)]).T)

    # evalpts.append(np.array([ np.linspace(0, L, N), np.ones(N) * 1 ]).T)
    evalpts = np.concatenate(evalpts)

    plt.figure()
    plt.scatter(knots[::2, ::2, 0].flatten(), knots[::2, ::2, 1].flatten())

    N = 400
    r0 = evalHermiteSpline1D(evalpts[:, 0], knots[:, 2])
    # r0 = evalHermiteSpline2D_(np.array([ np.linspace(0, L, N), np.ones(N) * 1 ]).T, knots)
    print(r0)
    plt.scatter(r0[:, 0], r0[:, 1], s=5, c='k')

    spline = HermiteSpline2D(knots)
    so = _S(spline(torch.Tensor(evalpts)))

    plt.scatter(so[:, 0], so[:, 1], s=2)

    plt.title('Example mapping a 2D coordinate to a 2D output')


def example2D_hermite_unit_area_integral():
    x = np.arange(3)
    X, Y = np.meshgrid(x, x)
    knots = np.zeros((len(x) * 2, len(x) * 2, 1))
    # knots[1::2,::2] = np.random.uniform(-0.5,0.5,size=knots[::2,0].shape)
    # knots[::2,::2,0] = np.random.uniform(-0.1,0.1,size=(len(x)))[:,None]
    knots[::2, ::2, 0] = np.random.uniform(-0.1, 0.1, size=knots[::2, ::2, 0].shape)

    spl = HermiteSpline2D(knots)

    def unit_area_integral_numerical(xpos, ypos):
        L = 100
        cr = np.linspace(-0.5, 0.5, L)
        X, Y = np.meshgrid(cr + xpos, cr + ypos)

        X = X.flatten()
        Y = Y.flatten()
        coords = np.zeros((len(X), 2))
        coords[:, 0] = Y
        coords[:, 1] = X
        coords = Tensor(coords)

        v = _S(spl(coords).reshape((L, L)))
        return v.sum() / (L * L)

    L = 40
    cr = np.linspace(0, len(x) - 1, L)
    X, Y = np.meshgrid(cr, cr)

    X = X.flatten()
    Y = Y.flatten()
    coords = np.zeros((len(X), 2))
    coords[:, 0] = Y
    coords[:, 1] = X
    coords = Tensor(coords)

    v = _S(spl(coords).reshape((L, L)))

    plt.figure()
    plt.plot(cr, v[:, L // 2])

    spl_1D = HermiteSpline1D(knots[:, 0])

    r = np.random.uniform(-0.3, 0.3)
    pos = Tensor([[1, 1]]) + r
    int_a = _S(spl.unit_area_integral(pos))[0, 0]

    pos_1D = Tensor([1]) + r
    int_a_1D = _S(spl_1D.unit_integral(pos_1D))[0, 0]

    int_n = unit_area_integral_numerical(1 + r, 1 + r)
    print(f"2D integral: {int_a:.4f}, 1D integral:{int_a_1D:.4f}. 2D numerical: {int_n:.4f}")

    assert abs(int_a - int_n) < 0.01


def example3D_hermite_integral():
    W = 6

    weights = np.zeros((W * 2, W * 2, W * 2, 1))
    weights[::2, ::2, ::2] = np.random.uniform(size=(W, W, 1))
    spl = HermiteSpline3D(weights)

    def unit_area_integral_numerical(zpos, ypos, xpos):
        L = 500
        cr = np.linspace(-0.5, 0.5, L)
        X, Y = np.meshgrid(cr + xpos, cr + ypos)

        X = X.flatten()
        Y = Y.flatten()
        coords = np.zeros((len(X), 3))
        coords[:, 0] = zpos
        coords[:, 1] = Y
        coords[:, 2] = X
        coords = Tensor(coords)

        v = _S(spl(coords).reshape((L, L)))
        return v.sum() / (L * L)

    xpos, ypos, zpos = np.random.uniform(2, W - 2, 3)
    # print(xpos,ypos,zpos)

    """    
    zpos = 3
    spl2D = HermiteSpline2D(weights[zpos*2])
    int_a_2D = _S(spl2D.unit_area_integral(Tensor([[ypos,xpos]])))[0,0]
    print(f"2D spline integral: {int_a_2D:.4f}")
    """

    pos = [zpos, ypos, xpos]
    int_n = unit_area_integral_numerical(*pos)
    int_a = _S(spl.unit_area_integral(Tensor([pos])))[0, 0]
    center = spl(Tensor([pos]))[0, 0]

    print(f"Unit area integral: Algebraic: {int_a:.4f}, Numerical: {int_n:.4f}. Center eval: {center:.3f}")

    assert abs(int_a - int_n) < 0.01


def example3D_catmull_integral():
    W = 6

    weights = np.zeros((W * 2, W * 2, W * 2, 1))
    weights[::2, ::2, ::2] = np.random.uniform(size=(W, W, 1))
    spl = CatmullRomSpline3D(weights)

    fs = spl.fixed_spline()

    def unit_area_integral_numerical(spline, zpos, ypos, xpos):
        L = 500
        cr = np.linspace(-0.5, 0.5, L)
        X, Y = np.meshgrid(cr + xpos, cr + ypos)

        X = X.flatten()
        Y = Y.flatten()
        coords = np.zeros((len(X), 3))
        coords[:, 0] = zpos
        coords[:, 1] = Y
        coords[:, 2] = X
        coords = Tensor(coords)

        v = _S(spline(coords).reshape((L, L)))
        return v.sum() / (L * L)

    xpos, ypos, zpos = np.random.uniform(2, W - 2, 3)
    # print(xpos,ypos,zpos)

    pos = [zpos, ypos, xpos]
    center = spl(Tensor([pos]))[0, 0]
    center_fs = fs(Tensor([pos]))[0, 0]

    int_n = unit_area_integral_numerical(spl, *pos)
    int_a = _S(spl.unit_area_integral(Tensor([pos])))[0, 0]

    print(
        f"Unit area integral: Algebraic: {int_a:.4f}, Numerical: {int_n:.4f}. Center eval: {center:.3f}. Center eval (precomputed): {center_fs:.3f}")
    assert abs(int_a - int_n) < 0.01


def array_view(x):
    from photonpy.utils.ui.show_image import array_view as v
    v(x)


if __name__ == '__main__':
    # example1D_hermite_unit_integral()
    # example3D_catmull_integral()
    # example3D_hermite_integral()
    # example2D_hermite_unit_area_integral()

    # example1D_hermite()
    # test_estimator_1D()
    # example_catmull_1D()
    example_catmull_3D()
    # example_hermite_2D()
    # example_hermite_2D_image()
    # example_hermite_3D_unit_area_integral()

    # s = test_spline_lsq()
