from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import torch

import pickle
# with open('../psfsim.pickle', 'rb') as f:
#     d = pickle.load(f)

df_psf = torch.load('../train_psf.pt')

zpos = 0

Y = np.arange(df_psf.shape[-1])
X = np.arange(df_psf.shape[-1])

X, Y = np.meshgrid(X, Y)
Z = df_psf[zpos].numpy()

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
#ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
#
# Y = torch.arange(d.roisize)
# X = torch.arange(d.roisize)
# Z = torch.arange(d.sim_zsize)
# Z, Y, X = torch.meshgrid(Z, Y, X, indexing='ij')
#
# c = d.psf
#
# z_cut = 1
# y_cut = 1
# x_cut = 1
# X = X[::z_cut, ::y_cut, ::x_cut]
# Y = Y[::z_cut, ::y_cut, ::x_cut]
# Z = Z[::z_cut, ::y_cut, ::x_cut]
# c = c[::z_cut, ::y_cut, ::x_cut]
#
# s = np.clip(c*100, a_min=0, a_max=5)
#
# img = ax.scatter(xs=X, ys=Y, zs=Z, c=c*5000, s=s, depthshade=True, marker='s', cmap=plt.hot(), alpha=0.9, linewidth=0)
# fig.colorbar(img)
# plt.show()