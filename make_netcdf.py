import numpy as np
from netCDF4 import Dataset
from lightweaver.fal import Falc82

file = Dataset("test_atmos_cube.nc", mode="w")
x_dim = file.createDimension("x", 4)
y_dim = file.createDimension("y", 4)
z_dim = file.createDimension("z", 82)

temperature = file.createVariable("temperature", "f8", (x_dim, y_dim, z_dim))
z = file.createVariable("z", "f8", (z_dim))
vlos = file.createVariable("vlos", "f8", (x_dim, y_dim, z_dim))
vturb = file.createVariable("vturb", "f8", (x_dim, y_dim, z_dim))
ne = file.createVariable("ne", "f8", (x_dim, y_dim, z_dim))
nh_tot = file.createVariable("nh_tot", "f8", (x_dim, y_dim, z_dim))
param = file.createVariable("param", "f8", (x_dim, y_dim, z_dim))

fal = Falc82()
z[:] = fal.z[None, None, :]
temperature[:, :, :] = fal.temperature[None, None, :]
vlos[:, :, :] = 0.0
vturb[:, :, :] = fal.vturb[None, None, :]
ne[:, :, :] = fal.ne[None, None, :]
nh_tot[:, :, :] = fal.nHTot[None, None, :]
param[:, :, :] = np.arange(16).reshape(4, 4)[:, :, None]

file.close()
