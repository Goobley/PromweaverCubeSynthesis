import numpy as np
import h5py
from lightweaver.fal import Falc82

file = h5py.File("test_atmos_cube.h5", mode="w")
temperature = file.require_dataset("temperature", (4, 4, 82), dtype="f8")
z = file.require_dataset("z", (4, 4, 82), dtype="f8")
vlos = file.require_dataset("vlos", temperature.shape, dtype="f8")
vturb = file.require_dataset("vturb", temperature.shape, dtype="f8")
ne = file.require_dataset("ne", temperature.shape, dtype="f8")
nh_tot = file.require_dataset("nh_tot", temperature.shape, dtype="f8")

fal = Falc82()
z[:, :, :] = fal.z[None, None, :]
temperature[:, :, :] = fal.temperature[None, None, :]
vlos[:, :, :] = 0.0
vturb[:, :, :] = fal.vturb[None, None, :]
ne[:, :, :] = fal.ne[None, None, :]
nh_tot[:, :, :] = fal.nHTot[None, None, :]

file.close()
