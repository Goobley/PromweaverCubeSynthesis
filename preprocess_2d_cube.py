import argparse
from pathlib import Path
import yt
from netCDF4 import Dataset
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess_2d_cube",
        description="Prep a 2D AMRVAC model for promweaver synthesis"
    )
    parser.add_argument(
        "--path",
        dest="cube_path",
        help="Input file",
        metavar="FILE",
        type=Path
    )
    parser.add_argument(
        "--output",
        dest="output_path",
        help="Output file",
        metavar="FILE",
        type=Path
    )
    parser.add_argument(
        "--prominence",
        dest="is_prom_view",
        help="Whether to write cube from prom or filament view",
        action="store_true"
    )
    args = parser.parse_args()

    base_altitude = 10e6
    units = {
        "length_unit": (1e8, "cm"),
        "temperature_unit": (1e6, "K"),
        "numberdensity_unit": (1e8, "cm**-3")
    }
    unit_velocity =    11645084.295622544
    unit_temperature = 1000000.00000
    unit_pressure =    3.1754922399999996E-002

    def temperature_(field, data):
        Te = data['Te'].value * data.ds.temperature_unit
        return Te

    def vturb_fn(T, epsilon=0.5, alpha=0.1, i=0, gamma=5/3, mH=1.6735575e-27):
        k = 1.380649e-23
        m = (1 + 4*alpha)/(1 + alpha + i) * mH
        return epsilon * np.sqrt(gamma * k * T / m)

    ds = yt.load(args.cube_path, units_override=units, unit_system="cgs")
    ds.add_field(("gas", "Temp"), function=temperature_, units="K", sampling_type="cell", force_override=True)

    x_left = -24
    x_right = 24
    y_bottom = 0
    y_top = 40

    # Resolution
    nyy = 1280
    nxx = 1536
    if args.is_prom_view:
        nyy //= 16
    else:
        nxx //= 16

    grid_kwargs = {
        "left_edge": [x_left, y_bottom, 0],
        "right_edge": [x_right, y_top, 1],
        "dims": [nxx, nyy, 1]
    }
    y_axis = ds.arbitrary_grid(**grid_kwargs)['y']
    x_axis = ds.arbitrary_grid(**grid_kwargs)['x']
    pressure = ds.arbitrary_grid(**grid_kwargs)['e'][:, :, 0] * unit_pressure
    temperature = ds.arbitrary_grid(**grid_kwargs)['Te'][:, :, 0] * unit_temperature
    rho = ds.arbitrary_grid(**grid_kwargs)['rho']
    mom_x = ds.arbitrary_grid(**grid_kwargs)['m1']
    mom_y = ds.arbitrary_grid(**grid_kwargs)['m2']
    mom_z = ds.arbitrary_grid(**grid_kwargs)['m3']
    v_x = mom_x[:, :, 0] * unit_velocity / rho[:, :, 0]
    v_y = mom_y[:, :, 0] * unit_velocity / rho[:, :, 0]
    v_z = mom_z[:, :, 0] * unit_velocity / rho[:, :, 0]

    nc = Dataset(args.output_path, mode="w")
    x_dim = nc.createDimension("x", nxx)
    y_dim = nc.createDimension("y", 1)
    z_dim = nc.createDimension("z", nyy)

    x = nc.createVariable("x", "f8", (x_dim,))
    x[:] = x_axis[:, 0] / 1e2
    y = nc.createVariable("y", "f8", (y_dim,))
    y[:] = 0.0
    z = nc.createVariable("z", "f8", (z_dim,))
    z_shifted = y_axis[0, :].value / 1e2
    z_shifted -= np.min(z_shifted)
    z_shifted += base_altitude
    z[:] = z_shifted

    nc_temperature = nc.createVariable("temperature", "f8", (x_dim, y_dim, z_dim))
    nc_temperature[...] = temperature[:, None, :]
    nc_vx = nc.createVariable("vx", "f8", (x_dim, y_dim, z_dim))
    nc_vx[...] = v_x[:, None, :] / 1e2
    nc_vy = nc.createVariable("vy", "f8", (x_dim, y_dim, z_dim))
    nc_vy[...] = v_z[:, None, :] / 1e2
    nc_vz = nc.createVariable("vz", "f8", (x_dim, y_dim, z_dim))
    nc_vz[...] = v_y[:, None, :] / 1e2
    nc_pressure = nc.createVariable("pressure", "f8", (x_dim, y_dim, z_dim))
    nc_pressure[...] = pressure[:, None, :] / 10
    nc_vturb = nc.createVariable("vturb", "f8", (x_dim, y_dim, z_dim))
    nc_vturb[...] = vturb_fn(temperature[:, None, :])

    nc.close()
