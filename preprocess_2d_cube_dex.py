import argparse
from pathlib import Path
import yt
from netCDF4 import Dataset
import numpy as np
import lightweaver as lw
import promweaver as pw
import astropy.constants as const


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="preprocess_2d_cube",
        description="Prep a 2D AMRVAC model for dex synthesis"
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
    grid_size = 31.25e5 # cm
    nyy = 1280
    nxx = 1536
    # NOTE(cmo): Drop resolution for initial testing
    # nyy //= 2
    # nxx //= 2

    x_left = -12
    x_right = 12
    y_bottom = 0
    y_top = 32

    # NOTE(cmo): Double resolution inside crop.
    nxx = 1536
    nyy = 2048

    # NOTE(cmo): Quadruple resolution inside crop.
    nxx = 1536*2
    nyy = 2048*2

    # NOTE(cmo): Triple resolution inside crop.
    nxx = 768*3
    nyy = 1024*3

    grid_kwargs = {
        "left_edge": [x_left, y_bottom, 0],
        "right_edge": [x_right, y_top, 1],
        "dims": [nxx, nyy, 1]
    }
    grid = ds.arbitrary_grid(**grid_kwargs)
    y_axis = grid['y']
    x_axis = grid['x']
    pressure = grid['e'][:, :, 0] * unit_pressure
    temperature = grid['Te'][:, :, 0] * unit_temperature
    rho = grid['rho']
    mom_x = grid['m1']
    mom_y = grid['m2']
    mom_z = grid['m3']
    v_x = mom_x[:, :, 0] * unit_velocity / rho[:, :, 0]
    v_y = mom_y[:, :, 0] * unit_velocity / rho[:, :, 0]
    v_z = mom_z[:, :, 0] * unit_velocity / rho[:, :, 0]

    nc = Dataset(args.output_path, mode="w")
    x_dim = nc.createDimension("x", nxx)
    z_dim = nc.createDimension("z", nyy)

    field_shape = (z_dim, x_dim)
    nc_temperature = nc.createVariable("temperature", "f4", field_shape)
    nc_temperature[...] = np.ascontiguousarray((temperature[:, :].T).astype(np.float32))
    nc_vx = nc.createVariable("vx", "f4", field_shape)
    nc_vx[...] = np.ascontiguousarray((v_x[:, None, :] / 1e2).T.astype(np.float32))
    nc_vy = nc.createVariable("vy", "f4", field_shape)
    nc_vy[...] = np.ascontiguousarray((v_z[:, None, :] / 1e2).T.astype(np.float32))
    nc_vz = nc.createVariable("vz", "f4", field_shape)
    nc_vz[...] = np.ascontiguousarray((v_y[:, None, :] / 1e2).T.astype(np.float32))
    nc_pressure = nc.createVariable("pressure", "f4", field_shape)
    nc_pressure[...] = np.ascontiguousarray((pressure[:, None, :] / 10).T.astype(np.float32))
    nc_vturb = nc.createVariable("vturb", "f4", field_shape)
    nc_vturb[...] = np.ascontiguousarray(vturb_fn(temperature[:, :].T).astype(np.float32))

    initial_ionisation_fraction = 0.8
    n_tot = nc_pressure / (const.k_B.value * nc_temperature)
    nh_tot = n_tot / (lw.DefaultAtomicAbundance.totalAbundance * (1.0 + initial_ionisation_fraction))
    ne = lw.DefaultAtomicAbundance.totalAbundance * nh_tot * initial_ionisation_fraction

    nc_nh_tot = nc.createVariable("nh_tot", "f4", field_shape)
    nc_nh_tot[...] = np.ascontiguousarray(nh_tot.astype(np.float32))
    nc_ne = nc.createVariable("ne", "f4", field_shape)
    nc_ne[...] = np.ascontiguousarray(ne.astype(np.float32))

    voxel_scale = (y_axis[0, 1] - y_axis[0, 0]).value / 1e2
    assert abs(voxel_scale - ((x_axis[1, 0] - x_axis[0, 0]).value / 1e2)) < 1e-2
    scale = nc.createVariable("voxel_scale", "f4")
    scale[...] = voxel_scale
    altitude = nc.createVariable("offset_z", "f4")
    altitude[...] = base_altitude
    offset_x = nc.createVariable("offset_x", "f4")
    offset_x[...] = -0.5 * nxx * voxel_scale

    bc_ctx = pw.compute_falc_bc_ctx(active_atoms=["H", "Ca"])
    tabulated = pw.tabulate_bc(bc_ctx, mu_grid=np.linspace(0.05, 1.0, 20))
    I_with_zero = np.zeros((tabulated["I"].shape[0], tabulated["I"].shape[1] + 1))
    I_with_zero[:, 1:] = tabulated["I"][...]
    tabulated["I"] = I_with_zero
    tabulated["mu_grid"] = np.concatenate([[0], tabulated["mu_grid"]])

    mu_dim = nc.createDimension("prom_bc_mu", tabulated["mu_grid"].shape[0])
    bc_wl_dim = nc.createDimension("prom_bc_wavelength", tabulated["wavelength"].shape[0])

    # TODO(cmo): Convert this to a group -- we don't have the reader yet
    mu_min = nc.createVariable("prom_bc_mu_min", "f4")
    mu_max = nc.createVariable("prom_bc_mu_max", "f4")
    mu_min[...] = tabulated["mu_grid"][0]
    mu_max[...] = tabulated["mu_grid"][-1]
    bc_wavelength = nc.createVariable("prom_bc_wavelength", "f4", ("prom_bc_wavelength",))
    bc_wavelength[...] = tabulated["wavelength"]
    bc_I = nc.createVariable("prom_bc_I", "f4", ("prom_bc_wavelength", "prom_bc_mu"))
    for la in range(tabulated["wavelength"].shape[0]):
        bc_I[la, :] = lw.convert_specific_intensity(
            tabulated["wavelength"][la],
            tabulated["I"][la, :],
            outUnits="kW / (m2 nm sr)"
        ).value
    nc.close()
