import gc
import dask.config
from config import AtomicConfig, AtomicModelRef, DepthDataConfig, PromweaverCubeConfig, WavelengthSynthConfig
import importlib
import argparse
from pathlib import Path
import lightweaver as lw
import numpy as np
import promweaver as pw
import xarray as xr
import dask.array as da
from dask.distributed import Client
import dask.config

def load_config(path: Path):
    with open(path, "r") as f:
        data = f.read()
    return PromweaverCubeConfig.model_validate_json(data)

def load_atomic_model_ref(ref: AtomicModelRef):
    module = importlib.import_module(ref.module)
    return getattr(module, ref.name)

def get_atomic_models(conf: AtomicConfig):
    if conf.atomic_models is None:
        return pw.default_atomic_models()
    else:
        return [load_atomic_model_ref(r) for r in conf.atomic_models]

def get_sliced_axis(conf: PromweaverCubeConfig, projection: str):
    slice_axis = conf.filament_slice_axis
    if projection == "Prominence":
        slice_axis = conf.prominence_slice_axis
    return slice_axis

def get_chunked_axes(conf: PromweaverCubeConfig, projection: str):
    slice_axis = get_sliced_axis(conf, projection)
    chunked_axes = [ax for ax in ["x", "y", "z"] if ax != slice_axis]
    return chunked_axes

def get_bc_type(bc: str):
    if bc == "UniformJ":
        return pw.UniformJPromBc
    elif bc == "Cones":
        return pw.ConePromBc
    raise ValueError(f"Unknown BC Type {bc}")

def result_template(data, conf: PromweaverCubeConfig, projection: str):
    atoms = get_atomic_models(conf.atoms)
    a_set = lw.RadiativeSet(atoms)
    a_set.set_active(*config.atoms.active_atoms)
    spect = a_set.compute_wavelength_grid()
    model_wavelengths = spect.wavelength

    chunked_axes = get_chunked_axes(conf, projection)

    wavelengths = conf.synth_config.wavelengths
    if wavelengths is not None:
        wavelengths = np.array(wavelengths)
    else:
        wavelengths = model_wavelengths

    coords = {
        "wavelength": wavelengths,
        "mu_synth": np.array(conf.synth_config.mus),
        "x": data.x,
        "y": data.y,
        "z": data.z,
    }

    def extract_coords(*keys):
        res = {k: coords[k] for k in keys}
        return res

    def construct_empty_xr(*dims, dtype=None):
        arr = da.zeros(
            shape=[len(coords[d]) for d in dims],
            chunks=[1 if (d in chunked_axes) else len(coords[d]) for d in dims],
            dtype=dtype
        )
        res = xr.DataArray(
            arr,
            coords=extract_coords(*dims),
            dims=dims
        )
        return res

    # TODO(cmo): The cube either needs to be rotated or we need to be smarter with the shape here
    data_vars = {
        "I": construct_empty_xr(*chunked_axes, "mu_synth", "wavelength"),
        "num_iter": construct_empty_xr(*chunked_axes, dtype=np.int32),
    }
    for atom in conf.atoms.active_atoms:
        axis_name = f"{atom}_level"
        coords[axis_name] = range(len(a_set[atom].levels))
        data_vars[f"pops_{atom}"] = construct_empty_xr(axis_name, "x", "y", "z")

    if conf.conserve_charge or conf.conserve_pressure:
        data_vars["ne"] = construct_empty_xr("x", "y", "z")
        data_vars["nh_tot"] = construct_empty_xr("x", "y", "z")

    # TODO(cmo): Depth data

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=coords
    )
    return ds

def compute_bc_table(conf: PromweaverCubeConfig, Nthreads=1):
    models = get_atomic_models(conf.atoms)
    bc_ctx = pw.compute_falc_bc_ctx(
        active_atoms=conf.atoms.active_atoms,
        atomic_models=models,
        prd=conf.prd,
        Nthreads=Nthreads,
        quiet=True
    )
    bc_table = pw.tabulate_bc(bc_ctx)
    return bc_table

def compute_model(data, conf: PromweaverCubeConfig, bc_table, projection, quiet=True):
    gc.collect()
    bc_provider = pw.TabulatedPromBcProvider(**bc_table)

    sliced_axis = get_sliced_axis(conf, projection)
    chunked_axes = get_chunked_axes(config, projection)
    axis = data[sliced_axis]
    get_arr = lambda arr: np.ascontiguousarray(data[arr].data.reshape(-1))
    z = np.ascontiguousarray(axis)
    store_arr = lambda arr: arr
    store_arr_pops = lambda arr: arr
    if axis[1] > axis[0]:
        z = np.ascontiguousarray(axis[::-1])
        get_arr = lambda arr: np.ascontiguousarray(data[arr].data.reshape(-1)[::-1])
        store_arr = lambda arr: arr[::-1]
        store_arr_pops = lambda arr: arr[:, ::-1]

    temperature = get_arr("temperature")
    vlos = get_arr(f"v{sliced_axis}")
    vturb = get_arr("vturb")
    pressure = None
    if "pressure" in data:
        pressure = get_arr("pressure")
    nh_tot = None
    if "nh_tot" in data:
        nh_tot = get_arr("nh_tot")
    ne = None
    if "ne" in data:
        ne = get_arr("ne")

    model = pw.StratifiedPromModel(
        projection=projection.lower(),
        z=z,
        temperature=temperature,
        vlos=vlos,
        vturb=vturb,
        pressure=pressure,
        nh_tot=nh_tot,
        ne=ne,
        initial_ionisation_fraction=0.8,
        altitude=data.z.min() + conf.base_altitude,
        active_atoms=conf.atoms.active_atoms,
        atomic_models=get_atomic_models(conf.atoms),
        Nrays=conf.num_rays,
        BcType=get_bc_type(conf.bc_type),
        bc_provider=bc_provider,
        prd=conf.prd,
        conserve_charge=conf.conserve_charge,
        conserve_pressure=conf.conserve_pressure,
        ctx_kwargs=conf.ctx_kwargs,
        Nthreads=conf.num_threads,
    )
    mus = np.array(conf.synth_config.mus)
    synth_wavelengths = conf.synth_config.wavelengths
    if synth_wavelengths is not None:
        synth_wavelengths = np.array(synth_wavelengths)
    try:
        # TODO(cmo): Depth data
        num_iter = model.iterate_se(
            quiet=quiet,
            popsTol=conf.pops_tol,
            prdIterTol=conf.prd_tol,
            JTol=conf.J_tol,
        )
        Iout = model.compute_rays(mus=mus, wavelengths=synth_wavelengths, compute_rays_kwargs={"squeeze": False})[:, :, 0]
        if synth_wavelengths is None:
            synth_wavelengths = model.spect.wavelength
        Iout = Iout.T

        coords = {
            "wavelength": synth_wavelengths,
            "mu_synth": np.array(conf.synth_config.mus),
            "x": data.x,
            "y": data.y,
            "z": data.z,
        }
        spatial_shape = [data.sizes[k] for k in ["x", "y", "z"]]
        data_vars = {
            "I": ([*chunked_axes, "mu_synth", "wavelength"], Iout.reshape(1, 1, *Iout.shape)),
            "num_iter": (chunked_axes, np.array(num_iter, dtype=np.int32).reshape(1, 1)),
        }
        for atom in conf.atoms.active_atoms:
            axis_name = f"{atom}_level"
            coords[axis_name] = range(model.eq_pops[atom].shape[0])
            data_vars[f"pops_{atom}"] = (
                [axis_name, "x", "y", "z"],
                store_arr_pops(model.eq_pops[atom]).reshape(
                    model.eq_pops[atom].shape[0],
                    *spatial_shape,
                )
            )
        if conf.conserve_charge or conf.conserve_pressure:
            data_vars[f"ne"] = (["x", "y", "z",], store_arr(model.atmos.ne).reshape(*spatial_shape))
            data_vars[f"nh_tot"] = (["x", "y", "z",], store_arr(model.atmos.nHTot).reshape(*spatial_shape))
        result = xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )
    except:
        synth_wavelengths = conf.synth_config.wavelengths
        if synth_wavelengths is not None:
            synth_wavelengths = np.array(synth_wavelengths)
        else:
            synth_wavelengths = model.spect.wavelength
        coords = {
            "wavelength": synth_wavelengths,
            "mu_synth": np.array(conf.synth_config.mus),
            "x": data.x,
            "y": data.y,
            "z": data.z,
        }
        data_vars = {
            "I": (
                [*chunked_axes, "mu_synth", "wavelength"],
                np.zeros((1, 1, mus.shape[0], synth_wavelengths.shape[0]))
            ),
            "num_iter": (chunked_axes, np.array(-1, dtype=np.int32).reshape(1, 1))
        }
        spatial_shape = [data.sizes[k] for k in ["x", "y", "z"]]
        for atom in conf.atoms.active_atoms:
            axis_name = f"{atom}_level"
            coords[axis_name] = range(model.eq_pops[atom].shape[0])
            data_vars[f"pops_{atom}"] = (
                [axis_name, "x", "y", "z"],
                store_arr_pops(model.eq_pops[atom]).reshape(
                    model.eq_pops[atom].shape[0],
                    *spatial_shape,
                )
            )
        if conf.conserve_charge or conf.conserve_pressure:
            data_vars[f"ne"] = (["x", "y", "z",], store_arr(model.atmos.ne).reshape(*spatial_shape))
            data_vars[f"nh_tot"] = (["x", "y", "z",], store_arr(model.atmos.nHTot).reshape(*spatial_shape))

        result = xr.Dataset(
            data_vars=data_vars,
            coords=coords
        )

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="pw_cube_synth",
        description="Synthesises cubes of stratified models using promweaver",
    )

    parser.add_argument(
        "--config",
        dest="config_path",
        default="pw_cube_synth_config.yaml",
        help="Path to config file",
        metavar="FILE",
        type=Path,
    )
    args = parser.parse_args()

    dask.config.set({
        "distributed.worker.memory.spill": 0.95,
        "distributed.worker.memory.pause": 0.95,
        "distributed.worker.memory.target": 0.90,
        "distributed.worker.memory.terminate": 0.94,
    })
    config = load_config(args.config_path)
    client = Client(n_workers=config.num_processes, threads_per_worker=1, memory_limit=config.memory_limit)
    print(f"Dask dashboard: <{client.dashboard_link}>")
    bc_table = compute_bc_table(config, Nthreads=config.num_threads)

    modes = [config.mode]
    if config.mode == "Both":
        modes = ["Filament", "Prominence"]
    for mode in modes:
        chunks = {k: 1 for k in get_chunked_axes(config, mode)}
        ds = xr.open_dataset(config.cube_path, chunks=chunks)

        print(f"Running {mode} mode (slicing {get_sliced_axis(config, mode)})")
        work = xr.map_blocks(
            compute_model,
            ds,
            template=result_template(ds, config, mode),
            kwargs={"conf": config, "bc_table": bc_table, "projection": mode}
        )
        delayed = work.to_netcdf(
            config.output_path,
            group=mode,
            mode="a",
            compute=False
        )

        result = delayed.compute()

