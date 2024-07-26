import gc
from config import AtomicConfig, AtomicModelRef, DepthDataConfig, PromweaverCubeConfig, WavelengthSynthConfig
import importlib
import argparse
from pathlib import Path
import lightweaver as lw
import numpy as np
import promweaver as pw
import xarray as xr
import dask.array as da
import multiprocessing
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, wait, ALL_COMPLETED
from compute_model_impl import compute_model_impl

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
    data = data.compute().to_dict()
    # with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as exe:
    #     future = exe.submit(compute_model_impl,
    #         data=data,
    #         conf=conf,
    #         bc_table=bc_table,
    #         projection=projection,
    #         quiet=quiet
    #     )
    #     futures = [future]
    #     wait(futures, return_when=ALL_COMPLETED)
    #     return xr.Dataset.from_dict(future.result())
    result = compute_model_impl(data=data, conf=conf, bc_table=bc_table, projection=projection, quiet=quiet)
    return xr.Dataset.from_dict(result)

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

    config = load_config(args.config_path)
    bc_table = compute_bc_table(config, Nthreads=config.num_threads)

    modes = [config.mode]
    if config.mode == "Both":
        modes = ["Filament", "Prominence"]
    for mode in modes:
        chunked_axes = get_chunked_axes(config, mode)
        chunk_0 = chunked_axes[0]
        chunk_1 = chunked_axes[1]
        ds = xr.open_dataset(config.cube_path)


        # print(f"Running {mode} mode (slicing {get_sliced_axis(config, mode)})")

        results = []
        total_cols = ds[chunk_0].shape[0] * ds[chunk_1].shape[0]
        completed = 0
        with tqdm(total=total_cols) as pbar:
            for c0_idx in range(ds[chunk_0].shape[0]):
                for c1_idx in range(ds[chunk_1].shape[0]):
                    input_data = ds.isel(**{chunk_0: [c0_idx], chunk_1: [c1_idx]})
                    column_result = compute_model(input_data, conf=config, bc_table=bc_table, projection=mode)
                    results.append(column_result)
                    completed += 1
                    pbar.update(completed)
        result = xr.combine_by_coords(results)
        result.to_netcdf(config.output_path)


        # work = xr.map_blocks(
        #     compute_model,
        #     ds,
        #     template=result_template(ds, config, mode),
        #     kwargs={"conf": config, "bc_table": bc_table, "projection": mode}
        # )
        # delayed = work.to_netcdf(
        #     config.output_path,
        #     group=mode,
        #     mode="a",
        #     compute=False
        # )

        # result = delayed.compute()

