from config import PromweaverCubeConfig, AtomicConfig, WavelengthSynthConfig
# import numpy as np

# basic_config = PromweaverCubeConfig(
#     cube_path="sparse_fil.nc",
#     output_path="sparse_fil_synth.nc",
#     mode="Filament",
#     atoms=AtomicConfig(
#         active_atoms=["H", "Ca"]
#     ),
#     prd=True,
#     num_processes=8,
#     num_threads=2,
#     memory_limit="4GB",
# )

entries = range(500, 1000, 10)
for entry in entries:
    # basic_config.cube_path = f"valeriia_prom_{entry:04d}.nc"
    # basic_config.output_path = f"valeriia_prom_{entry:04d}_synth.nc"
    config = PromweaverCubeConfig(
        cube_path=f"valeriia_prom_{entry:04d}.nc",
        output_path = f"valeriia_prom_{entry:04d}_synth.nc",
        num_processes=6,
        num_threads=12,
        memory_limit="14GB",
        mode="Prominence",
        bc_type="Cones",
        num_rays=10,
        atoms=AtomicConfig(
            active_atoms=["H", "Ca", "Mg"]
        ),
        prd=True,
        conserve_charge=True,
        conserve_pressure=True,
        pops_tol=1e-3,
        prd_tol=1e-2,
        J_tol=5e-3,
        prominence_slice_axis="x",
        filament_slice_axis="z",
        base_altitude=0.0,
    )
    content = config.model_dump_json(indent=2)

    with open(f"Configs/pw_cube_config_{entry}.json", "w") as f:
        f.write(content)