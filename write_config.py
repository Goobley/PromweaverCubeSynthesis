from config import PromweaverCubeConfig, AtomicConfig

basic_config = PromweaverCubeConfig(
    cube_path="sparse_fil.nc",
    output_path="sparse_fil_synth.nc",
    mode="Filament",
    atoms=AtomicConfig(
        active_atoms=["H", "Ca"]
    ),
    prd=True,
    num_processes=8,
    num_threads=2,
    memory_limit="4GB",
)
content = basic_config.model_dump_json(indent=2)

with open("pw_cube_config.json", "w") as f:
    f.write(content)