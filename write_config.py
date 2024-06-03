from config import PromweaverCubeConfig, AtomicConfig

basic_config = PromweaverCubeConfig(
    cube_path="valeriia_a0335_prom.nc",
    output_path="a0335_prom_synth.nc",
    mode="Prominence",
    atoms=AtomicConfig(
        active_atoms=["H", "Ca"]
    )
)
content = basic_config.model_dump_json(indent=2)

with open("pw_cube_config.json", "w") as f:
    f.write(content)