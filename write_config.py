from config import PromweaverCubeConfig

basic_config = PromweaverCubeConfig(
    cube_path="test_prom_cube.nc",
    mode="Both",
)
content = basic_config.model_dump_json(indent=2)

with open("pw_cube_config.json", "w") as f:
    f.write(content)