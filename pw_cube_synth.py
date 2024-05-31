from config import AtomicConfig, AtomicModelRef, DepthDataConfig, PromweaverCubeConfig, WavelengthSynthConfig
import importlib
import argparse
from pathlib import Path
import promweaver as pw

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

def result_template(data, conf: PromweaverCubeConfig):
    # TODO(cmo): Build the template
    pass



# def compute_bc_table():

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
    print(config)
