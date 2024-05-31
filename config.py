from pathlib import Path
from pydantic import BaseModel, Field
from typing import Literal, Union, List, Optional

class WavelengthSynthConfig(BaseModel):
    mus: List[float] = Field(default_factory=lambda: [1.0])
    """Inclinations (cos(theta)) to synthesise at"""
    wavelengths: Optional[List[float]] = None
    """Wavelength grid to synthesise at. Default: the grid used for the radiative model."""

class DepthDataConfig(BaseModel):
    wavelengths: Optional[List[float]] = None
    """Wavelengths to save following depth data fields. Default None, i.e. all. These are saved from the Context used for the final synthesis so have the same grids as used there."""
    eta: bool = True
    """Save total eta"""
    chi: bool = True
    """Save total chi"""
    tau: bool = False
    """Save tau"""
    J: bool = False
    """Save J"""
    sca: bool = False
    """Save background scattering"""
    Idir: bool = False
    """Save directional intensity (from the base context -- warning can be _very_ large)"""

class AtomicModelRef(BaseModel):
    """Description of how to load an atomic model"""
    module: str
    """The module to load from (e.g. lightweaver.rh_atoms)"""
    name: str
    """The model to load from module (e.g. H_6_atom)"""

class AtomicConfig(BaseModel):
    active_atoms: List[str]
    """List of models (by element name) to treat as active (e.g. ["H", "Ca", "Mg"])"""
    atomic_models: Optional[List[AtomicModelRef]] = None
    """Optional list of atomic models to override the default set. Note that if using this functionality, all desired models must be provided."""

class PromweaverCubeConfig(BaseModel):
    cube_path: Path
    """Path to netCDF cube of data"""
    mode : Union[Literal["Filament"], Literal["Prominence"], Literal["Both"]]
    """Synthesis mode"""
    atoms: AtomicConfig
    """Atomic model configuration"""
    prd : bool = False
    """Whether to use PRD in synthesis (default: False)"""
    conserve_charge : bool = True
    """Whether to conserve charge in the model (default: True)"""
    conserve_pressure: bool = True
    """Whether to conserve pressure in the model (default: True)"""
    synth_config: WavelengthSynthConfig = Field(default_factory=WavelengthSynthConfig)
    """Configuration for final wavelength synthesis"""
    depth_data_config: Optional[DepthDataConfig] = None
    """Configuration for depth data (Default: None, i.e. don't save)"""
    ctx_kwargs: dict = Field(default_factory=dict)
    """Extra kwargs to pass to context construction... override formal solvers etc."""
