import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from phi.torch.flow import *

import numpy as np
import torch
import yaml, time, os, gc

from pathlib import Path
from dataclasses import dataclass, field as dc_field
from typing import Any, Dict, List, Sequence, Optional
import collections.abc as cabc
from tqdm import tqdm

def sin(rad): return math.sin(rad)
def cos(rad): return math.cos(rad)


def get_time(sec):
    hours = int(sec // 3600)
    minutes = int((sec % 3600) // 60)
    seconds = sec % 60
    return f"{hours:02} h  {minutes:02} m  {seconds:06.3f} s"



@dataclass(slots=True)
class Occ:
    id: int                 # unique key
    weight: float           # how important this occupant is

    active_time: List[tuple[int, int]]      # exisiting time of this occupant
    pos: List[tuple[float, float]] | List[tuple[float, float, float]]    

    met: float      # metabolic rate
    clo: float      # cloth insulation
    sigma: float    # activity range

    ### ROI
    kernels: Any
    total_time: int

    def __hash__(self):
        return id(self)

    def is_active(self, t: int):
        for start, end in self.active_time:
            if start <= t < end: return True
        return False


@dataclass(slots=True)
class Scenario:
    # Control variable, Simulation bound, fields, learning rate will be defined in env.py

    dim:  int       # Simulation Dimension 
    Nx:   int       # Simulation Size x [m]
    Ny:   int       # Simulation Size y [m]
    Nz:   int       # Simulation Size z [m]

    h:    float     # Cell / Voxel Size [m]
    dt:   float     # Simulation Time Step [min]

    Nt:   int       # Total Simulation Step
    NOpt: int   # Total Optimization Step (epoch)

    # simulation setting
    init_room_temp: float   # inital Room Temperature (Uniform) [Celsius]
    rh: float               # Constant Relative Humidity [%]

    # Geometry
    outlet: tuple[tuple[float,float], tuple[float,float], tuple[float,float]] | tuple[tuple[float,float], tuple[float,float]]        # outlet position
    obstacles: list[tuple[tuple[float,float], tuple[float,float], tuple[float,float]]] | list[tuple[tuple[float,float], tuple[float,float]]]    # obstacle positions

    # Occupants
    occupants: list[Occ]

    outlet_theta: float = 0.0  # fixed outlet azimuth angle [rad]; ignored when control_vars includes theta

    # Loss weights for optimize()
    loss_weights: dict = dc_field(default_factory=lambda: {
        "velocity":    20.0,
        "temperature":  3.0,
        "continuous":   0.1,
    })

    lr: float = 0.07           # learning rate for optimizer
    pmv_loss_fn: str = "abs_gaussian"  # "abs_gaussian" or "abs2_gaussian"



__all__ = [# Imported from phi.torch.flow
            "CenteredGrid", "StaggeredGrid", "Box", "advect", "diffuse",
            "field", "math", "fluid", "extrapolation", "Obstacle", "ZERO_GRADIENT", "Scene",
            
            # Other libraries
            "warnings", "np", "torch", "time", "gc", "yaml", "Path", "tqdm", "os",
            
            # Utility functions
            "sin", "cos", "Occ", "Scenario",

            # LOG
            "get_time"
        ]