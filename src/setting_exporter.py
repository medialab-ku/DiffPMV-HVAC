from __future__ import annotations
from src import *
from src.config import device

from dataclasses import dataclass, asdict, is_dataclass
from typing import Any

def _to_serializable(obj: Any) -> Any:
    # Dataclasses -> dict
    if is_dataclass(obj):
        return {k: _to_serializable(v) for k, v in asdict(obj).items()}
    
    # Mapping
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    
    # Sequence
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    
    # Torch / Numpy
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return obj.tolist()
    
    return obj


def save_yaml(Scenario: Scenario, fileName: str | Path):
    # path: yaml file name
    path = Path(f"src/Cases/{fileName}")

    with path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(_to_serializable(asdict(Scenario)), fp, sort_keys=False, allow_unicode=True)

    print(f"Scenerio yaml file was exported to {path}")


def save_pt(control_vars: torch.tensor, fileName: str | Path):
    path = Path("src/Cases/" + fileName)

    torch.save(control_vars, path)

    print(f"Initial guess file was exported to {path}")


##  MODIFY HERE


def Case1():
    Lx, Ly, Lz = 4.2, 2.7, 3.0 # real size of domain [m]
    h = 0.3
    Nx, Ny, Nz = int(round(Lx / h)), int(round(Ly / h)), int(round(Lz / h))
    print(f"Resolution: ({Nx}, {Ny}, {Nz})")
    Nt = 360
    NOpt = 100
    dt = 1.0 # s 
    
    fileName = "Case1"

    Occs = [
    Occ(    
        id = 1,
        weight = 2.0,
        active_time=[(0, 360)],
        pos=[(2.1, 1.0, 1.5)] * Nt,
        met=1.0,
        clo=0.74,
        sigma=0.4,
        kernels=None,
        total_time=None,
    ),
    ]

    outlet = ((0.8, 1.6), (2.3, 2.7), (1.0, 2.0))

    obs = []

    scene = Scenario(
        dim=3,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        h=h,
        dt=dt,
        Nt=Nt,
        NOpt=NOpt,
        init_room_temp=29.8,
        rh=50.0,
        outlet=outlet,
        obstacles=obs,
        outlet_theta=math.degrees_to_radians(90.0),
        occupants=Occs,
        loss_weights={
            "velocity":    20.0,    # w1
            "temperature":  3.0,    # w2
            "continuous":   0.1,    # lambda3
        },
        lr=0.07,
        pmv_loss_fn="abs_gaussian",
    )

    control_vars = np.zeros((scene.Nt, 3))
    control_vars[:] = [0.6, 20.0, math.degrees_to_radians(120.0)]
    # control_vars[:] = [0.5, 20.0, math.degrees_to_radians(156.7)] # for RBC

    control_vars = torch.tensor(control_vars, dtype=torch.float32, device=device)

    return fileName, scene, control_vars


def Case2():
    Lx, Ly, Lz = 9.0, 2.5, 10.0 # real size of domain [m]
    h = 0.5
    Nx, Ny, Nz = int(round(Lx / h)), int(round(Ly / h)), int(round(Lz / h))
    print(f"Resolution: ({Nx}, {Ny}, {Nz})")
    Nt = 600
    NOpt = 100
    dt = 1.0 # s 
    
    fileName = "Case2"

    Occs = [
    Occ(    # Sofa
        id = 1,
        weight = 2.0,
        active_time=[(60, 300)],
        pos=[(5.0, 0.5, 7.0)] * Nt,
        met=1.0,
        clo=0.61,
        sigma=0.4,
        kernels=None,
        total_time=None,
    ),
    Occ(    # Kitchen
        id = 2,
        weight = 2.0,
        active_time=[(300, 600)],
        pos=[(7.0, 1.5, 0.5)] * Nt,
        met=2.0,
        clo=0.42,
        sigma=0.4,
        kernels=None,
        total_time=None,
    ),
    ]

    outlet = ((6.0, 7.5), (2.0, 2.5), (8.0, 8.5))

    obs = [
        ((0.0, 4.5), (0.0, 2.5), (5.5, 10.0)),
        ((0.0, 5.0), (0.0, 2.5), (0.0, 5.0)),
        ((5.0, 6.0), (0.0, 2.5), (0.0, 1.5)),
        ((8.0, 9.0), (0.0, 2.5), (0.0, 2.5)),
    ]

    scene = Scenario(
        dim=3,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        h=h,
        dt=dt,
        Nt=Nt,
        NOpt=NOpt,
        init_room_temp=29.8,
        rh=50.0,
        outlet=outlet,
        obstacles=obs,
        outlet_theta=math.degrees_to_radians(180.0),
        occupants=Occs,
        loss_weights={
            "velocity":    10.0,    # w1
            "temperature":  2.5,    # w2
            "continuous":   0.1,    # lambda3
        },
        lr=0.07,
        pmv_loss_fn="abs_gaussian",
    )

    control_vars = np.zeros((scene.Nt, 3))
    control_vars[:250] = [0.8, 21.0, math.degrees_to_radians(155.0)] # sofa
    control_vars[250:] = [1.3, 16.7, math.degrees_to_radians(100.0)] # kitchen

    control_vars = torch.tensor(control_vars, dtype=torch.float32, device=device)

    return fileName, scene, control_vars


def Case3():
    Lx, Ly, Lz = 6.4, 2.8, 7.6 # real size of domain [m]
    h = 0.4
    Nx, Ny, Nz = int(round(Lx / h)), int(round(Ly / h)), int(round(Lz / h))
    print(f"Resolution: ({Nx}, {Ny}, {Nz})")
    Nt = 360
    NOpt = 100
    dt = 1.0 # s 
    
    fileName = "Case3"

    # presenter's position
    positions = [None] * Nt
    sx = 2.4 # start position x
    slope = (6.4 - 2 * sx) / 120.0
    tx = 6.4 - sx # turning point
    for i in range(Nt):
        p_x = -slope * abs(i - 240.0) + tx
        if i < 120: p_x = sx
        positions[i] = (p_x, 1.2, 6.4)

    Occs = [
    Occ(    # listener
        id = 1,
        weight = 4.0,
        active_time=[(120, 360)],
        pos=[(3.2, 1.2, 5.2)] * Nt,
        met=1.2, # standing
        clo=0.74,
        sigma=0.6,
        kernels=None,
        total_time=None,
    ),
    Occ(    # presenter
        id = 2,
        weight = 3.0,
        active_time=[(120, 360)],
        pos=positions,
        met=1.6, # presenting
        clo=0.57,
        sigma=0.6,
        kernels=None,
        total_time=None,
    ),
    ]

    outlet = ((2.8, 3.6), (2.3, 2.8), (3.8, 4.2))

    obs = [
        ((0.0, 1.6), (0.0, 2.8), (0.0, 1.6)),
        ((4.8, 6.4), (0.0, 2.8), (0.0, 1.6)),
    ]

    scene = Scenario(
        dim=3,
        Nx=Nx,
        Ny=Ny,
        Nz=Nz,
        h=h,
        dt=dt,
        Nt=Nt,
        NOpt=NOpt,
        init_room_temp=29.8,
        rh=50.0,
        outlet=outlet,
        obstacles=obs,
        occupants=Occs,
        loss_weights={
            "velocity":     0.5 ,   # w1
            "temperature":  0.25,   # w2
            "continuous":   0.03,   # lambda3
        },
        lr=0.07,
        pmv_loss_fn="abs2_gaussian",
    )

    control_vars = np.zeros((scene.Nt, 4))
    control_vars[:, 0] = 0.9                             # velocity
    control_vars[:, 1] = 13.0                            # temperature
    control_vars[:, 2] = math.degrees_to_radians(135.0)  # phi (horizontal, perpendicular to wall)
    control_vars[:, 3] = math.degrees_to_radians(0.0)

    control_vars = torch.tensor(control_vars, dtype=torch.float32, device=device)

    return fileName, scene, control_vars


if __name__ == "__main__":
    import argparse

    cases = {"Case1": Case1, "Case2": Case2, "Case3": Case3}

    parser = argparse.ArgumentParser(description="Export scenario YAML and initial control variables (.pt)")
    parser.add_argument("case", choices=list(cases.keys()), help="Case to export")
    args = parser.parse_args()

    fileName, scene, control_vars = cases[args.case]()
    save_yaml(scene, f"{fileName}.yaml")
    save_pt(control_vars, f"{fileName}.pt")
