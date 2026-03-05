from __future__ import annotations
from src import *
from src.config import device

from phi.field import stack
from phi.math  import batch

class Env(torch.nn.Module):
    def __init__(self, scene: Scenario, control_vars, lr, roi_calc=True):
        super().__init__()
        self.scene = scene
        self.device = device

        # User Setting Variables
        self.control_vars = control_vars # initial control variable setting
        self.lr = torch.tensor(lr, dtype=torch.float32, device=self.device) # learning rate setting 

        # Scenario Settings
        self.dim = scene.dim
        self.Nx, self.Ny, self.Nz = scene.Nx, scene.Ny, scene.Nz
        self.h = scene.h
        self.dt = scene.dt
        self.init_room_temp = scene.init_room_temp
        self.Nt = scene.Nt
        self.NOpt = scene.NOpt
        self.rh = scene.rh
        self.bound = Box(x=(0,self.Nx*self.h), y=(0,self.Ny*self.h), z=(0, self.Nz*self.h))

        self.outlet = self.makeBox(scene.outlet)
        self.obstacles = [self.makeBox(ob) for ob in scene.obstacles]

        self.outlet_theta = scene.outlet_theta
        self.loss_weights = scene.loss_weights

        # Occupant Setting
        self._build_occupants(scene.occupants, roi_calc=roi_calc)

        # Velocity / Temperature fields Setting
        self.init_vf = StaggeredGrid(0, extrapolation.ZERO, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)
        self.init_tf = CenteredGrid(self.init_room_temp, extrapolation.BOUNDARY, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)

    
    def _build_occupants(self, occs, roi_calc=True):
        self.occs = [
            Occ(
                id = occ["id"],
                weight = occ["weight"],
                active_time = occ["active_time"],
                pos = occ["pos"],
                met = occ["met"],
                clo = occ["clo"],
                sigma = occ["sigma"],
                kernels = None,
                total_time=None,
            )
            for occ in occs
        ]

        if roi_calc: self._setROIKernel()
        self._setTotalTime()


    def _setROIKernel(self):
        for occ in self.occs:
            sigma2 = occ.sigma ** 2
            deno = (2.0 * math.pi * sigma2) ** (3/2)

            kernels = [None] * self.Nt

            for t in tqdm(range(self.Nt), desc=f"Setting occ{occ.id}'s Gaussian Kernel", unit="step"):
                grid = CenteredGrid(0, extrapolation.ZERO, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)

                # ROI Gaussian Kernel
                cg = CenteredGrid(grid.points) - occ.pos[t]
                df = CenteredGrid(math.vec_length(cg.values, eps=1e-6))
                gw = math.exp(-df * df / (2 * sigma2)) / deno

                normalizer = math.max(gw.values).native().item()
                kernels[t] = CenteredGrid(gw.values / normalizer, extrapolation.ZERO, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)

                for ob in self.obstacles:
                    obs_mask = ob.lies_inside(kernels[t].points)
                    kernels[t] = kernels[t] * (~obs_mask)

            occ.kernels = stack(kernels, batch('t'))
        
        # How to get a CenteredGrid from the stack: occs[0].kernels.t[150]
        print("Occupants Initialization Done")


    def _setTotalTime(self):
        for occ in self.occs:
            total_time = 0
            for start, end in occ.active_time:
                total_time += end - start
            occ.total_time = total_time
    
        

        
    @classmethod
    def from_yaml(cls, path: Path, roi_calc=True, **kwargs):
        scene = Scenario(**yaml.safe_load(path.read_text()))
        env = cls(scene, roi_calc=roi_calc, **kwargs)
        return env


    def makeBox(self, b):
        if len(b) == 2:     # dim = 2
            return Box(x=b[0], y=b[1])
        else:               # dim = 3
            return Box(x=b[0], y=b[1], z=b[2])
