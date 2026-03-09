from src import *
from src.env import Env
from src.losses import LossClass
from src.config import log
from collections import defaultdict

class Forward:
    def __init__(self, env:Env):
        self.env = env
        self.dim = env.dim
        self.dt = env.dt
        self.Nx, self.Ny, self.Nz = env.Nx, env.Ny, env.Nz

        self.Nt = env.Nt
        self.occs = env.occs

        self.viscosity_coeff = 1.597e-5
        self.diffusion_coeff = 2.256e-5

        self.outlet_cells = field.mask(env.outlet)
        self.obs = env.obstacles
        self.bound = env.bound


    def step(self, velocity: StaggeredGrid, temperature: CenteredGrid, step_vars):
        vel, tem, phi = step_vars[0], step_vars[1], step_vars[2]
        theta = step_vars[3] if len(step_vars) > 3 else self.env.outlet_theta

        velocity = field.where(self.outlet_cells, 
                               (vel * sin(phi) * sin(theta),
                                vel * cos(phi),
                                vel * sin(phi) * cos(theta)),
                                velocity)             
        temperature = field.where(self.outlet_cells, tem * math.ones_like(temperature), temperature)
        
        temperature = advect.mac_cormack(temperature, velocity, dt=self.dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)
        
        velocity = diffuse.explicit(velocity, self.viscosity_coeff, dt=self.dt, substeps=3) 
        temperature = diffuse.implicit(temperature, self.diffusion_coeff, dt=self.dt)

        velocity, _ = fluid.make_incompressible(velocity, obstacles=self.obs)
   
        return velocity, temperature
    

    def tensor_pmv(self, V: StaggeredGrid, T: CenteredGrid, rh=50.0, met=1.0, clo=0.61):
        """
            V: Air velocity [m/s]
            T: Air temperature [°C]
            tr: Mean radiant temperature assumed equal to air temperature
            rh: relative humidity [%], default 50%
            met: metabolic rate 
            clo: cloth insulation
        """
        
        cvf = math.vec_length(V.at_centers().values, eps=1e-5)
        cvf = CenteredGrid(cvf, extrapolation.BOUNDARY, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)
        
        # CenteredGrid
        pa = rh * 10 * math.exp(16.6536 - 4030.183 / (T + 235))

        # Constants
        M = met * 58.15  # Metabolic rate in W/m²
        icl = clo * 0.155  # Clothing insulation in m²·K/W
        W = 0  # External work (assumed to be 0 for sedentary activity)
        
        # Derived constants
        f_cl = 1.00 + 1.29 * icl if icl < 0.078 else 1.05 + 0.645 * icl  # Clothing area factor

        # Calculate hcf (convective heat transfer coefficient)
        hcf = 12.1 * field.maximum(1e-6, math.sqrt(cvf))
        hc = hcf # initialize
        tra = T + 273 + 0.4
        taa = T + 273
        t_cla = taa + (35.5 - T) / (3.5 * icl + 0.1)
        
        p1 = icl * f_cl
        p2 = p1 * 3.96
        p3 = p1 * 100
        p4 = p1 * taa
        p5 = (308.7 - 0.028 * (M - W)) + (p2 * (tra / 100.0) ** 4)
            
        xn = t_cla / 100
        xf = t_cla / 50
        tol = 0.00015

        # Iteratively calculate t_cl (clothing surface temperature)
        for _ in range(150):
            not_converged_mask = math.abs(xn - xf) > tol
            xf = field.where(not_converged_mask, 0.5 * (xf + xn), xf)
            hcn = 2.38 * math.abs(100.0 * xf - taa) ** 0.25
            hc_mask = hcf > hcn
            hc = field.where(hc_mask, hcf, hcn)
            xn = field.where(not_converged_mask, (p5 + p4 * hc - p2 * xf**4) / (100 + p3 * hc), xn)

            if float(math.max(math.abs(xn - xf).values)) < tol: break

        tcl = 100 * xn - 273
        
        # Calculate PMV components
        hl1 = 3.05 * 0.001 * (5733 - 6.99 * (M - W) - pa)
        hl2 = math.max(0.42 * ((M - W) - 58.15), 0.0)
        hl3 = 1.7 * 0.00001 * M * (5867 - pa)
        hl4 = 0.0014 * M * (34 - T)
        hl5 = 3.96 * f_cl * (xn**4 - (tra / 100.0) ** 4)
        hl6 = f_cl * hc * (tcl - T)
        ts = 0.303 * math.exp(-0.036 * M) + 0.028
        # Calculate PMV
        pmv = ts * ((M - W) - hl1 - hl2 - hl3 - hl4 - hl5 - hl6)

        # Create a new CenteredGrid for PMV
        pmv_grid = CenteredGrid(pmv, extrapolation.BOUNDARY, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)
        
        return pmv_grid


    def simulation(self, velocity: StaggeredGrid, temperature: CenteredGrid, vars):
        for t in tqdm(range(self.Nt), desc="Simulation Steps", unit="step"):
            velocity, temperature = self.step(velocity, temperature, vars[t])
        return velocity, temperature


    def optimize(self, velocity: StaggeredGrid, temperature: CenteredGrid, control_vars):
        L = LossClass(self.env)
        pmv_loss_fn = getattr(L, f"Loss_{self.env.pmv_loss_fn}")
        loss = 0.0
        loss_dict, weight_dict = defaultdict(float), defaultdict(float)

        # weight & loss dictionary initializing
        for occ in self.occs:
            loss_dict[f"occ{occ.id}_pmv"] = 0.0
            weight_dict[f"occ{occ.id}_pmv"] = occ.weight
        
        loss_dict["velocity"]    = 0.0
        loss_dict["temperature"] = 0.0
        loss_dict["continuous"]  = 0.0

        weight_dict["velocity"]    = self.env.loss_weights["velocity"]
        weight_dict["temperature"] = self.env.loss_weights["temperature"]
        weight_dict["continuous"]  = self.env.loss_weights["continuous"]


        for t in tqdm(range(self.Nt), desc="Simulation Steps", unit="step"):
            velocity, temperature = self.step(velocity, temperature, control_vars[t])

            # PMV loss
            for occ in self.occs:
                pmv = self.tensor_pmv(velocity, temperature, met=occ.met, clo=occ.clo)
                loss_pmv = pmv_loss_fn(occ, pmv, t)
                loss_dict[f"occ{occ.id}_pmv"] += loss_pmv
                loss += weight_dict[f"occ{occ.id}_pmv"] * loss_pmv
                del pmv

            # velocity / temperature loss
            loss_vel = L.Loss_velocity(control_vars[t, 0]) 
            loss_dict["velocity"] += loss_vel
            loss += weight_dict["velocity"] * loss_vel

            loss_tem = L.Loss_temperature(control_vars[t, 1])
            loss_dict["temperature"] += loss_tem
            loss += weight_dict["temperature"] * loss_tem

            if (t+1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        # continuous loss
        loss_con = L.Loss_continuous(control_vars)
        loss_dict["continuous"] += loss_con.item()
        loss += weight_dict["continuous"] * loss_con

        # log loss terms
        ll = ""
        for occ in self.occs:
            key_str = f"occ{occ.id}_pmv"
            ll += f"  occupant {occ.id} PMV: {weight_dict[key_str]} * {loss_dict[key_str]}\n"
        ll += f"   velocity loss: {weight_dict['velocity']} * {loss_dict['velocity']}\n"
        ll += f"temperature loss: {weight_dict['temperature']} * {loss_dict['temperature']}\n"
        ll += f" continuous loss: {weight_dict['continuous']} * {loss_dict['continuous']}"
        log.log_buffer.append(ll)
    
        return loss, velocity, temperature



    def step_with_co2(self, velocity: StaggeredGrid, temperature: CenteredGrid, co2: CenteredGrid, step_vars, t: int):
        """
        Step function using the same dynamics as step() but with CO2 tracking added.
        This allows fair comparison across methods by using identical physics,
        while monitoring CO2 concentration.
        """
        vel, tem, phi = step_vars[0], step_vars[1], step_vars[2]
        theta = step_vars[3] if len(step_vars) > 3 else self.env.outlet_theta


        # ========== CO2 STEP ==========
        alpha = 0.65  # recirculation rate
        C_fresh = 400.0  # ppm
        alpha_C = 50.0  # CO2 source coefficient
        k_C = 0.00108  # CO2 diffusion coefficient

        co2_mean = field.mean(co2)
        co2_inlet_value = alpha * C_fresh + (1 - alpha) * co2_mean
        co2 = field.where(self.outlet_cells, co2_inlet_value * math.ones_like(co2), co2)

        co2_source = 0.0
        for occ in self.occs:
            if occ.is_active(t):
                co2_source += alpha_C * occ.kernels.t[t]

        co2 = advect.mac_cormack(co2, velocity, dt=self.dt) + co2_source
        co2 = diffuse.explicit(co2, k_C, dt=self.dt, substeps=3)

        velocity = field.where(self.outlet_cells, 
                               (vel * sin(phi) * sin(theta),
                                vel * cos(phi),
                                vel * sin(phi) * cos(theta)),
                                velocity)      
        temperature = field.where(self.outlet_cells, tem * math.ones_like(temperature), temperature)

        temperature = advect.mac_cormack(temperature, velocity, dt=self.dt)
        velocity = advect.semi_lagrangian(velocity, velocity, dt=self.dt)

        velocity = diffuse.explicit(velocity, self.viscosity_coeff, dt=self.dt, substeps=3)
        temperature = diffuse.implicit(temperature, self.diffusion_coeff, dt=self.dt)

        velocity, _ = fluid.make_incompressible(velocity, obstacles=self.obs)

        return velocity, temperature, co2



    def DPDE_optimize(self, velocity: StaggeredGrid, temperature: CenteredGrid, control_vars):
        L = LossClass(self.env)
        loss = 0.0
        loss_dict, weight_dict = defaultdict(float), defaultdict(float)

        C_init = 450.0  # initial CO2 concentration in ppm
        co2 = CenteredGrid(C_init, extrapolation.ZERO_GRADIENT, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)

        loss_dict["energy"]     = 0.0
        loss_dict["constraint"] = 0.0
        loss_dict["centering"]  = 0.0
        loss_dict["co2"]        = 0.0  # CO2 loss
        loss_dict["continuous"] = 0.0

        weight_dict["energy"]     = 1.0     
        weight_dict["constraint"] = 100.0
        weight_dict["centering"]  = 0.1
        weight_dict["co2"]        = 100.0   
        weight_dict["continuous"] = 0.1

        for t in tqdm(range(self.Nt), desc="Simulation Steps", unit="step"):
            velocity, temperature, co2 = self.step_with_co2(velocity, temperature, co2, control_vars[t], t)

            loss_energy = L.DPDE_energy_loss(control_vars)
            loss_dict["energy"] += loss_energy
            loss += weight_dict["energy"] * loss_energy

            loss_constraint = L.DPDE_constraint_loss(temperature)
            loss_dict["constraint"] += loss_constraint
            loss += weight_dict["constraint"] * loss_constraint

            loss_centering = L.DPDE_centering_loss(temperature)
            loss_dict["centering"] += loss_centering
            loss += weight_dict["centering"] * loss_centering

            loss_co2 = L.DPDE_CO2_loss(co2)
            loss_dict["co2"] += loss_co2
            loss += weight_dict["co2"] * loss_co2

            if (t+1) % 5 == 0:
                torch.cuda.empty_cache()
                gc.collect()

        loss_con = L.Loss_continuous(control_vars)
        loss_dict["continuous"] += loss_con.item()
        loss += weight_dict["continuous"] * loss_con

        ll = ""
        ll += f"     energy loss: {weight_dict['energy']} * {loss_dict['energy']}\n"
        ll += f" constraint loss: {weight_dict['constraint']} * {loss_dict['constraint']}\n"
        ll += f"  centering loss: {weight_dict['centering']} * {loss_dict['centering']}\n"
        ll += f"        co2 loss: {weight_dict['co2']} * {loss_dict['co2']}\n"
        ll += f" continuous loss: {weight_dict['continuous']} * {loss_dict['continuous']}"
        log.log_buffer.append(ll)

        return loss, velocity, temperature
