from src import *
from src.env import Env

class LossClass:
    def __init__(self, env:Env):
        self.Nt = env.Nt
        self.dt = env.dt

    def Loss_abs_gaussian(self, occ:Occ, pmv:CenteredGrid, t:int):
        return field.l1_loss(occ.kernels.t[t] * pmv.__abs__()) / occ.total_time


    def Loss_abs2_gaussian(self, occ:Occ, pmv:CenteredGrid, t:int):
        return field.l2_loss(occ.kernels.t[t] * pmv.__abs__()) / occ.total_time


    def Loss_temperature(self, t:float):
        return abs(t - 14.4) / self.Nt


    def Loss_velocity(self, v:float):
        return v / self.Nt


    def Loss_continuous(self, vars: torch.tensor):
        dev = vars[1:].__sub__(vars[:-1])
        dev = dev.__div__(self.dt)
        return dev.__abs__().sum()

    ## SOTA ##################################################

    def SOTA_energy_loss(self, vars: torch.tensor):
        u1_velocity = vars[:, 0]
        u2_temp = vars[:, 1]

        w1, w2 = 30.0, 3.0
        u2_default = 14.4
        
        loss_energy = w1 * u1_velocity.abs().mean() + w2 * (u2_temp - u2_default).abs().mean()

        return loss_energy

    def SOTA_constraint_loss(self, temperature_field: CenteredGrid):
        T_min, T_max = 21.0, 22.0
        
        temp_mean = field.mean(temperature_field)
        violation_low = math.maximum(0.0, T_min - temp_mean)
        violation_high = math.maximum(0.0, temp_mean - T_max)
        
        loss_constraint = violation_low ** 2 + violation_high ** 2
        return loss_constraint 
    
    def SOTA_centering_loss(self, temperature_field: CenteredGrid):
        T_min, T_max = 21.0, 22.0
        T_middle = 0.5 * (T_min + T_max)
        
        temp_mean = field.mean(temperature_field)
        loss_centering = (temp_mean - T_middle) ** 2
        return loss_centering

    def SOTA_CO2_loss(self, co2_field: CenteredGrid):
        C_max = 1200.0  # ppm - maximum allowed CO2 concentration

        # Maximum CO2 concentration in the field
        co2_max = math.max(co2_field.values)

        # ReLU penalty: penalize violation above C_max
        # max(0, co2_max - C_max) using PhiFlow math
        violation = math.maximum(0.0, co2_max - C_max)
        loss = violation ** 2

        return loss

