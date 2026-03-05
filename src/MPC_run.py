from pathlib import Path
from src import *
from src.config import cfg, log
from src.env import Env
from src.MPC import PMVExplicitMPC, DynParams, CostParams, Bounds, PMVParams, MPCConfig
from src.forward import Forward



def save_pmv_graph(pmv_values:dict[Occ, list], env:Env,  output_path):
    import matplotlib.pyplot as plt

    steps = np.arange(1, env.Nt+1)
    plt.figure(figsize=(12, 5))  

    for occ in env.occs:
        plt.plot(steps, pmv_values[occ], linestyle='-', label=f"met: {occ.met}, clo: {occ.clo}") 

    plt.fill_between(steps, 0.5, 4.0, color='gray', alpha=0.5)

    plt.title("Weighted Average absolute PMV in ROIs")  
    plt.xlabel("Simulation steps")  
    plt.ylabel("Weighted Average absolute PMV")  
    plt.ylim(0, 4.0)
    plt.xticks(np.arange(0, env.Nt+1, 60))
    plt.xlim(1, env.Nt)
    plt.legend() 
    # plt.grid(True) 
    
    plt.savefig(output_path, dpi=300)  
    plt.close() 

env = Env.from_yaml(Path(cfg.scenario), control_vars=cfg.control_vars, lr=cfg.lr)

dyn = DynParams(
    # ========== RC Thermal Model Parameters ==========
    # These should be identified from CFD data via system identification!
    # Example method: Run CFD with various (u, ut) inputs, record T response,
    #                 fit parameters via least-squares regression.
    #
    # Current values are heuristic initial guesses.

    a_T=0.92,      # thermal retention (0 = instant equilibrium, 1 = no decay)
    b_u=0.01,      # velocity mixing effectiveness
    b_ut=0.05,     # supply temperature influence
    b_uT=0.02,     # bilinear heat transfer coupling
    b_occ=0.01,    # heat gain per occupant [K/person]
    b_ext=0.02,    # external temperature coupling
    c_0=0.0,       # constant offset
)

cost = CostParams(
    # ========== Multi-Objective Weights ==========
    # Trade-off between thermal comfort and operational cost.
    # Tuning guide:
    #   - Increase w_pmv for better comfort (PMV → 0)
    #   - Increase w_u, w_ut for lower energy consumption
    #   - Increase w_du, w_dut for smoother control (less actuator wear)

    w_pmv=10.0,       # Thermal comfort priority (PMV penalty)
    w_u=1.0,          # Fan energy cost (velocity²)
    w_ut=1.0,         # Heating/cooling energy cost (temperature deviation)
    w_du=0.1,         # Velocity smoothness (actuator wear)
    w_dut=0.1,        # Temperature smoothness (thermal shock prevention)
    ut_ref=14.4,      # Energy-optimal supply temperature [°C]
)

bounds = Bounds()
pmv = PMVParams(
    rh=env.scene.rh,
)

cfg_mpc = MPCConfig(
    # ========== MPC Algorithm Settings ==========
    use_roi_avg=True,                       # Use occupant-weighted ROI temperature
    enforce_occ_indices=None,               # None = all occupants, or list of indices
    pmv_linearize_T0=env.scene.init_room_temp,  # Initial linearization point [°C]
    pmv_linearize_v0=0.15,                  # Initial linearization velocity [m/s]
    update_linearization=True,              # Recompute PMV coefficients each step
    use_pmv_band_constraint=False,          # False = soft penalty, True = hard constraint
    phi_nominal=0.0,                        # Fixed diffuser angle [rad] (0 = horizontal)
    cvxpy_solver="OSQP",                    # QP solver (OSQP, ECOS, GUROBI)
)

mpc = PMVExplicitMPC(
    env=env,
    horizon=12,     # 12-step horizon
    dyn=dyn,
    cost=cost,
    bounds=bounds,
    pmv=pmv,
    cfg=cfg_mpc,
)

control_vars_MPC = mpc.run()   # shape (Nt, 3) on env.device

forward_module = Forward(env)


pmv_values = {occ: [None] * env.Nt for occ in env.occs} # Dictionary
pmv_fields = {occ: [None] * env.Nt for occ in env.occs}
velocity, temperature = env.init_vf, env.init_tf
for t in tqdm(range(env.Nt), desc="Simulation Steps", unit="step"):
    velocity, temperature = forward_module.step(velocity, temperature, control_vars_MPC[t])  

    for occ in env.occs:  
        pmv = forward_module.tensor_pmv(velocity, temperature, met=occ.met, clo=occ.clo) # occupants..
        # if occ.is_active(t):
        pmv_values[occ][t] = field.l1_loss(occ.kernels.t[t] * pmv.__abs__()).native().item() / field.l1_loss(occ.kernels.t[t]).native().item()
        # pmv_fields[occ][t] = pmv
        del pmv

    # Clear GPU memory every 5 steps
    if (t + 1) % 5 == 0:
        import gc
        gc.collect()
        torch.cuda.empty_cache()

folderName = log.mf(cfg.current_time)
save_pmv_graph(pmv_values, env, f"{folderName}/PMV_graph_MPC.png")    

torch.save(control_vars_MPC, f"{folderName}/final_vars_MPC.pt")
np_control_vars = control_vars_MPC.cpu().numpy()
np.savetxt(f"{folderName}/control_vars_MPC.txt", np_control_vars, fmt='%.6f', delimiter=', ', header='step, v, t, rad, ...', comments='')

pmv_save_dict = {f"met_{occ.met}_clo_{occ.clo}": np.array(values) for occ, values in pmv_values.items()}

np.save(f"{folderName}/pmv_values.npy", pmv_save_dict)

data_to_save = []
for t in range(env.Nt):
    row = [t + 1]
    for occ in env.occs:
        row.append(pmv_values[occ][t])
    data_to_save.append(row)

np_pmv_data = np.array(data_to_save)
header_str = "step, " + ", ".join([f"met_{occ.met}_clo_{occ.clo}" for occ in env.occs])
np.savetxt(f"{folderName}/pmv_values.txt", np_pmv_data, fmt='%.6f', delimiter=', ', header=header_str, comments='')