from src import *
from src.config import cfg, log, result_folder, device
from src.env import Env
from src.forward import Forward

import matplotlib.pyplot as plt

# Utility

def save_loss_graph(losses, NOpt, output_path):
    plt.figure()
    plt.plot(range(1, NOpt+1), losses, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss over epochs")
    plt.savefig(output_path)
    plt.close()


def save_pmv_graph(pmv_values:dict[Occ, list], env:Env,  output_path):
    time_steps = np.arange(0, env.Nt)

    plt.rcParams.update({
        'font.size': 14,
        'font.family': 'times new roman',
        'axes.labelsize': 16,
        'mathtext.fontset': 'stix',
        'mathtext.rm': 'Times New Roman'
    })

    colors = [
        {'active': '#FF0000', 'inactive': '#FF6666', 'bg': '#FF9999'},
        {'active': '#0000FF', 'inactive': '#6666FF', 'bg': '#99CCFF'},
        {'active': '#008800', 'inactive': '#66BB66', 'bg': '#99DD99'},
        {'active': '#CC6600', 'inactive': '#FFAA66', 'bg': '#FFCC99'},
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.fill_between(time_steps, 0.5, 2.2, color="#3D3D3D", alpha=0.3)

    for i, occ in enumerate(env.occs):
        c = colors[i % len(colors)]
        pmv = np.array(pmv_values[occ])

        # Background spans + state text
        for start, end in occ.active_time:
            ax.axvspan(start, end, alpha=0.15, color=c['bg'], zorder=0)
            text = f"\nState {occ.id}\n{occ.met} Met, {occ.clo} Clo"
            ax.text((start + end) / 2, 2.0, text,
                    ha='center', va='center', fontsize=16, fontweight='bold', color=c['active'])

        # Build segments (active / inactive)
        active_times = sorted(occ.active_time, key=lambda x: x[0])
        segments = []
        current = 0
        for start, end in active_times:
            if current < start:
                segments.append((current, start, False))
            segments.append((start, end, True))
            current = end
        if current < env.Nt:
            segments.append((current, env.Nt, False))

        # Plot each segment
        labeled = False
        for seg_start, seg_end, is_active in segments:
            mask = (time_steps >= seg_start) & (time_steps < seg_end)
            if not np.any(mask):
                continue
            if is_active:
                label = f"State {occ.id}" if not labeled else None
                ax.plot(time_steps[mask], pmv[mask],
                        color=c['active'], linewidth=2.5, linestyle='-', label=label)
                labeled = True

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(r"$\mathcal{P}_1(t)$")
    ax.set_xlim(0, env.Nt)
    ax.set_ylim(0, 2.2)
    ax.set_xticks(np.arange(0, env.Nt + 1, 60))
    ax.grid(True, alpha=0.2, linestyle='--')
    ax.legend(loc='lower left', fontsize=15)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()


####

def run_optimization(env:Env):
    # Forward Module including Simulation / Optimization Functions
    forward_module = Forward(env)

    # make directory to store optimization data
    folderName = log.mf(cfg.current_time)
    log.mlf()

    # Optimized Variables 
    control_vars = torch.nn.Parameter(cfg.control_vars.to(device=env.device, dtype=torch.float32))
    control_vars.requires_grad = True

    # Learning rate and optimizer/lr scheduler
    lr = torch.tensor(cfg.lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([control_vars], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=env.NOpt)

    # Storage
    losses = [None] * env.NOpt
    vars_data = torch.zeros((env.NOpt, *control_vars.shape), device=device) 

    # Best
    best_epoch = 0
    best_vars = control_vars
    best_loss = float('inf')

    # ★ Functional Gradient
    sim_grad = field.functional_gradient(forward_module.optimize, wrt='control_vars', get_output=True)

    start_time = time.perf_counter()

    for epoch in range(env.NOpt):
        step_start_time = time.perf_counter()
        log.log_buffer.append(f"{epoch+1} epoch(s)")

        # Save Control Variable
        print(f"Velocity Mean: {control_vars[:, 0].mean().item()}, Maximum: {control_vars[:, 0].amax().item()}")
        vars_data[epoch] = control_vars
        torch.save(vars_data, f"{folderName}/control_vars.pt")

        # Simulation & Calculate Loss
        (loss, vf, tf), grads = sim_grad(env.init_vf, env.init_tf, control_vars)

        loss_value = loss.native().item()
        losses[epoch] = loss_value

        # Best Epoch
        if loss_value < best_loss:
            best_epoch = epoch
            best_vars = control_vars.clone()
            best_loss = loss_value
            log.log_buffer.append("best epoch was updated")

        # Control Variables Optimization
        if epoch != env.NOpt-1:
            optimizer.zero_grad(set_to_none=True)
            control_vars.grad = grads.to(control_vars.device)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():  
                control_vars[:, 0].clamp_(min=0.0) # speed
                control_vars[:, 2].clamp_(min=math.degrees_to_radians(0.0), max=math.degrees_to_radians(90.0))


        step_time = get_time(time.perf_counter() - step_start_time)
        print(f"{epoch+1} epoch Loss {loss} ({step_time})")
        log.log_buffer.append(f"{epoch+1} epoch Loss {loss} ({step_time})")
        log.log_buffer.append(f"step time: {step_time}\n\n")
        log.log_flush()

    total_time = get_time(time.perf_counter() - start_time)
    print(total_time)
    log.log_buffer.append("\nOptimization Done")
    # log.log_buffer.append(f"the best epoch is {best_epoch} / {env.NOpt} with loss: {best_loss}")
    log.log_buffer.append(f"Final loss: {best_loss} at epoch {best_epoch} / {env.NOpt}")
    log.log_buffer.append(f"total time: {total_time}")
    log.log_flush()


    ## OPTIMIZATION DONE
    ## Save Loss Data
    save_loss_graph(losses, env.NOpt, f"{folderName}/loss_graph.png")  # save loss graph over epoch


    ## SAVE BEST DATA
    np.savetxt(f"{folderName}/best_vars.txt", best_vars.cpu().detach().numpy(), fmt="%.3f")   # save best vars

    # Re-simulation of Best Epoch
    pmv_values = {occ: [None] * env.Nt for occ in env.occs} # Dictionary
    pmv_fields = {occ: [None] * env.Nt for occ in env.occs}
    velocity, temperature = env.init_vf, env.init_tf
    for t in tqdm(range(env.Nt), desc="Simulation Steps", unit="step"):
        velocity, temperature = forward_module.step(velocity, temperature, best_vars[t])  

        for occ in env.occs:  
            pmv = forward_module.tensor_pmv(velocity, temperature, met=occ.met, clo=occ.clo) # occupants..
            # if occ.is_active(t):
            pmv_values[occ][t] = field.l1_loss(occ.kernels.t[t] * pmv.__abs__()).native().item() / field.l1_loss(occ.kernels.t[t]).native().item()
            pmv_fields[occ][t] = pmv
            del pmv

    save_pmv_graph(pmv_values, env, f"{folderName}/PMV_graph_{best_epoch+1}epoch.png")    



def run_simulation(env:Env):
    folderName = log.mf(cfg.current_time)
    forward_module = Forward(env)
    # vf, tf = forward_module.simulation(env.init_vf, env.init_tf, cfg.control_vars)

    pmv_values = {occ: [None] * env.Nt for occ in env.occs} # Dictionary
    pmv_fields = {occ: [None] * env.Nt for occ in env.occs}

    velocity, temperature = env.init_vf, env.init_tf
    for t in tqdm(range(env.Nt), desc="Simulation Steps", unit="step"):
        velocity, temperature = forward_module.step(velocity, temperature, cfg.control_vars[t])  

        for occ in env.occs:  
            pmv = forward_module.tensor_pmv(velocity, temperature, met=occ.met, clo=occ.clo) # occupants..
            pmv_values[occ][t] = field.l1_loss(occ.kernels.t[t] * pmv.__abs__()).native().item() / field.l1_loss(occ.kernels.t[t]).native().item()
            del pmv

    save_pmv_graph(pmv_values, env, f"{folderName}/PMV_graph.png")  



def run_SOTA_opt(env:Env):
    # Forward Module including Simulation / Optimization Functions
    forward_module = Forward(env)

    # make directory to store optimization data
    folderName = log.mf(cfg.current_time)
    log.mlf()

    # Optimized Variables 
    control_vars = torch.nn.Parameter(cfg.control_vars.to(device=env.device, dtype=torch.float32))
    control_vars.requires_grad = True

    # Learning rate and optimizer/lr scheduler
    lr = torch.tensor(cfg.lr, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam([control_vars], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=env.NOpt)

    # Storage
    losses = [None] * env.NOpt
    vars_data = torch.zeros((env.NOpt, *control_vars.shape), device=device) 

    # Best
    best_epoch = 0
    best_vars = control_vars
    best_loss = float('inf')

    # ★ Functional Gradient
    sim_grad = field.functional_gradient(forward_module.SOTA_optimize, wrt='control_vars', get_output=True)

    start_time = time.perf_counter()

    for epoch in range(env.NOpt):
        step_start_time = time.perf_counter()
        log.log_buffer.append(f"{epoch+1} epoch(s)")

        # Save Control Variable
        # print(f"Velocity Mean: {control_vars[:, 0].mean().item()}, Maximum: {control_vars[:, 0].amax().item()}")
        vars_data[epoch] = control_vars
        torch.save(vars_data, f"{folderName}/control_vars.pt")

        # Simulation & Calculate Loss
        (loss, vf, tf), grads = sim_grad(env.init_vf, env.init_tf, control_vars)

        # PhiFlow tensor to native Python number
        loss_value = float(loss)
        losses[epoch] = loss_value

        # Best Epoch
        if loss_value < best_loss:
            best_epoch = epoch
            best_vars = control_vars.clone()
            best_loss = loss_value
            log.log_buffer.append("best epoch was updated")

        # Control Variables Optimization
        if epoch != env.NOpt-1:
            optimizer.zero_grad(set_to_none=True)
            control_vars.grad = grads.to(control_vars.device)
            optimizer.step()
            scheduler.step()

            with torch.no_grad():  
                control_vars[:, 0].clamp_(min=0.0) # speed
                control_vars[:, 2].clamp_(min=math.degrees_to_radians(0.0), max=math.degrees_to_radians(90.0))


        step_time = get_time(time.perf_counter() - step_start_time)
        print(f"{epoch+1} epoch Loss {loss} ({step_time})")
        log.log_buffer.append(f"{epoch+1} epoch Loss {loss} ({step_time})")
        log.log_buffer.append(f"step time: {step_time}\n\n")
        log.log_flush()

    total_time = get_time(time.perf_counter() - start_time)
    print(total_time)
    log.log_buffer.append("\nOptimization Done")
    # log.log_buffer.append(f"the best epoch is {best_epoch} / {env.NOpt} with loss: {best_loss}")
    log.log_buffer.append(f"Final loss: {best_loss} at epoch {best_epoch} / {env.NOpt}")
    log.log_buffer.append(f"total time: {total_time}")
    log.log_flush()


    ### OPTIMIZATION DONE

    ## Save Loss Graph
    save_loss_graph(losses, env.NOpt, f"{folderName}/loss_graph.png")  # save loss graph over epoch

    ## SAVE BEST DATA
    np.savetxt(f"{folderName}/best_vars.txt", best_vars.cpu().detach().numpy(), fmt="%.3f")   # save best vars

    # Re-simulation of Best Epoch
    pmv_values = {occ: [None] * env.Nt for occ in env.occs} # Dictionary
    pmv_fields = {occ: [None] * env.Nt for occ in env.occs}
    velocity, temperature = env.init_vf, env.init_tf
    for t in tqdm(range(env.Nt), desc="Simulation Steps", unit="step"):
        velocity, temperature = forward_module.step(velocity, temperature, best_vars[t])  

        for occ in env.occs:  
            pmv = forward_module.tensor_pmv(velocity, temperature, met=occ.met, clo=occ.clo) # occupants..
            # if occ.is_active(t):
            pmv_values[occ][t] = field.l1_loss(occ.kernels.t[t] * pmv.__abs__()).native().item() / field.l1_loss(occ.kernels.t[t]).native().item()
            pmv_fields[occ][t] = pmv
            del pmv

    save_pmv_graph(pmv_values, env, f"{folderName}/PMV_graph_SOTA_{best_epoch+1}epoch.png")    


def run_SOTA_sim(env:Env):
    folderName = log.mf(cfg.current_time)
    forward_module = Forward(env)

    temp_avg = [None] * env.Nt
    co2_max = [None] * env.Nt

    velocity, temperature = env.init_vf, env.init_tf
    C_init = 450.0  # initial CO2 concentration in ppm (from SOTA env.py line 123)
    co2 = CenteredGrid(C_init, extrapolation.ZERO_GRADIENT, x=env.Nx, y=env.Ny, z=env.Nz, bounds=env.bound)

    for t in tqdm(range(env.Nt), desc="Simulation Steps", unit="step"):
        velocity, temperature, co2 = forward_module.step_with_co2(velocity, temperature, co2, cfg.control_vars[t], t)  
        temp_avg[t] = field.mean(temperature).native().item()
        co2_max[t] = math.max(co2.values).native().item()

    np.savetxt(f"{folderName}/temperature_avg.txt", temp_avg)
    np.savetxt(f"{folderName}/co2_max.txt", co2_max)



if __name__ == "__main__":
    env = Env.from_yaml(Path(cfg.scenario), control_vars=cfg.control_vars, lr=cfg.lr)
    if   cfg.run_mode == "OPTIMIZATION":    run_optimization(env)
    elif cfg.run_mode == "SIMULATION":      run_simulation(env)
    elif cfg.run_mode == "SOTA_OPT":        run_SOTA_opt(env)
    elif cfg.run_mode == "SOTA_SIM":        run_SOTA_sim(env)
