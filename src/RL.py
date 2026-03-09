from phi.flow import *
from tqdm import tqdm
import torch.nn as nn
from phiml.math import reshaped_numpy
import seaborn as sns
import gymnasium as gym
from gymnasium import spaces
from gymnasium.envs.registration import register

from src import *
from src.config import cfg             
from src.env import Env                
from src.forward import Forward        
from src.losses import LossClass       


register(
    id='RoomEnv-v1',
    entry_point='src.PMV_RL:RLEnv',
)

class RLEnv(gym.Env):
  metadata = {"render_modes": []}

  def __init__(self, params=None):
    super().__init__()

    self.action_low = np.array([0.0, 14.0, np.deg2rad(0.0)])
    self.action_high = np.array([2.5, 26.0, np.deg2rad(90.0)])
    self.action_space = spaces.Box(low=self.action_low, high=self.action_high, shape=(3,), dtype=np.float32)

    # observation: (weighted average of |PMV|, mean temperature, v_out, t_out, phi_out, t/Nt, prev_pmv_diff)
    self.observation_low_raw = np.array([0.0, 14.0, 0.0, 14.0, np.deg2rad(0.0), 0.0, -3.0])
    self.observation_high_raw = np.array([4.0, 30.0, 3.0, 26.0, np.deg2rad(90.0), 1.0, 3.0])

    self.observation_low = np.zeros(7, dtype=np.float32)
    self.observation_high = np.ones(7, dtype=np.float32)
    self.observation_space = spaces.Box(low=self.observation_low, high=self.observation_high, shape=(7,), dtype=np.float32) 

    env = Env.from_yaml(Path(cfg.scenario), control_vars=cfg.control_vars, lr=cfg.lr)  
    self.forward = Forward(env)
    self.lossClass = LossClass(env)

    self.env = env
    self.device = env.device

    self.Nx, self.Ny, self.Nz = env.Nx, env.Ny, env.Nz
    self.init_room_temp = env.init_room_temp
    self.Nt = env.Nt
    self.rh = env.rh
    self.bound = env.bound
    self.outlet = env.outlet
    self.obstacles = env.obstacles
    self.occs = env.occs

    self.record_actions = [] # for evaluation
    self.prev_pmv = 0.0    # track previous pmv
    self.prev_action = None

    self.user_init_action = None
    self.warm_start = 0
    if params is not None:
        if "init_action" in params and params["init_action"] is not None:
            a0 = np.asarray(params["init_action"], dtype=np.float32)
            self.user_init_action = np.clip(a0, self.action_low, self.action_high)
        if "warm_start" in params and params["warm_start"] is not None:
            self.warm_start = int(params["warm_start"])

    

  def _roi_weighted_mean(self, roi: CenteredGrid, value_grid: CenteredGrid):
        """
            (∫ ROI * value) / (∫ ROI)
        """
        num = field.l1_loss(roi * value_grid)
        den = field.l1_loss(roi) + 1e-8
        return num / den

  def _roi_weighted_mean_abs(self, roi: CenteredGrid, pmv_grid: CenteredGrid):
        """
            ROI-weighted mean of |PMV|
        """
        return self._roi_weighted_mean(roi, pmv_grid.__abs__())

  def _roi_weighted_max_abs(self, roi: CenteredGrid, pmv_grid: CenteredGrid):
        """
            ROI-based max proxy: max(ROI * |PMV|)
        """
        roi_abs = roi * pmv_grid.__abs__()
        return self._to_float(math.max(roi_abs.values))


  def step(self, action):
    t = self.t
    a = np.clip(np.asarray(action, dtype=np.float32), self.action_low, self.action_high)
    self.record_actions.append(a.copy())

    step_vars = torch.tensor(a, dtype=torch.float32, device=self.device)
    self.velocity, self.temperature = self.forward.step(self.velocity, self.temperature, step_vars)

    PMV_limit = 0.5
    gamma_pmv_hard = 40.0
    gamma_pmv_soft = 15.0 
    w1, w2 = 1.0, 0.2  
    u2_default = 14.4   # Supply air temperature default
    gamma_continuous = 0.01

    occs_active = [occ for occ in self.occs if occ.is_active(t)]

    if len(occs_active) == 0:
        pmv_avg = 0.0
        pmv_max_roi = 0.0
        T_roi_mean = self._to_float(self.temperature.values.mean)
    else:
        total_w = 0.0
        pmv_avg_acc = 0.0
        T_roi_acc = 0.0
        pmv_max_roi = 0.0

        for occ in occs_active:
            roi = occ.kernels.t[t]

            pmv = self.forward.tensor_pmv(self.velocity, self.temperature, rh=self.rh, met=occ.met, clo=occ.clo)

            occ_pmv_avg = self._to_float(self._roi_weighted_mean_abs(roi, pmv))
            occ_T_roi = self._to_float(self._roi_weighted_mean(roi, self.temperature))

            pmv_avg_acc += occ.weight * occ_pmv_avg
            T_roi_acc += occ.weight * occ_T_roi
            total_w += occ.weight

            pmv_max_roi = max(pmv_max_roi, self._roi_weighted_max_abs(roi, pmv))

        pmv_avg = pmv_avg_acc / max(total_w, 1e-6)
        T_roi_mean = T_roi_acc / max(total_w, 1e-6)


    # PMV Cost in ROI
    pmv_violation = max(0.0, pmv_max_roi - PMV_limit)
    pmv_hard = gamma_pmv_hard * (pmv_violation ** 2)
    pmv_soft = gamma_pmv_soft * pmv_avg

    # Energy Cost 
    # w1 * |u1| + w2 * |u2 - u2_default|
    energy_cost = w1 * float(a[0]) + w2 * abs(float(a[1]) - u2_default)

    reward = - (pmv_hard + pmv_soft + energy_cost)


    smoothness_cost = 0.0
    if self.prev_action is not None:
        smoothness_cost = gamma_continuous * np.abs(a-self.prev_action).sum() / self.env.dt
    reward -= smoothness_cost

    # Reward scaling for learning stability
    reward = reward / 1000.0

    # Observation
    v_out, t_out, phi_out = self._to_float(a[0]), self._to_float(a[1]), self._to_float(a[2])
    phase = self._to_float(self.t / self.Nt)
    pmv_diff = float(pmv_avg - self.prev_pmv)

    observation_raw = np.array([
        pmv_avg,
        T_roi_mean,
        v_out,
        t_out,
        phi_out,
        phase,
        pmv_diff
    ], dtype=np.float32)

    observation_raw = np.clip(observation_raw, self.observation_low_raw, self.observation_high_raw)
    observation = self._normalize_observation(observation_raw)

    # Clip observation to bounds
    observation = np.clip(observation, self.observation_low, self.observation_high)

    info = {
        "action": a,
        "t": self.t,
        "pmv_roi_avg_abs": pmv_avg,
        "pmv_roi_max_abs": pmv_max_roi,
        "pmv_violation": pmv_violation,
        "T_roi_mean": T_roi_mean,
        "energy_cost": energy_cost,
        "pmv_hard": pmv_hard,
        "pmv_soft": pmv_soft,
        "smooth_cost": smoothness_cost,
        "reward_raw": reward * 1000.0,
        "reward_scaled": reward,
    }

    self.prev_pmv = self._to_float(pmv_avg)
    self.prev_action = a.copy()

    self.t = t + 1
    terminated = self.t >= self.Nt

    return observation, reward, terminated, False, info


  def reset(self, seed=None, options=None):
    super().reset(seed=seed)

    self.velocity    = StaggeredGrid(0, extrapolation.ZERO, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)
    self.temperature = CenteredGrid(self.init_room_temp, extrapolation.BOUNDARY, x=self.Nx, y=self.Ny, z=self.Nz, bounds=self.bound)
    
    self.t = 0
    self.record_actions = []

    if self.user_init_action is not None:
        init_action = self.user_init_action.copy()
    else:
        init_action = np.array([0.6, 20.0, np.deg2rad(30.0)], dtype=np.float32)

    init_action = init_action + np.random.uniform(-0.05, 0.05, size=3).astype(np.float32)
    init_action = np.clip(init_action, self.action_low, self.action_high)

    self.prev_action = init_action.copy()

    occs_active = [occ for occ in self.occs if occ.is_active(self.t)]
    if len(occs_active) == 0:
        self.prev_pmv = 0.0
        T_roi_mean = self._to_float(self.temperature.values.mean)
    else:
        total_w = 0.0
        pmv_acc = 0.0
        T_roi_acc = 0.0
        for occ in occs_active:
            roi = occ.kernels.t[0]
            pmv0 = self.forward.tensor_pmv(V=self.velocity, T=self.temperature, rh=self.rh, met=occ.met, clo=occ.clo)
            occ_pmv = self._to_float(self._roi_weighted_mean_abs(roi, pmv0))
            occ_T = self._to_float(self._roi_weighted_mean(roi, self.temperature))
            pmv_acc += occ.weight * occ_pmv
            T_roi_acc += occ.weight * occ_T
            total_w += occ.weight
        self.prev_pmv = pmv_acc / max(total_w, 1e-8)
        T_roi_mean = T_roi_acc / max(total_w, 1e-8)

    observation_raw = np.array([
        self.prev_pmv, 
        T_roi_mean, 
        float(init_action[0]), 
        float(init_action[1]), 
        float(init_action[2]), 
        0.0, 
        0.0
    ], dtype=np.float32)

    observation_raw = np.clip(observation_raw, self.observation_low_raw, self.observation_high_raw)
    observation = self._normalize_observation(observation_raw)

    return observation, {}



  def render(self):
    pass 
  
  def _to_float(self, t):
        try:
            return float(t)
        except Exception:
            if hasattr(t, "native"):
                n = t.native()
                try:
                    return float(n.item())
                except Exception:
                    return float(np.array(n).mean())
            return float(t)
        

  def _normalize_observation(self, obs_raw):
        """
        Normalize raw observation from [low_raw, high_raw] to [0, 1]
        
        Formula: (x - x_min) / (x_max - x_min)
        """
        obs_range = self.observation_high_raw - self.observation_low_raw
        obs_range = np.where(obs_range < 1e-8, 1.0, obs_range)  # Avoid division by zero
        
        obs_normalized = (obs_raw - self.observation_low_raw) / obs_range
        obs_normalized = np.clip(obs_normalized, 0.0, 1.0)  # Ensure [0, 1] range
        
        return obs_normalized.astype(np.float32)