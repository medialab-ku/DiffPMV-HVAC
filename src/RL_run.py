from src.config import cfg, log, result_folder
import src.RL

from stable_baselines3 import PPO
import gymnasium as gym
from gymnasium.wrappers import RescaleAction
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np
from tqdm import tqdm
from pathlib import Path


def make_env(params):
    env = gym.make("RoomEnv-v1", params=params)
    env = RescaleAction(env, min_action=-1.0, max_action=1.0)
    return env


def train():
    params = {
        "init_action": np.array([0.6, 20.0, np.deg2rad(135.0)] + ([0.0] if cfg.control_vars.shape[1] == 4 else []), dtype=np.float32),
        "warm_start": 0
    }

    # Create environment and wrap with VecNormalize
    env = DummyVecEnv([lambda: make_env(params)])  # Wrap for VecNormalize

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1080,  
        batch_size=270,
        learning_rate=3e-4,   
        ent_coef=0.02,        
        clip_range=0.2,       
        max_grad_norm=0.5,    
        gamma=0.99,  
        n_epochs=10,  
        vf_coef=0.5,  
        verbose=1,
        seed=0,
        tensorboard_log="./ppo_tensorboard/",
    )

    checkpoint = CheckpointCallback(
        save_freq = 1800,
        save_path = str(Path(result_folder) / "RL"),
        name_prefix = "ppo_pmv",
    )

    new_logger = configure("./ppo_tensorboard/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)

    model.learn(total_timesteps=43_200, log_interval=1, callback=checkpoint)

    # Save both model and VecNormalize stats
    model.save(str(Path(result_folder) / "RL" / "ppo_pmv_final"))

    env.close()


def evaluate():
    print("Evaluation Start:")

    params = {
        "init_action": np.array([0.6, 20.0, np.deg2rad(135.0)] + ([0.0] if cfg.control_vars.shape[1] == 4 else []), dtype=np.float32),
        "warm_start": 0
    }

    # Load environment with normalization stats
    env = DummyVecEnv([lambda: make_env(params)])

    model = PPO.load(str(Path(result_folder) / "RL" /"ppo_pmv_final"))
    obs = env.reset()
    Nt = int(env.get_attr("Nt")[0])

    # actions = np.empty((360, 3), dtype=np.float32)
    actions = np.zeros((Nt, 3), dtype=np.float32)

    for i in tqdm(range(Nt)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, dones, infos = env.step(action)

        actions[i] = infos[0]["action"]

        if dones[0]:
            break

    np.save(str(Path(result_folder) / "RL" / "ppo_room_actions.npy"), actions)
    env.close()




if __name__ == "__main__":
    train()
    evaluate()
