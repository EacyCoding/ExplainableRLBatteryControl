# TODO: dqn zum laufen bringen (gab ein paper, mit open source code)

# examples/dqn_simple.py


import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import DQN
from citylearn.citylearn import CityLearnEnv

# 1) Dataset-Pfade
BASE_DIR    = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_DIR = os.path.join(
    BASE_DIR,
    "data", "datasets",
    "citylearn_challenge_2023_phase_2_local_evaluation"
)
SCHEMA_PATH = os.path.join(DATASET_DIR, "schema.json")

# 2) CityLearnEnv nur für Gebäude #1
base_env = CityLearnEnv(
    schema=SCHEMA_PATH,
    data_path=DATASET_DIR,
    buildings=[1],       # nur Building 1
    central_agent=True,  # ein einziger Agent
    solar_penalty=None,
    cost_function=None
)

# 3) Wrapper: Observation flattenen (Liste → 1D-Array)
class FlattenObs(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        # env.observation_space ist Liste von Boxen
        shapes = [s.shape[0] for s in env.observation_space]
        self._length = sum(shapes)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._length,), dtype=np.float32
        )

    def observation(self, obs):
        # obs ist Liste von Arrays
        flat = np.concatenate(obs, axis=0)
        return flat.astype(np.float32)

# 4) Wrapper: nur erste Act-Dimension als Discrete(n)
class FirstDimAction(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        # env.action_space ist Liste von MultiDiscrete
        md = env.action_space[0]
        # wir steuern hier nur dim 0, dessen Anzahl = md.nvec[0]
        self.n = int(md.nvec[0])
        self.action_space = spaces.Discrete(self.n)

    def action(self, a: int):
        # Erzeuge für alle dims Null-Aktion, setze nur die erste Dim
        full = np.zeros_like(self.env.action_space[0].nvec, dtype=int)
        full[0] = a
        # CityLearn erwartet Liste mit einem Array pro Gebäude
        return [full]

# 5) kombiniertes Env
env = FirstDimAction(FlattenObs(base_env))

# 6) Smoke-Test
obs, _ = env.reset()
a = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(a)
print(f"Random action {a} → reward {reward}")

# 7) Trainiere DQN kurz
model = DQN(
    policy="MlpPolicy",
    env=env,
    learning_rate=1e-3,
    gamma=0.99,
    verbose=1
)

model.learn(total_timesteps=10_000)
model.save("dqn_citylearn_onebuilding_simple")
print("Finished simple DQN test!")
