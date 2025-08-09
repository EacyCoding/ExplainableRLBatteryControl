import json
from typing import Tuple
import pandas as pd
import numpy as np
from citylearn.citylearn import CityLearnEnv
from citylearn.data import DataSet

class RBCBatteryAgent:
    def __init__(self, thresholds, obs_keys, action_space):
        self.thresholds = thresholds
        self.obs_keys = obs_keys
        self.action_space = action_space
        self.rules = []
        self.history = []

    def add_rule(self, condition_fn, action_str):
        self.rules.append((condition_fn, action_str))

    def act(self, observations) -> str:
        for condition, action in self.rules:
            if condition(observations):
                return action
        return "idle"

    def save(self, file_path="rbc_agent.json"):
        state = {
            'rules': [(None, act) for (_, act) in self.rules],  # serialize only action strings
            'thresholds': self.thresholds,
            'obs_keys': self.obs_keys,
            'action_space': self.action_space,
            'history': self.history
        }
        with open(file_path, "w") as f:
            json.dump(state, f, indent=2)

    def load(self, file_path="rbc_agent.json"):
        with open(file_path, "r") as f:
            state = json.load(f)
        self.thresholds = state.get('thresholds', {})
        self.obs_keys = state.get('obs_keys', [])
        self.action_space = state.get('action_space', [])
        self.history = state.get('history', [])
        # Reconstruct rules placeholders (user should re-add rules separately if needed)
        self.rules = [(None, act) for (_, act) in state.get('rules', [])]

    def evaluate(self, env) -> float:
        env.reset()
        total_reward = 0.0
        terminated = False
        truncated = False

        while not (terminated or truncated):
            flat  = env.observations[0]
            names = env.observation_names[0]
            obs   = dict(zip(names, flat))

            action = self.act(obs)
            obs_list, reward, terminated, truncated, _ = env.step([action])
            self.history.append((obs, action, reward))
            total_reward += reward

        return total_reward

    def evaluate_multiple(self, env, n_episodes) -> Tuple[float, float]:
        rewards = []
        for _ in range(n_episodes):
            rewards.append(self.evaluate(env))
        return np.mean(rewards), np.std(rewards)

def main():
    # Load CityLearn environment
    schema = DataSet().get_schema('citylearn_challenge_2023_phase_3_1')
    env = CityLearnEnv(schema)

    # Initialize agent
    thresholds = {'price': None, 'net_load': 2.0, 'solar_generation': 1.0}
    obs_keys = ['soc', 'price', 'net_load', 'solar_generation']
    action_space = ['charge', 'idle', 'discharge']
    agent = RBCBatteryAgent(thresholds, obs_keys, action_space)

    # Define thresholds dynamically (e.g., daily average price)
    prices_first_day = env.buildings[0].pricing.electricity_pricing[:24]
    thresholds['price'] = np.mean(prices_first_day)

    # Add rules
    agent.add_rule(
        lambda obs: obs['soc'] < 0.9 and obs['price'] < thresholds['price'] and obs['net_load'] < thresholds['net_load'],
        "charge"
    )
    agent.add_rule(
        lambda obs: obs['soc'] < 0.9 and obs['solar_generation'] > thresholds['solar_generation'] and obs['net_load'] < thresholds['net_load'],
        "charge"
    )
    agent.add_rule(
        lambda obs: obs['price'] > thresholds['price'] and obs['net_load'] > thresholds['net_load'] and obs['soc'] > 0.1,
        "discharge"
    )
    agent.add_rule(
        lambda obs: obs['soc'] > 0.9 and obs['price'] > thresholds['price'],
        "discharge"
    )
    agent.add_rule(
        lambda obs: obs['net_load'] > thresholds['net_load'] and obs['solar_generation'] > thresholds['solar_generation'] and obs['price'] > thresholds['price'],
        "discharge"
    )
    agent.add_rule(
        lambda obs: obs['net_load'] < thresholds['net_load'] and obs['price'] > thresholds['price'] and obs['solar_generation'] < thresholds['solar_generation'] and obs['soc'] < 0.1,
        "idle"
    )

    # Run simulation for one episode
    total_reward = agent.evaluate(env)
    print(f"Total reward (1 episode): {total_reward:.2f}")

    # Run multiple evaluations
    mean_reward, std_reward = agent.evaluate_multiple(env, n_episodes=20)
    print(f"Average reward over 20 episodes: {mean_reward:.2f} ± {std_reward:.2f}")

    # Log history to CSV
    logs = []
    for t, (obs, action, reward) in enumerate(agent.history):
        entry = {'timestep': t, **obs, 'action': action, 'reward': reward}
        logs.append(entry)
    df = pd.DataFrame(logs)
    df.to_csv('rbc_agent_logs.csv', index=False)
    print("Logs saved to 'rbc_agent_logs.csv'")

    # Save agent state
    agent.save()
    print("Agent state saved to 'rbc_agent.json'")

if __name__ == "__main__":
    main()
