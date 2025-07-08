import time
import numpy as np
from pathlib import Path
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import StableBaselines3Wrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy

# Create the environment
def make_env():
    env = CityLearnEnv(
        schema="citylearn_challenge_2023_phase_2_local_evaluation", # welcher Datensatz
        central_agent=True, # single agent for all buildings instead of one agent per building
        buildings=None, # None means all buildings in the schema
        solar_penalty=None, # keine strafe für Solarüberschuss
        cost_function=None # Standard cost function
    )
    return StableBaselines3Wrapper(env) # Wandelt die CityLearn-Umgebung so um, dass Stable-Baselines3 sie versteht

# Vectorize the train_env (SB3 agents expect a VectorEnv)
train_env = DummyVecEnv([make_env]) # hier: nur ein Env, aber SB3 kann mehrere parallel ausführen, deshalb das Format Vektor-Environment

# Eval-Env (ohne Exploration)
eval_env  = DummyVecEnv([make_env]) #  identische Umgebung wie train_env, nur dass SB3 beim Evaluieren keine zufällige Exploration einsetzt.

class CustomEvalCallback(BaseCallback):
    def __init__(self,
                 eval_env,
                 eval_freq: int = 5_000,
                 n_eval_episodes: int = 5,
                 best_model_save_path: str = "./logs/best_model",
                 verbose: int = 1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_model_save_path = best_model_save_path
        self.best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        # führe alle eval_freq Schritte eine Evaluation durch
        if self.n_calls % self.eval_freq == 0:
            # 1) Rewards und Längen holen
            rewards, lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                return_episode_rewards=True,
                warn=False
            )
            mean_reward = np.mean(rewards)
            mean_length = np.mean(lengths)

            # 2) Individual‐Infos aus dem Env abgreifen
            #    (Voraussetzung: CityLearnEnv gibt bei done info['energy_cost'] und info['comfort_violation'])
            energy_costs = []
            comfort_violations = []
            for _ in range(self.n_eval_episodes):
                # Wir resetten das eval_env, um an das letzte info‐Dict zu kommen
                _, _, done, infos = self.eval_env.step(self.model.predict(self.eval_env.reset(), deterministic=True)[0])
                # infos kann eine Liste sein, je nach VecEnv, wir nehmen das erste Element
                info = infos[0] if isinstance(infos, (list, tuple)) else infos
                energy_costs.append(info.get("energy_cost", np.nan))
                comfort_violations.append(info.get("comfort_violation", np.nan))

            mean_energy = np.nanmean(energy_costs)
            mean_comfort = np.nanmean(comfort_violations)

            # 3) In TensorBoard loggen
            self.logger.record("eval/mean_reward", mean_reward)
            self.logger.record("eval/mean_length", mean_length)
            self.logger.record("eval/mean_energy_cost", mean_energy)
            self.logger.record("eval/mean_comfort_violation", mean_comfort)
            self.logger.dump(self.num_timesteps)

            # 4) Bestes Modell speichern
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save(Path(self.best_model_save_path) / "best_model")

        return True

# Eval-Callback konfigurieren
custom_eval_callback  = CustomEvalCallback(
    eval_env=eval_env,
    eval_freq=5_000,                           # alle 5.000 Schritte evaluieren
    n_eval_episodes=3,                         # über 3 vollständige Episoden mitteln
    best_model_save_path="./logs/best_model",  # speichert das Modell mit der höchsten Eval-Performance
    verbose=1
)
"""
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./logs/best_model",  # speichert das Modell mit der höchsten Eval-Performance
    log_path="./logs/results",                 # speichert die Metriken (episode_reward, etc.)
    eval_freq=5_000,                           # alle 5.000 Schritte evaluieren
    n_eval_episodes=5,                         # über 5 vollständige Episoden mitteln
    deterministic=True,                        # Policy deterministisch ausführen
    render=False
)
"""

# Define the RL agent
model = PPO(
    policy="MlpPolicy", # Multi-Layer-Perceptron (klassisches Feed-Forward-Netzwerk) als Policy-Netz
    env=train_env, # die Umgebung, in der der Agent trainieren soll
    verbose=1, # Steuerung der Ausgabe-Häufigkeit: 0 → komplett still, 1 → Trainingsschritte in der Konsole, 2 → noch detailliertere Logs
    learning_rate=3e-4, # Die Schrittweite beim Update der Netz-Gewichte. Ein höherer Wert lässt das Netz schneller lernen, kann aber auch zu Instabilität führen.
    gamma=0.99, # Der Discount-Faktor für zukünftige Belohnungen. γ=0 → nur immediate Rewards zählen, γ≈1 → weit in die Zukunft schauen, Hier 0.99 heißt: Belohnungen in 100 Schritten sind noch um 0.99¹⁰⁰≈0.37 gewichtet.
    batch_size=64, # Größe der Mini-Batches pro Gradientenschritt. Kleinere Batches können zu stabilerem Lernen führen, aber auch langsamer sein.
    n_steps=2048, # Anzahl der Schritte pro Rollout, bevor ein Policy-Update stattfindet.
    ent_coef=0.01, # Entropie-Koeffizient für die Entropie-Belohnung, die Exploration fördert. Höhere Werte fördern mehr Exploration, niedrigere Werte mehr Exploitation.
    vf_coef=1.0,              # (statt Default 0.5) Value-Netz wird stärker gewichtet
    # b) Separate Netz-Architekturen für Policy und Value
    policy_kwargs={
        "net_arch": [
            # Gemeinsamer Backbone
            {"pi": [64, 64],      # Policy-Zweig: zwei Layer mit je 64 Neuronen
             "vf": [256, 256]}    # Value-Zweig: zwei Layer mit je 256 Neuronen
        ]
    },
     # c) GAE-Parameter (für stabilere Advantage-Schätzung)
    gae_lambda=0.9,           # (statt 0.95) etwas weniger Bias in der TD-Schätzung
    normalize_advantage=True,  # normiert die Advantages auf Varianz=1, macht Updates stabiler
    # d) Gradient Clipping
    max_grad_norm=0.5,         # verhindert zu große Gradienten im Value-Netz
    tensorboard_log="./tensorboard_logs/" # ← hier die Logs hinschreiben
)

# Start training
TIMESTEPS = 20_000
start = time.time()
model.learn( # führt während des Trainings automatisch die Evaluation durch und unterbricht nicht den Trainingsfluss.
    total_timesteps=TIMESTEPS,
    callback=custom_eval_callback
)
print(f"Training took {time.time() - start:.1f} seconds")


# Save the trained model
model.save("ppo_citylearn_sb3_improved ")
