# Normalize observation space features to mean 0 and std 1
"""
    # ——————————————————————————————
    # Normalize each feature to mean 0 and std 1 (better than squashing everything between 0 and 1)
    #—————————————————————————————————
    # 0) month (1–12) → [0,1]
    x[0] = (x[0] - 1) / 11.0

    # 1) hour (1–24)
    x[1] = (x[1]-1) / 23.0

    # 2) day_type (1–7)
    x[2] = (x[2] - 1) / 6.0

    # 3) daylight_savings_status (0/1) – bleibt 0–1
    x[3] = x[3]

    # #4) indoor_dry_bulb_temperature per Z-Score
    x[4] = (x[4] - self.temp_mean) / self.temp_std


    # 5) average_unmet_cooling_setpoint_difference (–10…+10)
    x[5] = (x[5] + 10.0) / 20.0

    # 6) indoor_relative_humidity (%) 0–100
    x[6] = x[6] / 100.0

    # 7) non_shiftable_load (kWh, z.B. 0–20)
    x[7] = x[7] / 20.0

    # 8) dhw_demand (kWh, 0–20)
    x[8] = x[8] / 20.0

    # 9) cooling_demand (kWh, 0–20)
    x[9] = x[9] / 20.0

    # 10) heating_demand (kWh, 0–20)
    x[10] = x[10] / 20.0

    # 11) solar_generation (W/kW, 0–1000)
    x[11] = x[11] / 1000.0

    # 12) occupant_count (people, 0–10)
    x[12] = x[12] / 10.0

    # 13) temperature_set_point (°C, z.B. 10–30)
    x[13] = (x[13] - 10.0) / 20.0

    # 14) hvac_mode (0=Off,1=Cool,2=Heat)
    x[14] = x[14] / 2.0

    # 15) outdoor_dry_bulb_temperature (–20…+50°C)
    x[15] = (x[15] + 20.0) / 70.0

    # 16) outdoor_relative_humidity (%) 0–100
    x[16] = x[16] / 100.0

    # 17) diffuse_solar_irradiance (W/m2, 0–1000)
    x[17] = x[17] / 1000.0

    # 18) direct_solar_irradiance (W/m2, 0–1000)
    x[18] = x[18] / 1000.0

    # 19) kg_CO2/kWh (0–1)
    x[19] = x[19] / 1.0

    # 20) Electricity Pricing [$/kWh] (0–1)
    x[20] = x[20] / 1.0
    # ——————————————————————————————
"""

# Obs dict for reward function SolarPenaltyReward
"""
    net = float(self.bld.loc[self.t, 'non_shiftable_load']) - frac * self.capacity
    obs_dict = {
        # für SolarPenaltyReward
        'net_electricity_consumption': net,
        'electricity_pricing':         float(self.prc.loc[self.t, 'Electricity Pricing [$/kWh]']),
        'kg_CO2/kWh':                  float(self.car.loc[self.t, 'kg_CO2/kWh']),
        # falls du ComfortReward nutzen möchtest, brauchst du zusätzlich:
        #'hvac_mode':                           int(self.bld.loc[self.t, 'hvac_mode']),
        #'indoor_dry_bulb_temperature':        float(self.bld.loc[self.t, 'indoor_dry_bulb_temperature']),
        #'indoor_dry_bulb_temperature_cooling_set_point':  float(self.bld.loc[self.t, 'temperature_set_point']),
        #'indoor_dry_bulb_temperature_heating_set_point':  float(self.bld.loc[self.t, 'temperature_set_point']),
        #'comfort_band':                       float(self.bld.loc[self.t, 'comfort_band']),  # falls vorhanden
        #'cooling_demand':                    float(self.bld.loc[self.t, 'cooling_demand']),
        #'heating_demand':                    float(self.bld.loc[self.t, 'heating_demand']),
    }
"""

# when tuned:
"""
# Save the trained model
model.save("dqn_model")

# Save the VecNormalize‑Wrapper with all its Running‑Stats
train_env.save("vecnormalize.pkl")
"""

class TrainLoggerCallback(BaseCallback):
    """Logging State, Action, Reward per step and Loss per update phase."""
    def __init__(self, verbose=0, n_envs: int = 1):
        super().__init__(verbose)
        # State/Action/Reward per Timestep
        self.rows = []
        # Loss for each internal update phase
        self.losses = []
        self.loss_timesteps = []
         # Episode returns
        self.episode_rewards = []
        self._current_ep_rewards = [0.0] * n_envs

        # DataFrames for storing results
        self.df = pd.DataFrame() # timestep rows
        self.ep_df = pd.DataFrame() # episodic returns

    def _on_step(self) -> bool:
        obs_vec    = self.locals.get("new_obs")
        actions    = self.locals.get("actions")
        rewards    = self.locals.get("rewards")
        dones      = self.locals.get("dones")
        step       = int(self.num_timesteps)

        obs = obs_vec[0].flatten().tolist()
        action = int(actions[0])
        reward = float(rewards[0])

        # Store state/action/reward
        row = { **{f"x{i}": obs[i] for i in range(len(obs))},
                "action": action,
                "reward": reward,
                "step": step }
        self.rows.append(row)

        # SB3 writes train/loss automatically into self.logger.name_to_value
        current_loss = self.logger.name_to_value.get("train/loss")
        print(f"Step {step}: Action {action}, Reward {reward}, Loss {current_loss}")
        if current_loss is not None:
            self.losses.append(float(current_loss))
            self.loss_timesteps.append(step)

        # Episodic Reward
        for idx, (r, d) in enumerate(zip(rewards, dones)):
            self._current_ep_rewards[idx] += float(r)
            if d:
                # Episode finished in sub-env idx
                print(f"Sub-env {idx} done at step {step}, total reward: {self._current_ep_rewards[idx]:.3f}")
                self.episode_rewards.append(self._current_ep_rewards[idx])
                # Reset counter
                self._current_ep_rewards[idx] = 0.0

        return True

    def _on_training_end(self) -> None:
        # Convert timestep rows to DataFrame
        self.df = pd.DataFrame(self.rows)
        
        # 2) DataFrame for episodic Returns
        self.ep_df = pd.DataFrame({
            "episode": range(1, len(self.episode_rewards) + 1),
            "return": self.episode_rewards
        })
        super()._on_training_end()



# --------------------------------------------------------
### (Right now Unnecessary) Visualizations 
# --------------------------------------------------------
# Heatmap of feature correlations
real_labels = [
    'month','hour','day_type','daylight_savings_status',
    'indoor_dry_bulb_temperature',
    'average_unmet_cooling_setpoint_difference',
    'indoor_relative_humidity',
    'non_shiftable_load','dhw_demand',
    'cooling_demand','heating_demand',
    'solar_generation','occupant_count',
    'temperature_set_point','hvac_mode',
    'outdoor_dry_bulb_temperature',
    'outdoor_relative_humidity',
    'diffuse_solar_irradiance','direct_solar_irradiance',
    'kg_CO2/kWh','Electricity Pricing [$/kWh]'
]
mapping    = {f"x{i}": real_labels[i] for i in range(len(real_labels))}
mapping.update({"action":"action","reward":"reward"})
train_df.rename(columns=mapping, inplace=True)

# Take a subset of the first 1000 steps for correlation analysis
sub = train_df.iloc[:1000]
corr = sub.corr()

plt.figure(figsize=(12,10))
sns.heatmap(
    corr,
    cmap="RdBu_r",
    center=0
)
plt.title("Feature ↔ Feature, Action & Reward Correlation\n(first 1000 steps)")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
# Which features are correlated and typically change together?


# --------------------------------------------------------

# Plot one feature vs action and reward
plt.figure(figsize=(6,2))
plt.scatter(train_df["average_unmet_cooling_setpoint_difference"], train_df["action"], c=train_df["reward"], cmap="viridis", s=10)
plt.colorbar(label="Reward")
plt.xlabel("Feature x5")
plt.ylabel("Action")
plt.show()

# --------------------------------------------------------

# Parallel Coordinates Plot: color by target action
obs_cols = train_env.get_attr("obs_cols")[0] # feature names
rename_map = { f"x{i}": obs_cols[i] for i in range(len(obs_cols)) }
df_named = train_df.rename(columns=rename_map)

# 200 random samples from the training DataFrame
sample = df_named.sample(200, random_state=0).copy()
# Convert action to string for coloring
sample["action"] = sample["action"].astype(str)

plt.figure(figsize=(12,6))
parallel_coordinates(
    sample,
    class_column="action",
    #cols=[f"x{i}" for i in range(len(sample.columns) - 4)],  # x0…xN
    cols=obs_cols,
    alpha=0.3,
)
plt.title("State Vector colored by Action")
plt.ylabel("Normalized Feature Value")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Action", bbox_to_anchor=(1.05,1), loc="upper left")
plt.show()

# --------------------------------------------------------

# Currently broken
"""
from sklearn.decomposition import PCA

# Mapping von x0…xN auf die echten Feature-Namen
mapping = { f"x{i}": obs_cols[i] for i in range(len(obs_cols)) }
df_renamed = train_df.rename(columns=mapping)

# Features x0…xN
X = df_renamed[[c for c in df_renamed.columns if c.startswith("x")]].values
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)

plt.figure(figsize=(6,5))
sc = plt.scatter(
    pcs[:,0], pcs[:,1],
    c=train_df["action"], cmap="tab10",
    s=10, alpha=0.6
)
plt.colorbar(sc, label="Action")
plt.xlabel("PCA 1"); plt.ylabel("PCA 2")
plt.title("PCA-Projektion der Zustände, eingefärbt nach Aktion")
plt.show()

# 2D PCA Projection of States colored by Reward
# Each point represents a state, colored by the action taken in that state
"""

# --------------------------------------------------------
# Test how many outliers get clipped by the 5std bounds
# For building 1: 3 outliers
"""
# 1) Reconstruct your “observation” DataFrame in raw units
raw_df = pd.concat([
    bld[[
        'month', 'hour', 'day_type', 'daylight_savings_status',
        'indoor_dry_bulb_temperature',
        'average_unmet_cooling_setpoint_difference',
        'indoor_relative_humidity',
        'non_shiftable_load', 'dhw_demand',
        'cooling_demand', 'heating_demand',
        'solar_generation', 'occupant_count',
        'temperature_set_point', 'hvac_mode'
    ]].reset_index(drop=True),
    wth[[
        'outdoor_dry_bulb_temperature',
        'outdoor_relative_humidity',
        'diffuse_solar_irradiance',
        'direct_solar_irradiance'
    ]].reset_index(drop=True),
    car[['kg_CO2/kWh']].reset_index(drop=True),
    prc[['Electricity Pricing [$/kWh]']].reset_index(drop=True)
], axis=1)

# 2) Compute mean, std, ±5σ bounds, and actual min/max
stats = raw_df.describe().T[['mean','std']].copy()
stats['lower_5sigma'] = stats['mean'] - 5 * stats['std']
stats['upper_5sigma'] = stats['mean'] + 5 * stats['std']
stats['min']          = raw_df.min()
stats['max']          = raw_df.max()
stats['below_lower']  = stats['min']  < stats['lower_5sigma']
stats['above_upper']  = stats['max']  > stats['upper_5sigma']

# 3) See which features (if any) would get clipped:
print(stats[['mean','std','lower_5sigma','upper_5sigma','min','max','below_lower','above_upper']])
"""

# --------------------------------------------------------

# multi-objective reinforcement learning (MORL) / multi-criteria optimization.
"""
# Comfort-Penalty
# I use Category II of the Standard 16798-1/2 of REHVA using the PMV-PPD method for buildings with an active cooling system. 
# Temperature in summer: 23-26 °C
# Temperature in winter: 20-24 °C
# deviation range: ~0-5 °C
comfort_penalty = 0.0
temp, rel_hum = obs['indoor_dry_bulb_temperature'], obs['indoor_relative_humidity']
if temp<20: comfort_penalty += (20-temp) 
if temp>24: comfort_penalty += (temp-24)
norm_comfort_penalty = comfort_penalty / 5.0 # normalize to [0,1]

# Optional: Emissions, Ramping
emis = max(0.0, net_load) * obs['kg_CO2/kWh']
if self.prev_net_load is None:
    ramp = 0.0
else:
    ramp = abs(net_load - self.prev_net_load)
self.prev_net_load = net_load

# weights
w_cost, w_penalty = 0.5, 0.5
#w_cost, w_pen, w_emis, w_ramp = 0.35, 0.35, 0.25, 0.05

- normalize to same scale
- rework reward function: transform to same scale and test different weights
- look into Stable Baselines 3 documentation for reward: 
if it even converges to 0: hat probleme damit bei reward funktionen zu 0 zu konvergieren, 
maximiert das immer gegen +/- unendlich
- workaround: calculate cost without batterie/pv and with it and maximize the difference
"""
# --------------------------------------------------------
"""
obs_dict, reward_dict, terminated, truncated, _ = env.step(action)
obs    = obs_dict["0"]
reward = reward_dict["0"]
done   = terminated["0"] or truncated["0"]

action = self.act(obs)
self.history.append((obs, action, reward))
total_reward += reward
"""

# --------------------------------------------------------

{'conditions': [
        {'field':'soc','op':'<=','threshold_name':'soc_strong_charge'}], 
        'action':'strong_charge'}, 
    {'conditions': [
        #{'field':'solar_generation','op':'>','threshold_name':'solar_high'}, 
        {'field':'soc','op':'<=','threshold_name':'soc_mild_discharge'}], 
        'action':'strong_charge'},
    # Mild charge if still low SOC
    {'conditions': [
        {'field':'soc','op':'<','threshold_name':'soc_mild_charge'}], 
        'action':'mild_charge'},
    # Strong discharge if high SOC and high price
    {'conditions': [
        {'field':'soc','op':'>=','threshold_name':'soc_strong_discharge'},
        {'field':'price','op':'>','threshold_name':'price_high'}
    ], 'action':'strong_discharge'},
    # Mild discharge if moderately high SOC and price not low
    {'conditions': [
        {'field':'soc','op':'>','threshold_name':'soc_mild_discharge'},
        {'field':'price','op':'>','threshold_name':'price_low'}
    ], 'action':'mild_discharge'}   
# --