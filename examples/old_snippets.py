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