# Explainable Reinforcement Learning for Building Energy Management using CityLearn Dataset

This repository contains the code for the bachelor thesis ***Explainable Reinforcement Learning for Building Energy Management using the CityLearn Dataset***.

It includes:

* RL agents: **Deep Q-Network (DQN)**, **Advantage Actor–Critic (A2C)**, **Proximal Policy Optimization (PPO)**
* Baselines: **RBC** (rule-based controller) and **RBC0** (no-battery baseline)
* Model comparison plots
* CityLearn environment

## Setup

```bash
# install python modules
pip install -r requirements.txt
```

> The notebooks were developed on a JupyterHub instance on **HAICORE** (KIT HPC).
> **Adjust the file paths** at the top of each notebook to your local paths berfore running.
> More info on HAICORE: [https://www.nhr.kit.edu/userdocs/haicore/](https://www.nhr.kit.edu/userdocs/haicore/)

Using Jupyter notebooks for RL and RBC allows to rerun and adapt small code cells quickly without rerunning the entire notebook which saves time.

## Data

Expected folder structure:

```text
data/
  datasets/
    citylearn_challenge_2023_phase_3_1/
      Building_1.csv
      Building_1.pth
      Building_1.txt
      Building_2.csv
      Building_2.pth
      Building_2.txt
      Building_3.csv
      Building_3.pth
      Building_3.txt
      Building_4.csv
      Building_4.pth
      Building_4.txt
      Building_5.csv
      Building_5.pth
      Building_5.txt
      Building_6.csv
      Building_6.pth
      Building_6.txt
      carbon_intensity.csv
      pricing.csv
      schema.json
      weather.csv
    citylearn_challenge_2023_phase_3_2/
    citylearn_challenge_2023_phase_3_3/
examples/
  pricing_germany_2023_june_to_august.csv
```

## How to run

1. Open the notebooks in `examples/`, the final notebooks for every model are named `{model}_agent_final.ipynb`.
2. After training, a training file is saved which is loaded again for evaluation. After evaluation, the evaluation is saved in an evaluation file.
3. The evaluation file is loaded for the creation of the decision-tree.
4. The evaluation files of the different models can be used in `model_comparison.ipynb` to plot the comparison graphs of the different RL models and baselines.


## Notes

* **RBC vs. RBC0**: RBC is a hand-tuned controller with SoC/price rules; RBC0 is an idle baseline without battery use.
---


## CityLearn
CityLearn is an open source Farama Foundation Gymnasium environment for the implementation of Multi-Agent Reinforcement Learning (RL) for building energy coordination and demand response in cities. A major challenge for RL in demand response is the ability to compare algorithm performance. Thus, CityLearn facilitates and standardizes the evaluation of RL agents such that different algorithms can be easily compared with each other.

![Demand-response](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/dr.jpg)

## Environment Overview

CityLearn includes energy models of buildings and distributed energy resources (DER) including air-to-water heat pumps, electric heaters and batteries. A collection of building energy models makes up a virtual district (a.k.a neighborhood or community). In each building, space cooling, space heating and domestic hot water end-use loads may be independently satisfied through air-to-water heat pumps. Alternatively, space heating and domestic hot water loads can be satisfied through electric heaters.

![Citylearn](https://github.com/intelligent-environments-lab/CityLearn/blob/master/assets/images/environment.jpg)

## Installation
Install latest release in PyPi with `pip`:
```console
pip install CityLearn
```

## Documentation
Refer to the [docs](https://intelligent-environments-lab.github.io/CityLearn/).