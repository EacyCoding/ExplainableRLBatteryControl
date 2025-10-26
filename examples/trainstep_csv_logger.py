import os, csv, time
from typing import List, Optional, Sequence, Any, Dict
import numpy as np
import pandas as pd
from stable_baselines3.common.callbacks import BaseCallback

class TrainStepCSVLogger(BaseCallback):
    def __init__(
        self,
        save_dir: str = "logs/ppo",
        filename: str = "train_steps.csv",
        action_labels: Optional[Sequence[str]] = None,
        action_fracs: Optional[Sequence[float]] = None,
        flush_every_steps: int = 5000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.filename = filename
        self.action_labels = list(action_labels) if action_labels is not None else None
        self.action_fracs = np.asarray(action_fracs, dtype=float) if action_fracs is not None else None
        self.flush_every_steps = int(flush_every_steps)
        self._fh = None
        self._writer = None
        self._rows: List[list] = []
        self._ep_counters: Dict[int, int] = {}
        self.df: Optional[pd.DataFrame] = None
        self.ep_df: Optional[pd.DataFrame] = None

    def _on_training_start(self) -> None:
        os.makedirs(self.save_dir, exist_ok=True)
        path = os.path.join(self.save_dir, self.filename)
        self._fh = open(path, "w", newline="")
        self._writer = csv.writer(self._fh)
        self._writer.writerow(["step","env_id","episode","reward","done","action_id","action_frac","action_label"])
        self._fh.flush()
        if self.verbose:
            print(f"[TrainStepCSVLogger] Writing to: {os.path.abspath(path)}")

    def _on_step(self) -> bool:
        acts  = self.locals.get("actions")
        rews  = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [])
        n_envs = len(rews) if hasattr(rews, "__len__") else 1
        step = int(self.num_timesteps)
        for env_id in range(n_envs):
            act  = acts[env_id]  if hasattr(acts, "__len__")  else acts
            rew  = rews[env_id]  if hasattr(rews, "__len__")  else rews
            done = dones[env_id] if hasattr(dones, "__len__") else dones
            info = infos[env_id] if (isinstance(infos, (list, tuple)) and env_id < len(infos)) else {}
            done = bool(done)
            rew  = float(np.asarray(rew).sum())
            if env_id not in self._ep_counters:
                self._ep_counters[env_id] = 1
            episode = self._ep_counters[env_id]
            action_id = None
            action_frac = None
            action_label = None
            for k in ("action_id","action_discrete_id","discrete_action","action_idx","discrete_id"):
                if isinstance(info, dict) and k in info:
                    try:
                        action_id = int(info[k])
                    except Exception:
                        pass
                    break
            if action_id is None:
                try:
                    if np.isscalar(act):
                        action_id = int(act)
                    else:
                        arr = np.asarray(act)
                        if arr.size == 1:
                            action_id = int(arr.reshape(-1)[0])
                        else:
                            action_frac = float(arr.reshape(-1)[0])
                except Exception:
                    pass
            if action_id is not None and self.action_fracs is not None:
                if 0 <= action_id < len(self.action_fracs):
                    action_frac = float(self.action_fracs[action_id])
            elif action_frac is not None and self.action_fracs is not None:
                diffs = np.abs(self.action_fracs - action_frac)
                nearest_idx = int(np.argmin(diffs))
                action_id = nearest_idx
                action_frac = float(self.action_fracs[nearest_idx])
            if self.action_labels is not None and action_id is not None:
                if 0 <= action_id < len(self.action_labels):
                    action_label = str(self.action_labels[action_id])
            row = [step, env_id, episode, rew, int(done), action_id, action_frac, action_label]
            self._rows.append(row)
            if done:
                self._ep_counters[env_id] = episode + 1
        if len(self._rows) >= self.flush_every_steps:
            self._writer.writerows(self._rows)
            self._rows.clear()
            self._fh.flush()
        return True

    def _on_training_end(self) -> None:
        if self._fh is not None:
            if self._rows:
                self._writer.writerows(self._rows)
                self._rows.clear()
                self._fh.flush()
            self._fh.close()
            self._fh = None
            self._writer = None
        csv_path = os.path.join(self.save_dir, self.filename)
        try:
            self.df = pd.read_csv(csv_path)
        except Exception:
            if hasattr(self, "_rows_backup") and self._rows_backup:
                self.df = pd.DataFrame(self._rows_backup, columns=["step","env_id","episode","reward","done","action_id","action_frac","action_label"])
            else:
                self.df = pd.DataFrame(columns=["step","env_id","episode","reward","done","action_id","action_frac","action_label"])
        if isinstance(self.df, pd.DataFrame) and not self.df.empty:
            self.ep_df = (self.df.groupby(["env_id","episode"], as_index=False)
                            .agg(ep_return=("reward","sum"),
                                 ep_length=("reward","size")))
        else:
            self.ep_df = pd.DataFrame(columns=["env_id","episode","ep_return","ep_length"])
