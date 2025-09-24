
"""
a2c_decision_tree.py

Surrogate decision tree for SB3 A2C/PPO: learns a small classifier mapping (state -> action).
- Collects a dataset by rolling out the trained policy.
- Trains a DecisionTreeClassifier (optionally with GridSearchCV).
- Saves text/graph exports for inspection.

Works with:
- Discrete action spaces (recommended). For continuous policies, pass a discretizer.
- VecEnvs / Monitor / CityLearn (tries to use env.observation_names if available).

Usage (in your A2C notebook):
----------------------------------------------------------------
from a2c_decision_tree import collect_policy_dataset, train_policy_tree

df, feature_names = collect_policy_dataset(
    model,                 # trained A2C
    eval_env,              # vectorized env or single env (deterministic)
    n_steps=20000,         # or n_episodes=200
    deterministic=True,
    action_labels=ACTION_LABELS  # optional, for pretty outputs
)

clf, report = train_policy_tree(
    df,
    action_labels=ACTION_LABELS,
    feature_names=feature_names,
    out_dir="explain/a2c_tree",
    do_gridsearch=True,       # try False for a quick result
    max_depth=4,
    min_samples_leaf=50
)
print(report["summary"])
# Files in out_dir:
#  - tree_rules.txt
#  - tree_plot.png (matplotlib)
#  - optionally tree.dot / tree.svg (if graphviz available)
----------------------------------------------------------------
"""

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import os, io, math, warnings
import numpy as np
import pandas as pd

# sklearn imports
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# plotting (graphviz optional)
import matplotlib.pyplot as plt

def _flatten_obs(obs: Any) -> Tuple[np.ndarray, List[str]]:
    """Flatten observation (np.ndarray or dict) -> (1D array, feature_names).
    - Dict -> concatenate in sorted(key) order with key:idx naming.
    - ndarray -> flatten with x{i} naming.
    """
    if isinstance(obs, dict):
        keys = sorted(obs.keys())
        parts = []
        names = []
        for k in keys:
            v = np.asarray(obs[k]).ravel()
            parts.append(v)
            for i in range(v.size):
                names.append(f"{k}:{i}")
        flat = np.concatenate(parts) if parts else np.array([], dtype=float)
        return flat.astype(float), names

    arr = np.asarray(obs).ravel().astype(float)
    names = [f"x{i}" for i in range(arr.size)]
    return arr, names

def _maybe_env_obs_names(env, fallback: List[str]) -> List[str]:
    """Try to read env-specific observation names (e.g., CityLearn)."""
    try:
        # CityLearn 2023: env.unwrapped.observation_names[0] (single-building)
        names = getattr(getattr(env, 'unwrapped', env), 'observation_names', None)
        if isinstance(names, (list, tuple)) and len(names) > 0:
            # sometimes names is list of list (per-building); grab first if shape matches
            if isinstance(names[0], (list, tuple)) and len(names[0]) == len(fallback):
                return list(names[0])
            if len(names) == len(fallback):
                return list(names)
    except Exception:
        pass
    return fallback

def collect_policy_dataset(
    model,
    env,
    n_steps: Optional[int] = 20000,
    n_episodes: Optional[int] = None,
    deterministic: bool = True,
    action_labels: Optional[Sequence[str]] = None,
    discretize_continuous: Optional[callable] = None,  # fn(action_vec)->action_id
    progress: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """Roll out policy to collect (features -> action) pairs.

    Returns:
        df with columns: x0..xN, 'action' (int), 'action_label' (optional)
        feature_names list aligned to x0..xN (possibly replaced by env names)
    """
    # reset env and handle VecEnv/SB3 differences
    vec_env = getattr(env, 'num_envs', 1) > 1
    obs = env.reset()
    # SB3 VecEnv reset returns np.array shape (n_env, obs_dim) or dict of that shape
    # normalize to per-env iteration
    rows = []
    feature_names: List[str] = None

    steps = 0
    episodes = 0
    done_flags = np.zeros(getattr(env, 'num_envs', 1), dtype=bool)

    while True:
        # stop condition
        if n_steps is not None and steps >= n_steps:
            break
        if n_episodes is not None and episodes >= n_episodes:
            break

        # predict action(s)
        action, _ = model.predict(obs, deterministic=deterministic)

        # derive discrete id
        if discretize_continuous is not None:
            # convert continuous action vectors into a discrete id
            if getattr(env, 'num_envs', 1) > 1:
                action_ids = [int(discretize_continuous(a)) for a in action]
            else:
                action_ids = [int(discretize_continuous(action))]
        else:
            # assume discrete integer action space
            if np.isscalar(action):
                action_ids = [int(action)]
            else:
                try:
                    # vectorized discrete actions
                    arr = np.asarray(action).reshape(-1)
                    action_ids = [int(a) for a in arr]
                except Exception:
                    raise ValueError("Action is not discrete. Provide discretize_continuous=...")

        # step
        obs_next, rewards, dones, infos = env.step(action)

        # collect rows per env
        n_envs = getattr(env, 'num_envs', 1)
        if n_envs == 1:
            flat, names = _flatten_obs(obs)
            if feature_names is None: feature_names = names
            row = dict((f"x{i}", flat[i]) for i in range(len(flat)))
            row['action'] = action_ids[0]
            if action_labels is not None and 0 <= action_ids[0] < len(action_labels):
                row['action_label'] = str(action_labels[action_ids[0]])
            rows.append(row)
        else:
            # VecEnv: iterate
            if isinstance(obs, dict):
                # dict of arrays shape (n_env, feat)
                # split by env
                split_obs = []
                keys = sorted(obs.keys())
                for e in range(n_envs):
                    sub = {k: np.asarray(obs[k])[e] for k in keys}
                    split_obs.append(sub)
            else:
                # array shape (n_env, feat)
                split_obs = [np.asarray(obs)[e] for e in range(n_envs)]

            for e in range(n_envs):
                flat, names = _flatten_obs(split_obs[e])
                if feature_names is None: feature_names = names
                row = dict((f"x{i}", flat[i]) for i in range(len(flat)))
                row['action'] = action_ids[e]
                if action_labels is not None and 0 <= action_ids[e] < len(action_labels):
                    row['action_label'] = str(action_labels[action_ids[e]])
                rows.append(row)

        steps += n_envs
        # episode counting
        if n_envs == 1:
            if bool(dones):
                episodes += 1
        else:
            for e in range(n_envs):
                if isinstance(dones, (list, tuple, np.ndarray)) and bool(dones[e]):
                    episodes += 1

        obs = obs_next

        if progress and (steps % 5000 == 0):
            print(f"Collected ~{steps} steps...")

    df = pd.DataFrame(rows)
    # try to swap feature names for env-provided labels
    if feature_names is None:
        feature_names = [c for c in df.columns if c.startswith('x')]
    feature_names = _maybe_env_obs_names(env, feature_names)
    return df, feature_names

def train_policy_tree(
    df: pd.DataFrame,
    action_labels: Optional[Sequence[str]] = None,
    feature_names: Optional[List[str]] = None,
    out_dir: str = "explain/a2c_tree",
    do_gridsearch: bool = True,
    max_depth: int = 4,
    min_samples_leaf: int = 50,
    random_state: int = 0
):
    """Train a DecisionTreeClassifier on df[x*] -> df['action'] and export artifacts."""
    os.makedirs(out_dir, exist_ok=True)
    feat_cols = sorted([c for c in df.columns if c.startswith('x')], key=lambda c: int(c[1:]) if c[1:].isdigit() else 1e9)
    X = df[feat_cols].to_numpy()
    y = df['action'].astype(int).to_numpy()

    # Feature names fallback
    if feature_names is None:
        feature_names = feat_cols

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=random_state)

    # Quick tree or grid search
    if not do_gridsearch:
        clf = DecisionTreeClassifier(
            criterion='gini',
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            class_weight='balanced'
        )
        clf.fit(X_train, y_train)
        best_params = clf.get_params()
    else:
        param_grid = {
            'criterion': ['gini', 'entropy', 'log_loss'],
            'max_depth': [3, 4, 5, 6, None],
            'min_samples_leaf': [1, 5, 10, 25, 50],
            'min_samples_split': [2, 5, 10, 20],
            'class_weight': [None, 'balanced'],
            'ccp_alpha': [0.0, 0.0005, 0.001, 0.005],
            'splitter': ['best']
        }
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        grid = GridSearchCV(DecisionTreeClassifier(random_state=random_state), param_grid=param_grid,
                            scoring='balanced_accuracy', cv=cv, n_jobs=-1, verbose=1)
        grid.fit(X_train, y_train)
        clf = grid.best_estimator_
        best_params = grid.best_params_

    # Metrics
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    try:
        report = classification_report(y_test, y_pred, target_names=action_labels) if action_labels is not None else classification_report(y_test, y_pred)
    except Exception:
        report = classification_report(y_test, y_pred)

    # Save textual rules
    rules_txt = export_text(clf, feature_names=feature_names)
    with open(os.path.join(out_dir, 'tree_rules.txt'), 'w') as f:
        f.write(rules_txt)

    # Try Graphviz first
    svg_path = None
    try:
        from sklearn.tree import export_graphviz
        import graphviz
        dot = export_graphviz(
            clf,
            out_file=None,
            feature_names=feature_names,
            class_names=action_labels if action_labels is not None else [str(i) for i in sorted(set(y))],
            filled=True,
            rounded=True,
            special_characters=True,
            max_depth=None
        )
        graph = graphviz.Source(dot)
        svg_path = os.path.join(out_dir, 'tree.svg')
        graph.render(svg_path, format='svg', cleanup=True)
        svg_path = svg_path + '.svg' if not svg_path.endswith('.svg') else svg_path
    except Exception as e:
        # fallback: matplotlib PNG
        png_path = os.path.join(out_dir, 'tree_plot.png')
        plt.figure(figsize=(32, 16), dpi=200)
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=action_labels if action_labels is not None else [str(i) for i in sorted(set(y))],
            filled=True,
            rounded=True
        )
        plt.tight_layout()
        plt.savefig(png_path)
        plt.close()

    summary = f"Surrogate Tree Accuracy: {acc:.3f}\nBest params: {best_params}\nConfusion matrix (rows=true, cols=pred):\n{cm}\n\nClassification report:\n{report}"
    result = {
        'clf': clf,
        'feature_names': feature_names,
        'accuracy': float(acc),
        'best_params': best_params,
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'summary': summary,
        'rules_path': os.path.join(out_dir, 'tree_rules.txt'),
        'graph_path': svg_path if svg_path is not None else os.path.join(out_dir, 'tree_plot.png')
    }
    return clf, result


def explain_sample_path(clf: DecisionTreeClassifier, feature_names: List[str], x_row: np.ndarray) -> List[Tuple[str, float, float, str]]:
    """Return the decision path (feature, value, threshold, side) for a single sample x_row (1D)."""
    x = np.asarray(x_row).reshape(1, -1)
    tree = clf.tree_
    node = 0
    path = []
    while tree.feature[node] != -2:  # -2 => leaf
        f_idx = tree.feature[node]
        thr = tree.threshold[node]
        val = float(x[0, f_idx])
        go_left = val <= thr
        path.append((feature_names[f_idx], val, thr, "left" if go_left else "right"))
        node = tree.children_left[node] if go_left else tree.children_right[node]
    return path
