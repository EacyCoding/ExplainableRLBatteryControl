
"""
Surrogate Decision Tree for RBC Agent Logs (state -> action)

- Robust loader that supports CSV and JSONL logs with varying schemas
- Auto-detects target (action) column, or discretizes continuous commands
- Prunes leaky/meta columns automatically
- Performs grid search (balanced_accuracy) with stratified CV
- Prints global + local explanations
- Exports a crisp SVG (Graphviz) or falls back to Matplotlib PNG

USAGE (inside a notebook):
    RBC_LOG_FILE = "path/to/your/rbc_log.csv"   # or .jsonl
    # Optionally customize LABELS or mapping:
    # ACTION_LABELS = ["discharge_full", "discharge_half", "idle", "charge_half", "charge_full"]
    # or INT_TO_FRAC = [-1.0, -0.5, 0.0, 0.5, 1.0]

    %run rbc_surrogate_tree.py

USAGE (CLI):
    python rbc_surrogate_tree.py --log-file path/to/log.csv

Notes:
- If your log contains a continuous command (e.g., power setpoint),
  we discretize into K bins (default K=5) centered by quantiles and sign symmetry.
- To force your own mapping, set ACTION_LABELS or INT_TO_FRAC as globals before running,
  or pass --bins K to control discretization.
"""

import os, sys, json, argparse, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ------------------ Config (overridable via globals or CLI) ------------------
DEFAULT_TEST_SIZE = 0.25
DEFAULT_RANDOM_STATE = 0
DEFAULT_N_SPLITS = 5
DEFAULT_BINS = 5  # for continuous->discrete fallback

# Candidate columns for the action/decision in diverse RBC logs
ACTION_ID_CANDIDATES = [
    "action_id", "rbc_action_id", "act_id", "decision_id",
    "action", "act", "decision", "cmd_id", "power_cmd_id",
]

# Candidate columns for a continuous control signal (to discretize if no ID found)
ACTION_CONT_CANDIDATES = [
    "command", "cmd", "power_cmd", "setpoint", "set_point",
    "power", "p_set", "p_cmd", "battery_power", "storage_power",
]

# Columns we never want to use as features (meta/leakage)
EXPLICIT_DROP = {
    "global_step", "env_id", "episode", "step", "timestep",
    "reward", "done", "terminal", "truncate",
    "action_id", "action", "action_label", "action_frac",
    "rbc_action_id", "act_id", "decision_id", "decision",
    "cmd_id", "power_cmd_id"
}

# Prefixes that likely leak target/labels or future info
LEAK_PREFIXES = (
    "action", "act", "cmd", "decision",
    "reward", "done", "terminal", "adv", "return", "value",
    "target", "future", "next_", "label", "y_", "gt_", "truth"
)

# ------------------ Helpers ------------------
def _read_any_log(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {os.path.abspath(path)}")
    ext = os.path.splitext(path)[1].lower()
    if ext in [".csv", ".tsv"]:
        # attempt to auto-detect delimiter
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep=None, engine="python")
        return df
    if ext in [".jsonl", ".ndjson"]:
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # try forgiving trailing commas
                    line = line.rstrip(",")
                    rows.append(json.loads(line))
        return pd.DataFrame(rows)
    if ext in [".json"]:
        data = json.load(open(path, "r", encoding="utf-8"))
        if isinstance(data, list):
            return pd.DataFrame(data)
        if isinstance(data, dict) and "data" in data and isinstance(data["data"], list):
            return pd.DataFrame(data["data"])
        return pd.json_normalize(data)
    # fallback: try CSV first
    try:
        return pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Unsupported log extension '{ext}' and CSV fallback failed: {e}")

def _pick_action_column(df: pd.DataFrame):
    # 1) try known id columns
    for c in ACTION_ID_CANDIDATES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c, "id"
    # 2) try to coerce string labels to ids if present
    for c in ACTION_ID_CANDIDATES:
        if c in df.columns:
            # factorize strings/categories into int ids
            ids, uniques = pd.factorize(df[c])
            df["_derived_action_id"] = ids.astype(int)
            return "_derived_action_id", "factorized"
    # 3) try continuous control candidates
    for c in ACTION_CONT_CANDIDATES:
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c]):
            return c, "continuous"
    # 4) last resort: scan for 'action' substring numeric column
    for c in df.columns:
        if "action" in c.lower() and pd.api.types.is_numeric_dtype(df[c]):
            return c, "id"
    raise KeyError("No suitable action/decision column found. "
                   "Tried ID candidates and continuous control columns.")

def _discretize_if_needed(df: pd.DataFrame, col: str, mode: str, bins: int = DEFAULT_BINS):
    """
    Returns (target_col_name, class_names). If mode=='id' or 'factorized', ensure int.
    If 'continuous', discretize into bins; attempt sign-aware symmetry for odd bins.
    """
    if mode in ("id", "factorized"):
        y = df[col].astype(int)
        k = int(y.max()) + 1 if y.min() == 0 else len(np.unique(y))
        # Optional global mappings if present
        class_names = None
        g = globals()
        if "ACTION_LABELS" in g:
            labs = list(g["ACTION_LABELS"])
            if len(labs) >= k:
                class_names = labs[:k]
        elif "INT_TO_FRAC" in g:
            frac = list(g["INT_TO_FRAC"])
            mapping = { -1.0:"discharge_full", -0.5:"discharge_half", 0.0:"idle", 0.5:"charge_half", 1.0:"charge_full" }
            try:
                class_names = [mapping.get(float(frac[i]), str(frac[i])) for i in range(k)]
            except Exception:
                class_names = [str(i) for i in range(k)]
        else:
            class_names = [str(i) for i in sorted(np.unique(y))]
        df["_target_action_id"] = y.to_numpy()
        return "_target_action_id", class_names

    # continuous -> discretize
    x = df[col].astype(float).to_numpy()
    if len(np.unique(x[~np.isnan(x)])) < bins:
        # low cardinality, factorize unique sorted values
        uniq = np.sort(np.unique(x))
        mapping = {v: i for i, v in enumerate(uniq)}
        y = np.array([mapping.get(v, 0) for v in x])
        class_names = [f"{v:.4g}" for v in uniq]
        df["_target_action_id"] = y
        return "_target_action_id", class_names

    # sign-aware bins if odd number (e.g., 5) to get discharge/idle/charge
    if bins % 2 == 1:
        # center bin around 0 by quantiles on |x|
        q = np.quantile(np.abs(x), np.linspace(0, 1, (bins + 1) // 2 + 1))
        q = np.unique(q)
        # build symmetric edges
        neg_edges = -q[::-1]
        pos_edges = q
        edges = np.unique(np.concatenate([neg_edges, pos_edges]))
        edges[0] = np.min([edges[0], x.min()]) - 1e-9
        edges[-1] = np.max([edges[-1], x.max()]) + 1e-9
        y = np.digitize(x, edges) - 1  # bin indices
        # compress to 0..K-1 contiguous
        uniq = np.unique(y)
        remap = {v: i for i, v in enumerate(uniq)}
        y = np.array([remap[v] for v in y])
        # names: negative bins -> discharge, middle -> idle-ish, positive -> charge
        k = len(uniq)
        mid = k // 2
        class_names = []
        for i in range(k):
            if i < mid:
                class_names.append(f"discharge_{mid - i}")
            elif i == mid:
                class_names.append("idle")
            else:
                class_names.append(f"charge_{i - mid}")
        df["_target_action_id"] = y
        return "_target_action_id", class_names

    # even bins: plain quantile binning
    q = np.quantile(x, np.linspace(0, 1, bins + 1))
    q = np.unique(q)
    if len(q) - 1 < bins:
        # fallback to uniform bins
        q = np.linspace(x.min(), x.max(), bins + 1)
    y = np.digitize(x, q[1:-1])  # 0..bins-1
    class_names = [f"[{q[i]:.3g},{q[i+1]:.3g})" for i in range(len(q) - 1)]
    df["_target_action_id"] = y
    return "_target_action_id", class_names

def _select_features(df: pd.DataFrame, target_col: str):
    num_bool_cols = df.select_dtypes(include=[np.number, bool]).columns
    feat_cols = []
    for c in num_bool_cols:
        if c == target_col: 
            continue
        if c in EXPLICIT_DROP:
            continue
        low = c.lower()
        if any(low.startswith(p) for p in LEAK_PREFIXES):
            continue
        feat_cols.append(c)
    feat_cols = sorted(set(feat_cols))
    if not feat_cols:
        # try objects that can be coerced
        for c in df.columns:
            if c == target_col or c in EXPLICIT_DROP:
                continue
            low = c.lower()
            if any(low.startswith(p) for p in LEAK_PREFIXES):
                continue
            if pd.api.types.is_object_dtype(df[c]):
                try:
                    df[c] = pd.to_numeric(df[c])
                    feat_cols.append(c)
                except Exception:
                    pass
    assert len(feat_cols) > 0, (
        f"No features selected – check your columns!\n"
        f"(numeric candidates: {list(num_bool_cols)[:10]})"
    )
    return feat_cols

def _class_names_from_globals(y):
    g = globals()
    k = int(np.max(y)) + 1
    if "ACTION_LABELS" in g:
        labs = list(g["ACTION_LABELS"])
        if len(labs) >= k:
            return labs[:k]
    if "INT_TO_FRAC" in g:
        frac = list(g["INT_TO_FRAC"])
        mapping = { -1.0:"discharge_full", -0.5:"discharge_half", 0.0:"idle", 0.5:"charge_half", 1.0:"charge_full" }
        try:
            return [mapping.get(float(frac[i]), str(frac[i])) for i in range(k)]
        except Exception:
            return [str(i) for i in range(k)]
    return [str(i) for i in range(k)]

# ------------------ Core Pipeline ------------------
def train_surrogate_tree(df: pd.DataFrame, bins: int = DEFAULT_BINS, random_state: int = DEFAULT_RANDOM_STATE):
    action_col, mode = _pick_action_column(df)
    target_col, class_names = _discretize_if_needed(df, action_col, mode, bins=bins)
    print(f"[target] column='{target_col}' (source='{action_col}', mode='{mode}')")
    if class_names is None:
        y_tmp = df[target_col].astype(int).to_numpy()
        class_names = _class_names_from_globals(y_tmp)

    feat_cols = _select_features(df, target_col)
    print("n_features:", len(feat_cols), "sample:", feat_cols[:10])

    # Prepare arrays
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy()
    y = df[target_col].astype(int).to_numpy()

    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DEFAULT_TEST_SIZE, random_state=random_state, stratify=y
    )

    # GridSearch
    param_grid = {
        "criterion": ["gini", "entropy", "log_loss"],
        "max_depth": [3, 4, 5, 6, None],
        "min_samples_leaf": [1, 5, 10, 25, 50],
        "min_samples_split": [2, 5, 10, 20],
        "class_weight": [None, "balanced"],
        "ccp_alpha": [0.0, 0.0005, 0.001, 0.005],
        "splitter": ["best"]
    }
    cv = StratifiedKFold(n_splits=DEFAULT_N_SPLITS, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        DecisionTreeClassifier(random_state=random_state),
        param_grid=param_grid,
        scoring="balanced_accuracy",
        cv=cv,
        n_jobs=-1,
        verbose=1
    )
    grid.fit(X_train, y_train)
    print("Best balanced_accuracy:", grid.best_score_)
    print("Best params:", grid.best_params_)

    clf = grid.best_estimator_

    # Quality
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Surrogate Tree Accuracy: {acc:.3f}")
    print("Confusion matrix (rows=true, cols=pred):\n", confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred, target_names=[str(x) for x in class_names]))

    # Feature importances
    print("\nTop feature importances:")
    imp = clf.feature_importances_
    assert len(imp) == len(feat_cols), "Mismatch between Importances and Feature-Namen."
    imp_idx = np.argsort(imp)[::-1]
    for i in imp_idx[:10]:
        if imp[i] <= 0: break
        print(f"- {feat_cols[i]}: {imp[i]:.4f}")

    # Local explanation
    def explain_sample(idx_in_df: int):
        x = df.iloc[idx_in_df][feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy().reshape(1, -1)
        true_a = int(df.iloc[idx_in_df][target_col])
        pred_a = int(clf.predict(x)[0])
        proba  = clf.predict_proba(x)[0]
        tree = clf.tree_
        node = 0
        path = []
        while tree.feature[node] != -2:  # -2 => leaf
            f_idx = tree.feature[node]
            thr = tree.threshold[node]
            val = float(x[0, f_idx])
            go_left = val <= thr
            path.append((feat_cols[f_idx], val, thr, "left" if go_left else "right"))
            node = tree.children_left[node] if go_left else tree.children_right[node]
        def _label(a):
            try:
                return class_names[a]
            except Exception:
                return str(a)
        print(f"True action: {true_a} ({_label(true_a)}), Pred: {pred_a} ({_label(pred_a)})")
        print("Proba:", {str(class_names[i]): float(f"{p:.3f}") for i,p in enumerate(proba)})
        print("Decision path:")
        for name, val, thr, side in path:
            print(f" - {name}: {val:.4f} <= {thr:.4f} -> {side}")
        return pred_a

    # Attach artifacts
    artifacts = {
        "clf": clf,
        "feat_cols": feat_cols,
        "class_names": class_names,
        "target_col": target_col,
        "df": df,
        "explain_sample": explain_sample,
    }
    return artifacts

def export_tree_svg_png(clf, feature_names, class_names):
    """Export SVG (Graphviz) or fallback PNG with Matplotlib. Returns file paths."""
    out_svg, out_png = None, None
    try:
        from sklearn.tree import export_graphviz
        import graphviz
        dot = export_graphviz(
            clf,
            out_file=None,
            feature_names=feature_names,
            class_names=[str(c) for c in class_names],
            filled=True,
            rounded=True,
            special_characters=True,
        )
        src = graphviz.Source(dot)
        svg_bytes = src.pipe(format='svg')
        out_svg = "rbc_surrogate_tree.svg"
        with open(out_svg, "wb") as f:
            f.write(svg_bytes)
        print(f"Exported SVG -> {out_svg}")
    except Exception as e:
        print("Graphviz unavailable; fallback to matplotlib. Error:", e)
        plt.figure(figsize=(32, 16), dpi=250)
        plot_tree(
            clf,
            feature_names=feature_names,
            class_names=[str(c) for c in class_names],
            filled=True,
            rounded=True,
            max_depth=4,
            fontsize=12
        )
        plt.tight_layout()
        out_png = "rbc_surrogate_tree.png"
        plt.savefig(out_png, bbox_inches="tight", dpi=200)
        plt.close()
        print(f"Exported PNG -> {out_png}")
    return out_svg, out_png

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-file", type=str, default=os.environ.get("RBC_LOG_FILE", ""))
    parser.add_argument("--bins", type=int, default=int(os.environ.get("RBC_BINS", DEFAULT_BINS)))
    args = parser.parse_args() if any(a.startswith("--") for a in sys.argv[1:]) else argparse.Namespace(
        log_file=os.environ.get("RBC_LOG_FILE", ""),
        bins=int(os.environ.get("RBC_BINS", DEFAULT_BINS))
    )

    if not args.log_file:
        # Try to import from a notebook variable if present
        if "RBC_LOG_FILE" in globals():
            args.log_file = globals()["RBC_LOG_FILE"]
        else:
            print("ERROR: Provide --log-file or set RBC_LOG_FILE env/global.")
            return

    df = _read_any_log(args.log_file)
    print(f"Loaded {len(df)} rows from {os.path.abspath(args.log_file)}")
    artifacts = train_surrogate_tree(df, bins=args.bins)

    out_svg, out_png = export_tree_svg_png(
        artifacts["clf"], artifacts["feat_cols"], artifacts["class_names"]
    )

    # Quick local explanation on a random sample
    idx = np.random.randint(0, len(artifacts["df"]))
    print(f"\nExample local explanation for df index {idx}:")
    _ = artifacts["explain_sample"](idx)

    # Export text rules (depth-limited for readability)
    rules = export_text(artifacts["clf"], feature_names=list(artifacts["feat_cols"]), max_depth=5)
    with open("rbc_surrogate_rules.txt", "w", encoding="utf-8") as f:
        f.write(rules)
    print("Exported text rules -> rbc_surrogate_rules.txt")

    print("\nArtifacts:")
    if out_svg: print(f"- SVG: {out_svg}")
    if out_png: print(f"- PNG: {out_png}")
    print("- Rules: rbc_surrogate_rules.txt")

if __name__ == "__main__":
    main()
