from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import pinocchio as pin

from fr3_rollout import make_rollout_context, load_scenario_yaml

SEED = 20260520
TRAIN_FRAC = 0.85
DT = 0.02

SRC = Path("/home/simon/davide_fr3_ws/src/cbf_safety_filter")
ROLLOUT_DIR = SRC / "data/rollouts/fullpool_candidates_seed20260520_n1500"
SCENARIO_DIR = SRC / "config/generated_scenarios_fullpool_candidates"
URDF_PATH = SRC / "include/urdf/fr3_robot.urdf"

OUT = SRC / "data/fullpool_splits/fullpool_seed20260520_n1500_splitfirst_g300_allfail"
OUT.mkdir(parents=True, exist_ok=True)

context = make_rollout_context(urdf_path=str(URDF_PATH))
model = context["model"]
ee_frame_id = context["ee_frame_id"]
data = model.createData()


def scenario_name_from_npz(p: Path) -> str:
    return p.name.replace("_offline_rollout.npz", "")


def compute_final_dist_to_goal(scenario_name: str, z):
    scenario = load_scenario_yaml(str(SCENARIO_DIR / f"{scenario_name}.yaml"))

    q_last = np.asarray(z["q_arm"][-1], dtype=float)
    dq_last = np.asarray(z["dq_arm"][-1], dtype=float)
    ddq_last = np.asarray(z["ddq_safe_arm"][-1], dtype=float)
    qp_final = bool(z["qp_solved"][-1])

    if qp_final:
        q_final_arm = q_last + dq_last * DT + 0.5 * ddq_last * DT**2
    else:
        q_final_arm = q_last.copy()

    q_full = np.zeros(model.nq)
    q_full[:7] = q_final_arm

    pin.forwardKinematics(model, data, q_full)
    pin.updateFramePlacements(model, data)
    ee_final = data.oMf[ee_frame_id].translation.copy()

    goal = np.asarray(scenario["goal_ee_pos"], dtype=float)
    goal_tol = float(scenario["goal_tolerance"])
    return float(np.linalg.norm(ee_final - goal)), goal_tol


rows = []

for npz_path in sorted(ROLLOUT_DIR.glob("scenario_*_offline_rollout.npz")):
    scenario = scenario_name_from_npz(npz_path)

    try:
        z = np.load(npz_path, allow_pickle=False)
        steps = int(len(z["q_arm"]))
        if steps == 0:
            raise RuntimeError("empty rollout")

        qp = np.asarray(z["qp_solved"]).astype(bool)
        min_h_series = np.asarray(z["min_h"], dtype=float)
        min_psi_series = np.asarray(z["min_psi"], dtype=float)
        min_dist_series = (
            np.asarray(z["min_dist"], dtype=float)
            if "min_dist" in z
            else np.full_like(min_h_series, np.nan)
        )

        qp_final = bool(qp[-1])
        qp_fail_count = int((~qp).sum())

        final_min_h = float(min_h_series[-1])
        final_min_psi = float(min_psi_series[-1])
        min_h_rollout = float(np.nanmin(min_h_series))
        min_psi_rollout = float(np.nanmin(min_psi_series))
        min_dist_rollout = (
            float(np.nanmin(min_dist_series))
            if np.isfinite(min_dist_series).any()
            else np.nan
        )

        final_dist_to_goal, goal_tol = compute_final_dist_to_goal(scenario, z)

        if not qp_final:
            status = "bad_qp"
            reason = "final_qp_solved_false"
        elif not np.isfinite(final_min_h):
            status = "bad_safety"
            reason = "final_min_h_not_finite"
        elif not np.isfinite(final_min_psi):
            status = "bad_safety"
            reason = "final_min_psi_not_finite"
        elif final_min_h < 0.0:
            status = "bad_safety"
            reason = "final_min_h_negative"
        elif final_min_psi < 0.0:
            status = "bad_safety"
            reason = "final_min_psi_negative"
        elif not np.isfinite(final_dist_to_goal):
            status = "bad_goal"
            reason = "final_dist_to_goal_not_finite"
        elif final_dist_to_goal > goal_tol:
            status = "bad_goal"
            reason = "final_dist_to_goal_above_tol"
        else:
            status = "good"
            reason = "ok"

        rows.append({
            "scenario": scenario,
            "npz_path": str(npz_path),
            "scenario_path": str(SCENARIO_DIR / f"{scenario}.yaml"),
            "status": status,
            "reason": reason,
            "steps": steps,
            "qp_final": qp_final,
            "qp_fail_count": qp_fail_count,
            "final_min_h": final_min_h,
            "final_min_psi": final_min_psi,
            "min_h_rollout": min_h_rollout,
            "min_psi_rollout": min_psi_rollout,
            "min_dist_rollout": min_dist_rollout,
            "final_dist_to_goal": final_dist_to_goal,
            "goal_tol": goal_tol,
        })

    except Exception as e:
        rows.append({
            "scenario": scenario,
            "npz_path": str(npz_path),
            "scenario_path": str(SCENARIO_DIR / f"{scenario}.yaml"),
            "status": "bad_runtime_or_corrupt",
            "reason": repr(e),
        })


df = pd.DataFrame(rows)
df.to_csv(OUT / "fr3_fullpool_catalog.csv", index=False)

print("============================================================")
print("FULLPOOL CATALOG COUNTS")
print("============================================================")
print(df["status"].value_counts().to_string())

# ============================================================
# 1) SPLIT FIRST, BEFORE ANY TRAIN SELECTION OR DUPLICATION
# ============================================================
train_parts = []
val_parts = []

for status, group in df.groupby("status"):
    group = group.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_train = int(round(TRAIN_FRAC * len(group)))

    train_parts.append(group.iloc[:n_train].copy())
    val_parts.append(group.iloc[n_train:].copy())

train_candidates = pd.concat(train_parts, ignore_index=True)
val_candidates = pd.concat(val_parts, ignore_index=True)

overlap = set(train_candidates.scenario) & set(val_candidates.scenario)
if overlap:
    raise SystemExit(f"Train/val leakage after split: {sorted(list(overlap))[:10]}")

train_candidates.to_csv(OUT / "fr3_fullpool_train_candidates_splitfirst.csv", index=False)
val_candidates.to_csv(OUT / "fr3_fullpool_val_candidates_splitfirst.csv", index=False)

# ============================================================
# 2) SELECT TRAINING SET ONLY FROM TRAIN SPLIT
# ============================================================
train_good_all = train_candidates[train_candidates.status == "good"].copy()
train_bad_safety_all = train_candidates[train_candidates.status == "bad_safety"].copy()
train_bad_qp_all = train_candidates[train_candidates.status == "bad_qp"].copy()
train_bad_goal_all = train_candidates[train_candidates.status == "bad_goal"].copy()

problems = []
if len(train_good_all) < 500:
    problems.append(f"need 500 train good rollouts for 300 good + 200 hard_good, have {len(train_good_all)}")
if len(train_bad_safety_all) == 0:
    problems.append("need at least 1 train bad_safety rollout, have 0")
if len(train_bad_qp_all) == 0:
    problems.append("need at least 1 train bad_qp rollout, have 0")
if len(train_bad_goal_all) == 0:
    problems.append("need at least 1 train bad_goal rollout, have 0")
if problems:
    raise SystemExit("Not enough candidates in train split:\n  " + "\n  ".join(problems))

good_hard_sorted = train_good_all.sort_values(
    ["min_h_rollout", "min_psi_rollout", "steps"],
    ascending=[True, True, False],
    kind="stable",
)

train_hard_good = good_hard_sorted.iloc[:200].copy()
remaining_good = good_hard_sorted.iloc[200:].copy()
train_good = remaining_good.sample(n=300, random_state=SEED).copy()

bad_safety_sorted = train_bad_safety_all.sort_values(
    ["min_h_rollout", "min_psi_rollout"],
    ascending=[True, True],
    kind="stable",
)
bad_qp_sorted = train_bad_qp_all.sort_values(
    ["qp_fail_count", "steps"],
    ascending=[False, False],
    kind="stable",
)
bad_goal_sorted = train_bad_goal_all.sort_values(
    ["final_dist_to_goal", "steps"],
    ascending=[False, False],
    kind="stable",
)

# New fullpool policy: use all available failure cases from the train split.
train_bad_safety = bad_safety_sorted.copy()
train_bad_qp = bad_qp_sorted.copy()
train_bad_goal = bad_goal_sorted.copy()

for frame, cat in [
    (train_good, "good"),
    (train_hard_good, "hard_good"),
    (train_bad_safety, "bad_safety"),
    (train_bad_qp, "bad_qp"),
    (train_bad_goal, "bad_goal"),
]:
    frame.loc[:, "train_category"] = cat

train_unique = pd.concat(
    [train_good, train_hard_good, train_bad_safety, train_bad_qp, train_bad_goal],
    ignore_index=True,
)

# ============================================================
# 3) VALIDATION SET ONLY FROM VAL SPLIT, NO DUPLICATION
# ============================================================
val_unique = val_candidates.copy()
val_unique["val_category"] = val_unique["status"]

val_good = val_unique[val_unique.status == "good"].copy()
if len(val_good) > 0:
    val_hard_good_names = set(
        val_good.sort_values(
            ["min_h_rollout", "min_psi_rollout", "steps"],
            ascending=[True, True, False],
            kind="stable",
        ).head(min(100, len(val_good))).scenario
    )
    val_unique.loc[val_unique.scenario.isin(val_hard_good_names), "val_category"] = "hard_good"

leak = set(train_unique.scenario) & set(val_unique.scenario)
if leak:
    raise SystemExit(f"Train/val leakage after selection: {sorted(list(leak))[:10]}")

# ============================================================
# 4) DUPLICATE ONLY AFTER SPLIT AND TRAIN SELECTION
# ============================================================
weights = {
    "good": 1,
    "hard_good": 2,
    "bad_safety": 4,
    "bad_qp": 4,
    "bad_goal": 4,
}

weighted_records = []
for _, row in train_unique.iterrows():
    cat = row["train_category"]
    for dup in range(weights[cat]):
        weighted_records.append({
            "path": row["npz_path"],
            "rollout_path": row["npz_path"],
            "scenario": row["scenario"],
            "category": cat,
            "dup_index": dup,
        })

random.Random(SEED).shuffle(weighted_records)

# ============================================================
# 5) WRITE OUTPUTS
# ============================================================
train_unique.to_csv(OUT / "fr3_fullpool_train_unique.csv", index=False)
val_unique.to_csv(OUT / "fr3_fullpool_val_unique.csv", index=False)

with open(OUT / "fr3_fullpool_train_weighted.jsonl", "w") as f:
    for rec in weighted_records:
        f.write(json.dumps(rec) + "\n")

with open(OUT / "fr3_fullpool_train_weighted.txt", "w") as f:
    for rec in weighted_records:
        f.write(rec["path"] + "\n")

with open(OUT / "fr3_fullpool_train_unique.txt", "w") as f:
    for _, row in train_unique.iterrows():
        f.write(row["npz_path"] + "\n")

with open(OUT / "fr3_fullpool_val_unique.jsonl", "w") as f:
    for _, row in val_unique.iterrows():
        f.write(json.dumps({
            "path": row["npz_path"],
            "rollout_path": row["npz_path"],
            "scenario": row["scenario"],
            "category": row["val_category"],
        }) + "\n")

with open(OUT / "fr3_fullpool_val_unique.txt", "w") as f:
    for _, row in val_unique.iterrows():
        f.write(row["npz_path"] + "\n")

for cat, g in train_unique.groupby("train_category"):
    g[["scenario", "npz_path"]].to_csv(OUT / f"train_{cat}.csv", index=False)

for cat, g in val_unique.groupby("val_category"):
    g[["scenario", "npz_path"]].to_csv(OUT / f"val_{cat}.csv", index=False)

summary = {
    "seed": SEED,
    "train_frac_before_duplication": TRAIN_FRAC,
    "split_policy": (
        "Stratified train/val split is performed before train selection and before duplication. "
        "Validation has no duplication and no test split; final test is held out separately."
    ),
    "train_policy": (
        "300 random good from train-good leftovers after hard_good selection, "
        "200 hard_good selected from train-good by smallest min_h/min_psi, "
        "all train bad_safety, all train bad_qp, all train bad_goal."
    ),
    "duplication_policy": weights,
    "catalog_counts": df["status"].value_counts().to_dict(),
    "train_candidate_counts_after_split": train_candidates["status"].value_counts().to_dict(),
    "val_candidate_counts_after_split": val_candidates["status"].value_counts().to_dict(),
    "train_unique_counts": train_unique["train_category"].value_counts().to_dict(),
    "train_weighted_counts": pd.Series([r["category"] for r in weighted_records]).value_counts().to_dict(),
    "train_unique_total": int(len(train_unique)),
    "train_weighted_total": int(len(weighted_records)),
    "val_unique_counts": val_unique["val_category"].value_counts().to_dict(),
    "val_unique_total": int(len(val_unique)),
    "train_val_overlap_count": int(len(leak)),
    "paths": {
        "catalog": str(OUT / "fr3_fullpool_catalog.csv"),
        "train_candidates_splitfirst": str(OUT / "fr3_fullpool_train_candidates_splitfirst.csv"),
        "val_candidates_splitfirst": str(OUT / "fr3_fullpool_val_candidates_splitfirst.csv"),
        "train_unique_csv": str(OUT / "fr3_fullpool_train_unique.csv"),
        "train_weighted_jsonl": str(OUT / "fr3_fullpool_train_weighted.jsonl"),
        "train_weighted_txt": str(OUT / "fr3_fullpool_train_weighted.txt"),
        "val_unique_csv": str(OUT / "fr3_fullpool_val_unique.csv"),
        "val_unique_jsonl": str(OUT / "fr3_fullpool_val_unique.jsonl"),
        "val_unique_txt": str(OUT / "fr3_fullpool_val_unique.txt"),
    },
}
(OUT / "fr3_fullpool_split_summary.json").write_text(json.dumps(summary, indent=2))

print()
print("============================================================")
print("SPLIT-FIRST FULLPOOL SUMMARY")
print("============================================================")
print("Train candidates after split:")
print(train_candidates["status"].value_counts().to_string())
print()
print("Val candidates after split:")
print(val_candidates["status"].value_counts().to_string())
print()
print("Train unique:")
print(train_unique["train_category"].value_counts().to_string())
print()
print("Train weighted:")
print(pd.Series([r["category"] for r in weighted_records]).value_counts().to_string())
print(f"weighted total: {len(weighted_records)}")
print()
print("Validation unique:")
print(val_unique["val_category"].value_counts().to_string())
print(f"val total: {len(val_unique)}")
print()
print("Train/val overlap:", len(leak))
print()
print("Saved split to:")
print(OUT)
