"""
Generate CausalMan causal inference benchmark data from the command line.

For each (variant, seed) combination the script writes:
  - observational.csv            training data with no interventions
  - task1_force_ltl_control.csv  Task 1 control arm   do(PF_M1_T1_Force_LTL = 15000)
  - task1_force_ltl_treatment.csv Task 1 treatment arm do(PF_M1_T1_Force_LTL = 18000)
  - task2_force_control.csv      Task 2 control arm   do(PF_M1_T1_Force = 16000)
  - task2_force_treatment.csv    Task 2 treatment arm do(PF_M1_T1_Force = 30000)

Usage examples:
    python generate_causal_inference_data.py --variant small
    python generate_causal_inference_data.py --variant medium --seeds 4 6 42 66 90
    python generate_causal_inference_data.py --variant micro --seeds 42 --samples 5000
    python generate_causal_inference_data.py --variant large --output my_output/
"""

import argparse
import os
import sys

from causalman import CausalMan  # type: ignore[import-not-found]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Outcome variable per scale — the binary process-result column used to compute
# treatment effects. Update the relevant entry if the column name differs for a scale.
OUTCOME_BY_SCALE = {
    "micro":  "Sec_C2_Machine1_ProcessResult",
    "small":  "Sec_C2_Machine1_ProcessResult",
    "medium": "Sec_C2_Machine1_ProcessResult",
    "large":  "Sec_C2_Machine1_ProcessResult",
}

# Task 1 raises the lower tolerance limit for the T1 press-fitting force.
# Task 2 raises the T1 press-fitting force itself to an abnormal level.
TASKS = [
    {
        "slug":      "task1_force_ltl",
        "variable":  "PF_M1_T1_Force_LTL",
        "control":   15000.0,
        "treatment": 18000.0,
    },
    {
        "slug":      "task2_force",
        "variable":  "PF_M1_T1_Force",
        "control":   16000.0,
        "treatment": 30000.0,
    },
]


# ---------------------------------------------------------------------------
# Core generation logic
# ---------------------------------------------------------------------------

def generate(scale: str, seeds: list[int], n_samples: int, output_root: str) -> None:
    outcome = OUTCOME_BY_SCALE[scale]
    os.makedirs(output_root, exist_ok=True)

    for seed in seeds:
        seed_dir = os.path.join(output_root, scale, f"seed_{seed:03d}")
        os.makedirs(seed_dir, exist_ok=True)
        print(f"\n── {scale}  seed={seed} ──")

        # -- Observational data ------------------------------------------------
        # Run the simulator with no interventions to produce training data.
        sim = CausalMan(
            name=f"causalman_{scale}", seed=seed, batch_multiplier=1,
            parallelize=True, save_path=os.path.join(seed_dir, "observational"),
        )
        df_obs, _, _, dag = sim.sample()

        # Step 1: nodes the DAG marks as observable (Boolean True or string "Observable").
        dag_observable = [
            n for n, d in dag.nodes(data=True)
            if d.get("Observable") in (True, "Observable") and n in df_obs.columns
        ]
        # Step 2: drop columns constant in the observational data (no information).
        observable = [col for col in dag_observable if df_obs[col].nunique(dropna=False) > 1]
        print(f"  observable columns : {len(observable)}  "
              f"({len(dag_observable) - len(observable)} constant columns removed)")

        if len(df_obs) < n_samples:
            print(
                f"  WARNING: simulator produced {len(df_obs)} rows "
                f"but {n_samples} were requested. "
                "Increase --batch-multiplier or reduce --samples.",
                file=sys.stderr,
            )
            n_samples_actual = len(df_obs)
        else:
            n_samples_actual = n_samples

        df_obs[observable].sample(n=n_samples_actual, random_state=seed).to_csv(
            os.path.join(seed_dir, "observational.csv"), index=False
        )
        print(f"  observational : {n_samples_actual:,} rows  |  {len(observable)} observable columns")

        # -- Interventional arms -----------------------------------------------
        # For each task, sample control and treatment distributions via do-calculus.
        for task in TASKS:
            arms = {}
            for arm in ("control", "treatment"):
                sim = CausalMan(
                    name=f"causalman_{scale}", seed=seed, batch_multiplier=1,
                    parallelize=True, save_path=os.path.join(seed_dir, str(task["slug"]), arm),
                )
                sim.intervention_dict = {task["variable"]: task[arm]}
                df, _, _, _ = sim.sample()
                df[observable].sample(n=n_samples_actual, random_state=seed).to_csv(
                    os.path.join(seed_dir, f"{task['slug']}_{arm}.csv"),
                    index=False,
                )
                arms[arm] = df[observable]

            # ATE = E[Y | do(treatment)] - E[Y | do(control)]
            ate = arms["treatment"][outcome].mean() - arms["control"][outcome].mean()
            print(f"  {task['slug']:<25} ATE = {ate:.4f}")

    print(f"\nDone → {os.path.abspath(output_root)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CausalMan causal inference benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--variant", required=True,
        choices=["micro", "small", "medium", "large"],
        help="CausalMan scale variant to generate data for.",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=[42], metavar="SEED",
        help="One or more random seeds (default: 42). "
             "Full benchmark: --seeds 4 6 42 66 90",
    )
    parser.add_argument(
        "--samples", type=int, default=10_000, metavar="N",
        help="Number of rows to write per dataset (default: 10000).",
    )
    parser.add_argument(
        "--output", default="output/causalman_causal_inference", metavar="DIR",
        help="Root output directory (default: output/causalman_causal_inference).",
    )

    args = parser.parse_args()
    generate(args.variant, args.seeds, args.samples, args.output)


if __name__ == "__main__":
    main()
