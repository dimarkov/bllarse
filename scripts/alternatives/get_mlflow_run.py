"""Find or create an MLflow parent run and print its run ID.

Usage:
    RUN_ID=$(python scripts/alternatives/get_mlflow_run.py \
        --experiment-name bllarse --run-name IBProbit)
"""

import argparse
import mlflow


def main():
    parser = argparse.ArgumentParser(description="Get or create an MLflow parent run")
    parser.add_argument("--experiment-name", required=True, help="MLflow experiment name")
    parser.add_argument("--run-name", required=True, help="Name for the parent run")
    args = parser.parse_args()

    mlflow.set_experiment(args.experiment_name)

    runs = mlflow.search_runs(
        filter_string=f"run_name = '{args.run_name}'",
        output_format="list",
    )

    if runs:
        print(runs[0].info.run_id)
    else:
        run = mlflow.start_run(run_name=args.run_name)
        run_id = run.info.run_id
        mlflow.end_run()
        print(run_id)


if __name__ == "__main__":
    main()
