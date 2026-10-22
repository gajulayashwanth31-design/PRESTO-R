"""Train the Phase 4 lambda screening grid and select from validation results."""

import argparse
from pathlib import Path

from presto.agents.train import train_agent
from presto.config import load_config
from presto.eval.select_lambda import select_lambda


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/agent/presto_dqn.yaml")
    parser.add_argument("--lambdas", nargs="+", type=float, default=[0.0, 0.05, 0.1, 0.25, 0.5])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--steps", type=int, default=150000)
    parser.add_argument("--out", default="results/tuning/lambda.csv")
    args = parser.parse_args()
    scenarios = ["congestion", "node_failure", "fluctuation", "mixed"]
    run_patterns = []
    for value in args.lambdas:
        name = f"presto_lam{value:g}"
        run_patterns.append(str(Path("runs") / name / "seed_*"))
        for seed in args.seeds:
            cfg = load_config(
                args.config,
                [
                    f"reward.lam={value}",
                    f"agent.name={name}",
                    f"agent.total_steps={args.steps}",
                    f"agent.eval_scenarios={scenarios}",
                ],
            )
            print(f"Training {name}, seed={seed}, steps={args.steps}", flush=True)
            train_agent(cfg, seed)
    summary, selected = select_lambda(run_patterns, args.out)
    print(summary.to_string(index=False))
    print(
        f"Selected lambda={selected:g} by the predeclared validation PDR rule; "
        "no test seeds were evaluated."
    )


if __name__ == "__main__":
    main()
