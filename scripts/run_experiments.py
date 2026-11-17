from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.generate_data import generate_simulated_dataset
from src.load_data import (
    load_responses,
    load_restrictions,
)
from src.optimizer import optimize_assignment
from src.random_assignment import generate_random_assignment
from src.metrics import calculate_metrics


DAYS = [
    "segunda",
    "terca",
    "quarta",
    "quinta",
    "sexta",
]


def load_config() -> dict:

    path = PROJECT_ROOT / "config.yaml"

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def build_day_bounds(
    config: dict,
) -> dict[str, tuple[int, int]]:

    minimum = int(
        config["home_office"]["min_per_day"]
    )

    maximum = int(
        config["home_office"]["max_per_day"]
    )

    return {
        day: (
            minimum,
            maximum,
        )
        for day in DAYS
    }


def build_team_bounds(
    config: dict,
) -> dict[str, dict[str, tuple[int, int]]]:

    bounds = {}

    for team, values in (
        config["team_limits"].items()
    ):

        minimum = int(
            values["min_per_day"]
        )

        maximum = int(
            values["max_per_day"]
        )

        bounds[team] = {
            day: (
                minimum,
                maximum,
            )
            for day in DAYS
        }

    return bounds


def mean_random_metrics(
    responses: pd.DataFrame,
    restrictions: pd.DataFrame,
    day_bounds: dict,
    team_bounds: dict,
    number_of_runs: int,
    base_seed: int,
) -> dict:
    """
    Executa vários sorteios balanceados para a mesma
    população e retorna a média das métricas.

    Isso produz um baseline mais confiável do que usar
    um único sorteio.
    """

    metrics_list = []

    for index in range(number_of_runs):

        assignment = (
            generate_random_assignment(
                responses=responses,
                restrictions=restrictions,
                seed=base_seed + index,
                day_bounds=day_bounds,
                team_day_bounds=team_bounds,
            )
        )

        metrics = calculate_metrics(
            assignment
        )

        metrics_list.append(
            metrics
        )

    numeric_keys = [
        "average_score",
        "average_rank",
        "average_regret",
        "first_choice_pct",
        "top2_pct",
        "top3_pct",
        "low_preference_pct",
        "worst_choice_pct",
    ]

    result = {}

    for key in numeric_keys:

        result[key] = float(
            np.mean(
                [
                    metrics[key]
                    for metrics
                    in metrics_list
                ]
            )
        )

    return result


def main() -> None:

    config = load_config()

    number_of_experiments = int(
        config["experiments"][
            "number_of_experiments"
        ]
    )

    random_runs = int(
        config["experiments"][
            "random_baselines_per_experiment"
        ]
    )

    base_seed = int(
        config["experiments"]["base_seed"]
    )

    restriction_rate = float(
        config["simulation"][
            "restriction_rate"
        ]
    )

    day_weights = (
        config["preferences"][
            "day_weights"
        ]
    )

    day_bounds = build_day_bounds(
        config
    )

    team_bounds = build_team_bounds(
        config
    )

    output_root = (
        PROJECT_ROOT
        / config["paths"]["outputs"]
        / "monte_carlo"
    )

    simulation_cache = (
        output_root
        / "simulated_data"
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    simulation_cache.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    print("=" * 70)
    print("MONTE CARLO — HOME OFFICE OPTIMIZATION")
    print("=" * 70)

    print(
        f"Experimentos: {number_of_experiments}"
    )

    print(
        f"Sorteios por experimento: {random_runs}"
    )

    print()

    for experiment_index in range(
        number_of_experiments
    ):

        experiment_number = (
            experiment_index + 1
        )

        population_seed = (
            base_seed
            + experiment_index
        )

        experiment_data_dir = (
            simulation_cache
            / f"experiment_{experiment_number:04d}"
        )

        # ====================================================
        # Gerar população
        # ====================================================

        generate_simulated_dataset(
            output_dir=experiment_data_dir,
            seed=population_seed,
            restriction_rate=restriction_rate,
            day_weights=day_weights,
        )

        responses = load_responses(
            experiment_data_dir
            / "responses_70.csv"
        )

        restrictions = load_restrictions(
            experiment_data_dir
            / "restrictions.csv",
            responses=responses,
        )

        # ====================================================
        # Otimização
        # ====================================================

        optimization = optimize_assignment(
            responses=responses,
            restrictions=restrictions,
            day_bounds=day_bounds,
            team_day_bounds=team_bounds,
        )

        if optimization.status != "Optimal":

            print(
                f"[{experiment_number}/"
                f"{number_of_experiments}] "
                f"INFACTÍVEL — pulando."
            )

            rows.append(
                {
                    "experiment": experiment_number,
                    "population_seed": population_seed,
                    "status": optimization.status,
                }
            )

            continue

        optimized_metrics = (
            calculate_metrics(
                optimization.assignment
            )
        )

        # ====================================================
        # Sorteios
        # ====================================================

        random_seed = (
            base_seed
            + 100000
            + experiment_index * random_runs
        )

        random_metrics = (
            mean_random_metrics(
                responses=responses,
                restrictions=restrictions,
                day_bounds=day_bounds,
                team_bounds=team_bounds,
                number_of_runs=random_runs,
                base_seed=random_seed,
            )
        )

        # ====================================================
        # Ganhos
        # ====================================================

        score_gain = (
            optimized_metrics[
                "average_score"
            ]
            - random_metrics[
                "average_score"
            ]
        )

        first_choice_gain = (
            optimized_metrics[
                "first_choice_pct"
            ]
            - random_metrics[
                "first_choice_pct"
            ]
        )

        top2_gain = (
            optimized_metrics[
                "top2_pct"
            ]
            - random_metrics[
                "top2_pct"
            ]
        )

        regret_reduction = (
            random_metrics[
                "average_regret"
            ]
            - optimized_metrics[
                "average_regret"
            ]
        )

        rows.append(
            {
                "experiment": (
                    experiment_number
                ),
                "population_seed": (
                    population_seed
                ),
                "status": (
                    optimization.status
                ),
                "restrictions": (
                    len(restrictions)
                ),

                # Random
                "random_average_score": (
                    random_metrics[
                        "average_score"
                    ]
                ),
                "random_first_choice_pct": (
                    random_metrics[
                        "first_choice_pct"
                    ]
                ),
                "random_top2_pct": (
                    random_metrics[
                        "top2_pct"
                    ]
                ),
                "random_average_regret": (
                    random_metrics[
                        "average_regret"
                    ]
                ),

                # Optimized
                "optimized_average_score": (
                    optimized_metrics[
                        "average_score"
                    ]
                ),
                "optimized_first_choice_pct": (
                    optimized_metrics[
                        "first_choice_pct"
                    ]
                ),
                "optimized_top2_pct": (
                    optimized_metrics[
                        "top2_pct"
                    ]
                ),
                "optimized_average_regret": (
                    optimized_metrics[
                        "average_regret"
                    ]
                ),

                # Gain
                "score_gain": (
                    score_gain
                ),
                "first_choice_gain_pct": (
                    first_choice_gain
                ),
                "top2_gain_pct": (
                    top2_gain
                ),
                "regret_reduction": (
                    regret_reduction
                ),
            }
        )

        if (
            experiment_number == 1
            or experiment_number % 25 == 0
            or experiment_number
            == number_of_experiments
        ):

            print(
                f"[{experiment_number:04d}/"
                f"{number_of_experiments:04d}] "
                f"Opt={optimized_metrics['average_score']:.3f} | "
                f"Random={random_metrics['average_score']:.3f} | "
                f"Ganho={score_gain:.3f}"
            )

    # ========================================================
    # Resultados
    # ========================================================

    results = pd.DataFrame(rows)

    results_path = (
        output_root
        / "experiments.csv"
    )

    results.to_csv(
        results_path,
        index=False,
    )

    valid = results.loc[
        results["status"] == "Optimal"
    ].copy()

    if valid.empty:
        raise RuntimeError(
            "Nenhum experimento factível foi encontrado."
        )

    # ========================================================
    # Estatísticas finais
    # ========================================================

    optimized_wins = (
        valid["score_gain"] > 0
    ).sum()

    ties = (
        np.isclose(
            valid["score_gain"],
            0,
        )
    ).sum()

    losses = (
        valid["score_gain"] < 0
    ).sum()

    summary = {
        "experiments_requested": (
            number_of_experiments
        ),
        "experiments_valid": int(
            len(valid)
        ),
        "random_baselines_per_experiment": (
            random_runs
        ),

        "optimized_wins": int(
            optimized_wins
        ),
        "ties": int(ties),
        "losses": int(losses),

        "optimized_win_rate_pct": float(
            100
            * optimized_wins
            / len(valid)
        ),

        "average_random_score": float(
            valid[
                "random_average_score"
            ].mean()
        ),

        "average_optimized_score": float(
            valid[
                "optimized_average_score"
            ].mean()
        ),

        "average_score_gain": float(
            valid[
                "score_gain"
            ].mean()
        ),

        "minimum_score_gain": float(
            valid[
                "score_gain"
            ].min()
        ),

        "maximum_score_gain": float(
            valid[
                "score_gain"
            ].max()
        ),

        "average_first_choice_gain_pct": float(
            valid[
                "first_choice_gain_pct"
            ].mean()
        ),

        "average_top2_gain_pct": float(
            valid[
                "top2_gain_pct"
            ].mean()
        ),

        "average_regret_reduction": float(
            valid[
                "regret_reduction"
            ].mean()
        ),
    }

    summary_path = (
        output_root
        / "summary.json"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    # ========================================================
    # Resumo CSV
    # ========================================================

    pd.DataFrame(
        [summary]
    ).to_csv(
        output_root
        / "summary.csv",
        index=False,
    )

    # ========================================================
    # Terminal
    # ========================================================

    print()
    print("=" * 70)
    print("RESULTADO MONTE CARLO")
    print("=" * 70)

    print()

    print(
        f"Experimentos válidos: "
        f"{len(valid)}"
    )

    print(
        f"Otimização superior: "
        f"{summary['optimized_win_rate_pct']:.2f}%"
    )

    print()

    print(
        "Satisfação média:"
    )

    print(
        f"  Sorteio    : "
        f"{summary['average_random_score']:.3f}"
    )

    print(
        f"  Otimização : "
        f"{summary['average_optimized_score']:.3f}"
    )

    print()

    print(
        f"Ganho médio: "
        f"{summary['average_score_gain']:.3f}"
    )

    print(
        f"Ganho de 1ª escolha: "
        f"{summary['average_first_choice_gain_pct']:.2f} p.p."
    )

    print(
        f"Ganho de Top 2: "
        f"{summary['average_top2_gain_pct']:.2f} p.p."
    )

    print(
        f"Redução média de regret: "
        f"{summary['average_regret_reduction']:.3f}"
    )

    print()

    print(
        f"Resultados salvos em: "
        f"{output_root}"
    )


if __name__ == "__main__":
    main()