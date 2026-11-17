from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from src.generate_data import (
    generate_simulated_dataset,
)

from src.load_data import (
    load_responses,
    load_restrictions,
)

from src.optimizer import (
    optimize_assignment,
)

from src.random_assignment import (
    generate_random_assignment,
)

from src.metrics import (
    calculate_metrics,
    compare_methods,
    participant_results,
)

from src.reporting import (
    save_operational_reports,
)

from src.plots import (
    plot_first_choice_distribution,
    plot_rank_distribution,
    plot_method_comparison,
    plot_weekly_schedule_heatmap,
    plot_occupancy_percentage,
    plot_team_home_office_percentage,
    plot_preference_vs_assignment,
)


DAYS = [
    "segunda",
    "terca",
    "quarta",
    "quinta",
    "sexta",
]


def load_config() -> dict:

    config_path = (
        PROJECT_ROOT
        / "config.yaml"
    )

    if not config_path.exists():
        raise FileNotFoundError(
            "Arquivo config.yaml "
            f"não encontrado: {config_path}"
        )

    with open(
        config_path,
        "r",
        encoding="utf-8",
    ) as file:

        return yaml.safe_load(file)


def build_day_bounds(
    config: dict,
) -> dict[str, tuple[int, int]]:

    minimum = int(
        config[
            "home_office"
        ][
            "min_per_day"
        ]
    )

    maximum = int(
        config[
            "home_office"
        ][
            "max_per_day"
        ]
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
) -> dict[
    str,
    dict[
        str,
        tuple[int, int],
    ],
]:

    result = {}

    for team, values in (
        config[
            "team_limits"
        ].items()
    ):

        minimum = int(
            values[
                "min_per_day"
            ]
        )

        maximum = int(
            values[
                "max_per_day"
            ]
        )

        result[team] = {
            day: (
                minimum,
                maximum,
            )
            for day in DAYS
        }

    return result


def validate_population(
    responses: pd.DataFrame,
    config: dict,
) -> None:

    expected_total = int(
        config[
            "simulation"
        ][
            "total_employees"
        ]
    )

    if len(responses) != expected_total:
        raise ValueError(
            "Quantidade de participantes incorreta. "
            f"Esperado: {expected_total}. "
            f"Encontrado: {len(responses)}."
        )

    expected_teams = (
        config[
            "simulation"
        ][
            "teams"
        ]
    )

    actual_counts = (
        responses[
            "team"
        ]
        .value_counts()
        .to_dict()
    )

    for team, expected in (
        expected_teams.items()
    ):

        actual = actual_counts.get(
            team,
            0,
        )

        if actual != expected:
            raise ValueError(
                f"Quantidade incorreta para {team}. "
                f"Esperado: {expected}. "
                f"Encontrado: {actual}."
            )


def save_json(
    data: dict,
    path: Path,
) -> None:

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=2,
        )


def print_metrics(
    title: str,
    metrics: dict,
) -> None:

    print()
    print(title)
    print("-" * 60)

    print(
        "Satisfação média : "
        f"{metrics['average_score']:.3f} / 5"
    )

    print(
        "Ranking médio    : "
        f"{metrics['average_rank']:.3f}"
    )

    print(
        "Regret médio     : "
        f"{metrics['average_regret']:.3f}"
    )

    print(
        "1ª escolha       : "
        f"{metrics['first_choice_pct']:.2f}%"
    )

    print(
        "Top 2            : "
        f"{metrics['top2_pct']:.2f}%"
    )

    print(
        "Top 3            : "
        f"{metrics['top3_pct']:.2f}%"
    )

    print(
        "4ª ou 5ª opção   : "
        f"{metrics['low_preference_pct']:.2f}%"
    )

    print(
        "5ª opção         : "
        f"{metrics['worst_choice_pct']:.2f}%"
    )


def print_operational_summary(
    occupancy: pd.DataFrame,
    team_occupancy: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print("MAPA DE OCUPAÇÃO")
    print("=" * 70)

    print()

    print(
        f"{'Dia':<12}"
        f"{'HO':>6}"
        f"{'% HO':>10}"
        f"{'Escritório':>12}"
        f"{'% Esc.':>10}"
    )

    print("-" * 50)

    for row in occupancy.itertuples():

        print(
            f"{row.day_label:<12}"
            f"{row.home_office:>6}"
            f"{row.home_office_pct:>9.1f}%"
            f"{row.office:>12}"
            f"{row.office_pct:>9.1f}%"
        )

    print()
    print("HOME OFFICE POR EQUIPE")
    print("-" * 70)

    for team in sorted(
        team_occupancy["team"].unique()
    ):

        print()
        print(team)

        subset = team_occupancy[
            team_occupancy[
                "team"
            ] == team
        ]

        for row in subset.itertuples():

            print(
                f"  {row.day_label:<10}"
                f"HO: {row.home_office:>2}"
                f" ({row.home_office_pct:>5.1f}%)"
                f" | Escritório: "
                f"{row.office:>2}"
                f" ({row.office_pct:>5.1f}%)"
            )


def main() -> None:

    config = load_config()

    seed = int(
        config[
            "simulation"
        ][
            "seed"
        ]
    )

    restriction_rate = float(
        config[
            "simulation"
        ][
            "restriction_rate"
        ]
    )

    day_weights = (
        config[
            "preferences"
        ][
            "day_weights"
        ]
    )

    simulated_dir = (
        PROJECT_ROOT
        / config[
            "paths"
        ][
            "simulated_data"
        ]
    )

    outputs_root = (
        PROJECT_ROOT
        / config[
            "paths"
        ][
            "outputs"
        ]
    )

    experiment_dir = (
        outputs_root
        / f"experiment_seed_{seed}"
    )

    assignments_dir = (
        experiment_dir
        / "assignments"
    )

    operational_dir = (
        experiment_dir
        / "operational"
    )

    metrics_dir = (
        experiment_dir
        / "metrics"
    )

    figures_dir = (
        experiment_dir
        / "figures"
    )

    for directory in [
        experiment_dir,
        assignments_dir,
        operational_dir,
        metrics_dir,
        figures_dir,
    ]:

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    # ======================================================
    # 1. Dados
    # ======================================================

    print("=" * 70)
    print(
        "HOME OFFICE OPTIMIZATION — BEMAGRO"
    )
    print("=" * 70)

    print()
    print(
        "[1/7] Gerando população simulada..."
    )

    generate_simulated_dataset(
        output_dir=simulated_dir,
        seed=seed,
        restriction_rate=restriction_rate,
        day_weights=day_weights,
    )

    responses = load_responses(
        simulated_dir
        / "responses_70.csv"
    )

    restrictions = load_restrictions(
        simulated_dir
        / "restrictions.csv",
        responses=responses,
    )

    validate_population(
        responses,
        config,
    )

    print(
        f"Participantes: {len(responses)}"
    )

    print(
        "P&D: "
        f"{(responses['team'] == 'PD').sum()}"
    )

    print(
        "Plataforma: "
        f"{(responses['team'] == 'PLATAFORMA').sum()}"
    )

    print(
        "Restrições individuais: "
        f"{len(restrictions)}"
    )

    day_bounds = (
        build_day_bounds(
            config
        )
    )

    team_bounds = (
        build_team_bounds(
            config
        )
    )

    # ======================================================
    # 2. Sorteio
    # ======================================================

    print()
    print(
        "[2/7] Executando sorteio balanceado..."
    )

    random_assignment = (
        generate_random_assignment(
            responses=responses,
            restrictions=restrictions,
            seed=seed + 1,
            day_bounds=day_bounds,
            team_day_bounds=team_bounds,
        )
    )

    # ======================================================
    # 3. Otimização
    # ======================================================

    print()
    print(
        "[3/7] Executando otimização..."
    )

    optimization = (
        optimize_assignment(
            responses=responses,
            restrictions=restrictions,
            day_bounds=day_bounds,
            team_day_bounds=team_bounds,
        )
    )

    if optimization.status != "Optimal":

        raise RuntimeError(
            "Modelo não encontrou solução ótima. "
            f"Status: {optimization.status}"
        )

    optimized_assignment = (
        optimization.assignment
    )

    # ======================================================
    # 4. Métricas
    # ======================================================

    print()
    print(
        "[4/7] Calculando métricas..."
    )

    random_metrics = (
        calculate_metrics(
            random_assignment
        )
    )

    optimized_metrics = (
        calculate_metrics(
            optimized_assignment
        )
    )

    print_metrics(
        "SORTEIO BALANCEADO",
        random_metrics,
    )

    print_metrics(
        "OTIMIZAÇÃO",
        optimized_metrics,
    )

    summary = compare_methods(
        {
            "random": random_assignment,
            "optimized": optimized_assignment,
        }
    )

    # ======================================================
    # 5. Saídas operacionais
    # ======================================================

    print()
    print(
        "[5/7] Construindo mapa operacional..."
    )

    reports = (
        save_operational_reports(
            responses=responses,
            assignment=optimized_assignment,
            restrictions=restrictions,
            day_bounds=day_bounds,
            team_day_bounds=team_bounds,
            assignments_dir=assignments_dir,
            operational_dir=operational_dir,
        )
    )

    # Alocações básicas
    random_assignment.to_csv(
        assignments_dir
        / "random_assignment.csv",
        index=False,
    )

    optimized_assignment.to_csv(
        assignments_dir
        / "optimized_assignment.csv",
        index=False,
    )

    participant_results(
        responses,
        optimized_assignment,
    ).to_csv(
        assignments_dir
        / "employee_results.csv",
        index=False,
    )

    # ======================================================
    # 6. Resumo
    # ======================================================

    print()
    print(
        "[6/7] Salvando métricas..."
    )

    summary.to_csv(
        metrics_dir
        / "summary.csv",
        index=False,
    )

    gain = (
        optimized_metrics[
            "average_score"
        ]
        - random_metrics[
            "average_score"
        ]
    )

    relative_gain = (
        100.0
        * gain
        / random_metrics[
            "average_score"
        ]
    )

    validations = reports[
        "constraint_validation"
    ]

    validation_ok = bool(
        validations[
            "passed"
        ].all()
    )

    report = {
        "seed": seed,
        "participants": len(responses),
        "teams": (
            responses[
                "team"
            ]
            .value_counts()
            .to_dict()
        ),
        "restrictions": len(
            restrictions
        ),
        "solver_status": (
            optimization.status
        ),
        "objective_value": (
            optimization.objective_value
        ),
        "all_constraints_valid": (
            validation_ok
        ),
        "random": (
            random_metrics
        ),
        "optimized": (
            optimized_metrics
        ),
        "gain": {
            "average_score_absolute": (
                float(gain)
            ),
            "average_score_relative_pct": (
                float(relative_gain)
            ),
        },
    }

    save_json(
        report,
        metrics_dir
        / "summary.json",
    )

    # ======================================================
    # 7. Gráficos
    # ======================================================

    print()
    print(
        "[7/7] Gerando gráficos..."
    )

    plot_first_choice_distribution(
        responses,
        figures_dir,
    )

    plot_rank_distribution(
        random_assignment,
        figures_dir,
        filename=(
            "random_rank_distribution.png"
        ),
        title=(
            "Sorteio — posição recebida"
        ),
    )

    plot_rank_distribution(
        optimized_assignment,
        figures_dir,
        filename=(
            "optimized_rank_distribution.png"
        ),
        title=(
            "Otimização — posição recebida"
        ),
    )

    plot_method_comparison(
        summary,
        figures_dir,
    )

    plot_weekly_schedule_heatmap(
        responses,
        optimized_assignment,
        figures_dir,
    )

    plot_occupancy_percentage(
        reports[
            "occupancy_by_day"
        ],
        figures_dir,
    )

    plot_team_home_office_percentage(
        reports[
            "occupancy_by_team"
        ],
        figures_dir,
    )

    plot_preference_vs_assignment(
        reports[
            "preference_vs_assignment"
        ],
        figures_dir,
    )

    # ======================================================
    # Terminal
    # ======================================================

    print_operational_summary(
        reports[
            "occupancy_by_day"
        ],
        reports[
            "occupancy_by_team"
        ],
    )

    print()
    print("=" * 70)
    print("VALIDAÇÃO DAS RESTRIÇÕES")
    print("=" * 70)

    print(
        "Status do solver: "
        f"{optimization.status}"
    )

    print(
        "Todas as restrições respeitadas: "
        f"{'SIM' if validation_ok else 'NÃO'}"
    )

    if not validation_ok:

        print()
        print(
            "Restrições com problema:"
        )

        print(
            validations.loc[
                ~validations[
                    "passed"
                ]
            ].to_string(
                index=False
            )
        )

    print()
    print("=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)

    print()

    print(
        "Satisfação média — sorteio: "
        f"{random_metrics['average_score']:.3f}"
    )

    print(
        "Satisfação média — otimização: "
        f"{optimized_metrics['average_score']:.3f}"
    )

    print(
        "Ganho absoluto: "
        f"{gain:.3f}"
    )

    print(
        "Ganho relativo: "
        f"{relative_gain:.2f}%"
    )

    print()

    print(
        "Mapa semanal:"
    )

    print(
        figures_dir
        / "weekly_schedule_heatmap.png"
    )

    print()

    print(
        "Resultados:"
    )

    print(
        experiment_dir
    )


if __name__ == "__main__":
    main()