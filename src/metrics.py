from __future__ import annotations

from typing import Mapping

import pandas as pd

from .load_data import DAYS


def calculate_metrics(
    assignment: pd.DataFrame,
) -> dict:
    """
    Calcula as principais métricas de qualidade
    de uma alocação.
    """

    if assignment.empty:
        raise ValueError(
            "O DataFrame de alocação está vazio."
        )

    total = len(assignment)

    average_score = (
        assignment["score"].mean()
    )

    average_rank = (
        assignment["rank_received"]
        .mean()
    )

    average_regret = (
        assignment["regret"]
        .mean()
    )

    first_choice = (
        assignment["rank_received"]
        == 1
    ).sum()

    top2 = (
        assignment["rank_received"]
        <= 2
    ).sum()

    top3 = (
        assignment["rank_received"]
        <= 3
    ).sum()

    low_preference = (
        assignment["rank_received"]
        >= 4
    ).sum()

    worst_choice = (
        assignment["rank_received"]
        == 5
    ).sum()

    return {
        "participants": total,
        "average_score": float(
            average_score
        ),
        "average_rank": float(
            average_rank
        ),
        "average_regret": float(
            average_regret
        ),
        "first_choice_count": int(
            first_choice
        ),
        "first_choice_pct": float(
            100 * first_choice / total
        ),
        "top2_count": int(top2),
        "top2_pct": float(
            100 * top2 / total
        ),
        "top3_count": int(top3),
        "top3_pct": float(
            100 * top3 / total
        ),
        "low_preference_count": int(
            low_preference
        ),
        "low_preference_pct": float(
            100
            * low_preference
            / total
        ),
        "worst_choice_count": int(
            worst_choice
        ),
        "worst_choice_pct": float(
            100
            * worst_choice
            / total
        ),
    }


def day_distribution(
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantidade de participantes em home office
    em cada dia.
    """

    counts = (
        assignment["assigned_day"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    return pd.DataFrame(
        {
            "day": DAYS,
            "home_office": [
                int(counts[day])
                for day in DAYS
            ],
        }
    )


def team_day_distribution(
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Distribuição de P&D e Plataforma por dia.
    """

    table = pd.crosstab(
        assignment["assigned_day"],
        assignment["team"],
    )

    table = table.reindex(
        DAYS,
        fill_value=0,
    )

    table.index.name = "day"

    return (
        table
        .reset_index()
    )


def rank_distribution(
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantidade de participantes que receberam
    cada posição do ranking.
    """

    counts = (
        assignment["rank_received"]
        .value_counts()
        .reindex(
            [1, 2, 3, 4, 5],
            fill_value=0,
        )
    )

    return pd.DataFrame(
        {
            "rank": [
                1,
                2,
                3,
                4,
                5,
            ],
            "count": [
                int(counts[rank])
                for rank
                in [1, 2, 3, 4, 5]
            ],
        }
    )


def compare_methods(
    assignments: Mapping[
        str,
        pd.DataFrame,
    ],
) -> pd.DataFrame:
    """
    Compara diferentes métodos.

    Exemplo:

        {
            "optimized": df_opt,
            "random": df_random,
        }
    """

    rows: list[dict] = []

    for method, assignment in (
        assignments.items()
    ):
        metrics = calculate_metrics(
            assignment
        )

        rows.append(
            {
                "method": method,
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def participant_results(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Junta ranking original e resultado da alocação
    em uma única tabela.
    """

    columns = [
        "participant_id",
        "team",
        "pref_1",
        "pref_2",
        "pref_3",
        "pref_4",
        "pref_5",
        "importance",
    ]

    result = responses[
        columns
    ].merge(
        assignment[
            [
                "participant_id",
                "assigned_day",
                "rank_received",
                "score",
                "regret",
            ]
        ],
        on="participant_id",
        how="left",
    )

    return result