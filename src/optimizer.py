from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor
from typing import Mapping

import pandas as pd
import pulp

from .load_data import (
    DAYS,
    preference_lookup,
    rank_lookup,
)


@dataclass
class OptimizationResult:
    status: str
    objective_value: float | None
    assignment: pd.DataFrame


def _balanced_bounds(
    total: int,
    categories: int,
) -> tuple[int, int]:
    """
    Retorna os limites naturais para uma distribuição equilibrada.

    Exemplos:

    70 / 5 = 14
    -> (14, 14)

    72 / 5 = 14.4
    -> (14, 15)
    """

    lower = floor(
        total / categories
    )

    upper = ceil(
        total / categories
    )

    return lower, upper


def _default_day_bounds(
    responses: pd.DataFrame,
) -> dict[str, tuple[int, int]]:
    """
    Limites automáticos para o total de pessoas por dia.
    """

    lower, upper = _balanced_bounds(
        total=len(responses),
        categories=len(DAYS),
    )

    return {
        day: (lower, upper)
        for day in DAYS
    }


def _default_team_day_bounds(
    responses: pd.DataFrame,
) -> dict[str, dict[str, tuple[int, int]]]:
    """
    Gera limites equilibrados para cada equipe.

    No cenário atual:

    P&D:
        20 / 5 = 4 por dia.

    Plataforma:
        50 / 5 = 10 por dia.
    """

    result: dict[
        str,
        dict[
            str,
            tuple[int, int],
        ],
    ] = {}

    for team, group in responses.groupby(
        "team"
    ):
        lower, upper = _balanced_bounds(
            total=len(group),
            categories=len(DAYS),
        )

        result[team] = {
            day: (lower, upper)
            for day in DAYS
        }

    return result


def optimize_assignment(
    responses: pd.DataFrame,
    restrictions: pd.DataFrame | None = None,
    day_bounds: Mapping[
        str,
        tuple[int, int],
    ]
    | None = None,
    team_day_bounds: Mapping[
        str,
        Mapping[
            str,
            tuple[int, int],
        ],
    ]
    | None = None,
    solver_msg: bool = False,
) -> OptimizationResult:
    """
    Resolve o problema de alocação de home office.

    A função objetivo maximiza exclusivamente a pontuação
    derivada do ranking:

        1ª preferência -> 5
        2ª preferência -> 4
        3ª preferência -> 3
        4ª preferência -> 2
        5ª preferência -> 1

    A resposta de "importância" permanece disponível para análise,
    mas não altera a prioridade do colaborador no solver.
    """

    responses = responses.copy()

    participants = (
        responses["participant_id"]
        .tolist()
    )

    teams = (
        responses["team"]
        .unique()
        .tolist()
    )

    scores = preference_lookup(
        responses
    )

    ranks = rank_lookup(
        responses
    )

    if day_bounds is None:
        day_bounds = (
            _default_day_bounds(
                responses
            )
        )

    if team_day_bounds is None:
        team_day_bounds = (
            _default_team_day_bounds(
                responses
            )
        )

    if restrictions is None:
        restrictions = pd.DataFrame(
            columns=[
                "participant_id",
                "day",
            ]
        )

    blocked = {
        (
            row.participant_id,
            row.day,
        )
        for row in restrictions.itertuples(
            index=False
        )
    }

    problem = pulp.LpProblem(
        name="home_office_assignment",
        sense=pulp.LpMaximize,
    )

    x = pulp.LpVariable.dicts(
        "x",
        (
            participants,
            DAYS,
        ),
        lowBound=0,
        upBound=1,
        cat=pulp.LpBinary,
    )

    # ---------------------------------------------------------
    # Função objetivo
    # ---------------------------------------------------------

    problem += pulp.lpSum(
        scores[(participant, day)]
        * x[participant][day]
        for participant in participants
        for day in DAYS
    )

    # ---------------------------------------------------------
    # Restrição 1:
    # Cada pessoa recebe exatamente um dia.
    # ---------------------------------------------------------

    for participant in participants:
        problem += (
            pulp.lpSum(
                x[participant][day]
                for day in DAYS
            )
            == 1,
            f"one_day_{participant}",
        )

    # ---------------------------------------------------------
    # Restrição 2:
    # Equilíbrio total por dia.
    # ---------------------------------------------------------

    for day in DAYS:
        lower, upper = (
            day_bounds[day]
        )

        total_day = pulp.lpSum(
            x[participant][day]
            for participant in participants
        )

        problem += (
            total_day >= lower,
            f"day_min_{day}",
        )

        problem += (
            total_day <= upper,
            f"day_max_{day}",
        )

    # ---------------------------------------------------------
    # Restrição 3:
    # Equilíbrio por equipe.
    # ---------------------------------------------------------

    for team in teams:
        team_participants = (
            responses.loc[
                responses["team"] == team,
                "participant_id",
            ]
            .tolist()
        )

        for day in DAYS:
            lower, upper = (
                team_day_bounds[team][day]
            )

            total_team_day = (
                pulp.lpSum(
                    x[participant][day]
                    for participant
                    in team_participants
                )
            )

            problem += (
                total_team_day >= lower,
                f"team_min_{team}_{day}",
            )

            problem += (
                total_team_day <= upper,
                f"team_max_{team}_{day}",
            )

    # ---------------------------------------------------------
    # Restrição 4:
    # Dias individualmente inviáveis.
    # ---------------------------------------------------------

    for participant, day in blocked:
        if participant not in participants:
            continue

        if day not in DAYS:
            continue

        problem += (
            x[participant][day] == 0,
            f"blocked_{participant}_{day}",
        )

    # ---------------------------------------------------------
    # Solver
    # ---------------------------------------------------------

    solver = pulp.PULP_CBC_CMD(
        msg=solver_msg
    )

    problem.solve(solver)

    status = pulp.LpStatus[
        problem.status
    ]

    if status != "Optimal":
        return OptimizationResult(
            status=status,
            objective_value=None,
            assignment=pd.DataFrame(),
        )

    rows: list[dict] = []

    for participant in participants:
        team = responses.loc[
            responses["participant_id"]
            == participant,
            "team",
        ].iloc[0]

        for day in DAYS:
            value = pulp.value(
                x[participant][day]
            )

            if value is None:
                continue

            if value > 0.5:
                rank = ranks[
                    (
                        participant,
                        day,
                    )
                ]

                score = scores[
                    (
                        participant,
                        day,
                    )
                ]

                rows.append(
                    {
                        "participant_id": participant,
                        "team": team,
                        "assigned_day": day,
                        "rank_received": rank,
                        "score": score,
                        "regret": rank - 1,
                    }
                )

                break

    assignment = (
        pd.DataFrame(rows)
        .sort_values(
            "participant_id"
        )
        .reset_index(drop=True)
    )

    objective_value = float(
        pulp.value(
            problem.objective
        )
    )

    return OptimizationResult(
        status=status,
        objective_value=objective_value,
        assignment=assignment,
    )