from __future__ import annotations

from math import ceil, floor
from typing import Mapping

import numpy as np
import pandas as pd
import pulp

from .load_data import (
    DAYS,
    preference_lookup,
    rank_lookup,
)


def _balanced_bounds(
    total: int,
) -> tuple[int, int]:
    lower = floor(
        total / len(DAYS)
    )

    upper = ceil(
        total / len(DAYS)
    )

    return lower, upper


def _default_day_bounds(
    responses: pd.DataFrame,
) -> dict[str, tuple[int, int]]:
    lower, upper = _balanced_bounds(
        len(responses)
    )

    return {
        day: (lower, upper)
        for day in DAYS
    }


def _default_team_day_bounds(
    responses: pd.DataFrame,
) -> dict[
    str,
    dict[
        str,
        tuple[int, int],
    ],
]:
    result = {}

    for team, group in responses.groupby(
        "team"
    ):
        lower, upper = _balanced_bounds(
            len(group)
        )

        result[team] = {
            day: (lower, upper)
            for day in DAYS
        }

    return result


def generate_random_assignment(
    responses: pd.DataFrame,
    restrictions: pd.DataFrame | None = None,
    seed: int = 42,
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
) -> pd.DataFrame:
    """
    Gera uma alocação aleatória factível.

    O sorteio respeita:

    - um dia por participante;
    - limites totais por dia;
    - limites por equipe;
    - restrições individuais.

    Para obter uma solução diferente a cada seed, é criada uma
    função objetivo aleatória sem utilizar as preferências reais.
    """

    rng = np.random.default_rng(
        seed
    )

    participants = (
        responses["participant_id"]
        .tolist()
    )

    teams = (
        responses["team"]
        .unique()
        .tolist()
    )

    if restrictions is None:
        restrictions = pd.DataFrame(
            columns=[
                "participant_id",
                "day",
            ]
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

    blocked = {
        (
            row.participant_id,
            row.day,
        )
        for row in restrictions.itertuples(
            index=False
        )
    }

    random_values = {
        (
            participant,
            day,
        ): float(rng.random())
        for participant in participants
        for day in DAYS
    }

    problem = pulp.LpProblem(
        name="random_home_office_assignment",
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

    # Objetivo puramente aleatório.
    problem += pulp.lpSum(
        random_values[
            (
                participant,
                day,
            )
        ]
        * x[participant][day]
        for participant in participants
        for day in DAYS
    )

    # Uma atribuição por participante.
    for participant in participants:
        problem += (
            pulp.lpSum(
                x[participant][day]
                for day in DAYS
            )
            == 1
        )

    # Balanceamento total.
    for day in DAYS:
        lower, upper = (
            day_bounds[day]
        )

        total = pulp.lpSum(
            x[participant][day]
            for participant in participants
        )

        problem += total >= lower
        problem += total <= upper

    # Balanceamento por equipe.
    for team in teams:
        team_participants = (
            responses.loc[
                responses["team"]
                == team,
                "participant_id",
            ]
            .tolist()
        )

        for day in DAYS:
            lower, upper = (
                team_day_bounds[team][day]
            )

            total = pulp.lpSum(
                x[participant][day]
                for participant
                in team_participants
            )

            problem += total >= lower
            problem += total <= upper

    # Restrições individuais.
    for participant, day in blocked:
        if (
            participant in participants
            and day in DAYS
        ):
            problem += (
                x[participant][day]
                == 0
            )

    solver = pulp.PULP_CBC_CMD(
        msg=solver_msg
    )

    problem.solve(solver)

    status = pulp.LpStatus[
        problem.status
    ]

    if status != "Optimal":
        raise RuntimeError(
            "Não foi possível gerar um sorteio "
            f"factível. Status: {status}"
        )

    ranks = rank_lookup(
        responses
    )

    scores = preference_lookup(
        responses
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

            if (
                value is not None
                and value > 0.5
            ):
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

    return (
        pd.DataFrame(rows)
        .sort_values(
            "participant_id"
        )
        .reset_index(drop=True)
    )