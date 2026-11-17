from __future__ import annotations

from pathlib import Path
from typing import Mapping

import pandas as pd

from .load_data import DAYS


DAY_LABELS = {
    "segunda": "Segunda",
    "terca": "Terça",
    "quarta": "Quarta",
    "quinta": "Quinta",
    "sexta": "Sexta",
}


def build_weekly_schedule(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria o mapa semanal dos colaboradores.

    Exemplo:

    participant_id | team | segunda | terca | ...
    EMP0001        | PD   | OFFICE  | HO    | ...
    """

    assigned_day = dict(
        zip(
            assignment["participant_id"],
            assignment["assigned_day"],
        )
    )

    rows: list[dict] = []

    ordered_responses = responses.sort_values(
        ["team", "participant_id"]
    )

    for person in ordered_responses.itertuples(index=False):

        participant_id = person.participant_id
        team = person.team

        if participant_id not in assigned_day:
            raise ValueError(
                f"Participante sem alocação: {participant_id}"
            )

        day_assigned = assigned_day[participant_id]

        row = {
            "participant_id": participant_id,
            "team": team,
        }

        for day in DAYS:
            row[day] = (
                "HO"
                if day == day_assigned
                else "ESCRITORIO"
            )

        rows.append(row)

    return pd.DataFrame(rows)


def build_home_office_by_day(
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera uma lista operacional:

    dia -> pessoa -> equipe -> ranking recebido.
    """

    result = assignment[
        [
            "participant_id",
            "team",
            "assigned_day",
            "rank_received",
            "score",
            "regret",
        ]
    ].copy()

    result["day_order"] = result[
        "assigned_day"
    ].map(
        {
            day: index
            for index, day in enumerate(DAYS)
        }
    )

    result = (
        result.sort_values(
            [
                "day_order",
                "team",
                "participant_id",
            ]
        )
        .drop(columns=["day_order"])
        .rename(
            columns={
                "assigned_day": "day",
            }
        )
        .reset_index(drop=True)
    )

    return result


def build_occupancy_by_day(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula quantidade e percentual de pessoas:

    - em home office;
    - presencialmente no escritório;

    para cada dia da semana.
    """

    total_people = len(responses)

    home_office_counts = (
        assignment["assigned_day"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    rows: list[dict] = []

    for day in DAYS:

        home_office = int(
            home_office_counts[day]
        )

        office = (
            total_people
            - home_office
        )

        home_office_pct = (
            100.0
            * home_office
            / total_people
        )

        office_pct = (
            100.0
            * office
            / total_people
        )

        rows.append(
            {
                "day": day,
                "day_label": DAY_LABELS[day],
                "total_people": total_people,
                "home_office": home_office,
                "home_office_pct": home_office_pct,
                "office": office,
                "office_pct": office_pct,
            }
        )

    return pd.DataFrame(rows)


def build_occupancy_by_team(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calcula ocupação diária separadamente
    para P&D e Plataforma.
    """

    rows: list[dict] = []

    teams = sorted(
        responses["team"].unique()
    )

    for team in teams:

        team_people = responses.loc[
            responses["team"] == team,
            "participant_id",
        ]

        team_ids = set(team_people)

        team_total = len(team_ids)

        team_assignment = assignment[
            assignment["participant_id"].isin(
                team_ids
            )
        ]

        counts = (
            team_assignment["assigned_day"]
            .value_counts()
            .reindex(
                DAYS,
                fill_value=0,
            )
        )

        for day in DAYS:

            home_office = int(
                counts[day]
            )

            office = (
                team_total
                - home_office
            )

            rows.append(
                {
                    "team": team,
                    "day": day,
                    "day_label": DAY_LABELS[day],
                    "team_total": team_total,
                    "home_office": home_office,
                    "home_office_pct": (
                        100.0
                        * home_office
                        / team_total
                    ),
                    "office": office,
                    "office_pct": (
                        100.0
                        * office
                        / team_total
                    ),
                }
            )

    return pd.DataFrame(rows)


def build_preference_vs_assignment(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara:

    - quantas pessoas escolheram determinado dia
      como primeira preferência;

    - quantas efetivamente foram alocadas nesse dia.
    """

    demand = (
        responses["pref_1"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    assigned = (
        assignment["assigned_day"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    rows: list[dict] = []

    for day in DAYS:

        first_choice = int(
            demand[day]
        )

        final_assignment = int(
            assigned[day]
        )

        rows.append(
            {
                "day": day,
                "day_label": DAY_LABELS[day],
                "first_choice_demand": first_choice,
                "assigned_home_office": final_assignment,
                "difference": (
                    final_assignment
                    - first_choice
                ),
            }
        )

    return pd.DataFrame(rows)


def validate_assignment(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
    restrictions: pd.DataFrame,
    day_bounds: Mapping[
        str,
        tuple[int, int],
    ],
    team_day_bounds: Mapping[
        str,
        Mapping[
            str,
            tuple[int, int],
        ],
    ],
) -> pd.DataFrame:
    """
    Validação explícita da solução.

    Não mede se a solução é 'boa'.

    Mede se ela respeita as regras do problema.
    """

    rows: list[dict] = []

    expected_ids = set(
        responses["participant_id"]
    )

    assigned_ids = set(
        assignment["participant_id"]
    )

    # ------------------------------------------------------
    # Todos os participantes foram alocados?
    # ------------------------------------------------------

    rows.append(
        {
            "scope": "global",
            "constraint": "all_participants_assigned",
            "item": "todos",
            "actual": len(assigned_ids),
            "expected": len(expected_ids),
            "passed": (
                assigned_ids
                == expected_ids
            ),
        }
    )

    # ------------------------------------------------------
    # Um único dia por participante?
    # ------------------------------------------------------

    counts_per_person = (
        assignment["participant_id"]
        .value_counts()
    )

    invalid_people = counts_per_person[
        counts_per_person != 1
    ]

    rows.append(
        {
            "scope": "global",
            "constraint": "one_day_per_participant",
            "item": "todos",
            "actual": (
                len(invalid_people)
            ),
            "expected": 0,
            "passed": (
                len(invalid_people)
                == 0
            ),
        }
    )

    # ------------------------------------------------------
    # Restrições individuais
    # ------------------------------------------------------

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

    violations = 0

    for row in assignment.itertuples(
        index=False
    ):
        if (
            row.participant_id,
            row.assigned_day,
        ) in blocked:
            violations += 1

    rows.append(
        {
            "scope": "global",
            "constraint": "individual_restrictions",
            "item": "todos",
            "actual": violations,
            "expected": 0,
            "passed": (
                violations == 0
            ),
        }
    )

    # ------------------------------------------------------
    # Limites totais por dia
    # ------------------------------------------------------

    day_counts = (
        assignment["assigned_day"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    for day in DAYS:

        count = int(
            day_counts[day]
        )

        lower, upper = (
            day_bounds[day]
        )

        rows.append(
            {
                "scope": "day",
                "constraint": "home_office_limit",
                "item": day,
                "actual": count,
                "expected": (
                    f"{lower} <= x <= {upper}"
                ),
                "passed": (
                    lower
                    <= count
                    <= upper
                ),
            }
        )

    # ------------------------------------------------------
    # Limites por equipe
    # ------------------------------------------------------

    for team in sorted(
        responses["team"].unique()
    ):

        team_assignment = assignment[
            assignment["team"] == team
        ]

        team_counts = (
            team_assignment["assigned_day"]
            .value_counts()
            .reindex(
                DAYS,
                fill_value=0,
            )
        )

        for day in DAYS:

            count = int(
                team_counts[day]
            )

            lower, upper = (
                team_day_bounds[
                    team
                ][day]
            )

            rows.append(
                {
                    "scope": "team_day",
                    "constraint": (
                        "team_home_office_limit"
                    ),
                    "item": (
                        f"{team}:{day}"
                    ),
                    "actual": count,
                    "expected": (
                        f"{lower} <= x <= {upper}"
                    ),
                    "passed": (
                        lower
                        <= count
                        <= upper
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_operational_reports(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
    restrictions: pd.DataFrame,
    day_bounds: Mapping[
        str,
        tuple[int, int],
    ],
    team_day_bounds: Mapping[
        str,
        Mapping[
            str,
            tuple[int, int],
        ],
    ],
    assignments_dir: str | Path,
    operational_dir: str | Path,
) -> dict[str, pd.DataFrame]:
    """
    Gera e salva todo o pacote operacional.
    """

    assignments_dir = Path(
        assignments_dir
    )

    operational_dir = Path(
        operational_dir
    )

    assignments_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    operational_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    weekly_schedule = (
        build_weekly_schedule(
            responses,
            assignment,
        )
    )

    home_office_by_day = (
        build_home_office_by_day(
            assignment
        )
    )

    occupancy_by_day = (
        build_occupancy_by_day(
            responses,
            assignment,
        )
    )

    occupancy_by_team = (
        build_occupancy_by_team(
            responses,
            assignment,
        )
    )

    preference_vs_assignment = (
        build_preference_vs_assignment(
            responses,
            assignment,
        )
    )

    constraint_validation = (
        validate_assignment(
            responses=responses,
            assignment=assignment,
            restrictions=restrictions,
            day_bounds=day_bounds,
            team_day_bounds=team_day_bounds,
        )
    )

    weekly_schedule.to_csv(
        assignments_dir
        / "weekly_schedule.csv",
        index=False,
    )

    home_office_by_day.to_csv(
        assignments_dir
        / "home_office_by_day.csv",
        index=False,
    )

    occupancy_by_day.to_csv(
        operational_dir
        / "occupancy_by_day.csv",
        index=False,
    )

    occupancy_by_team.to_csv(
        operational_dir
        / "occupancy_by_team.csv",
        index=False,
    )

    preference_vs_assignment.to_csv(
        operational_dir
        / "preference_vs_assignment.csv",
        index=False,
    )

    constraint_validation.to_csv(
        operational_dir
        / "constraint_validation.csv",
        index=False,
    )

    return {
        "weekly_schedule": weekly_schedule,
        "home_office_by_day": home_office_by_day,
        "occupancy_by_day": occupancy_by_day,
        "occupancy_by_team": occupancy_by_team,
        "preference_vs_assignment": (
            preference_vs_assignment
        ),
        "constraint_validation": (
            constraint_validation
        ),
    }