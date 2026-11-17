from __future__ import annotations

from pathlib import Path

import pandas as pd


DAYS = [
    "segunda",
    "terca",
    "quarta",
    "quinta",
    "sexta",
]

VALID_TEAMS = {
    "PD",
    "PLATAFORMA",
}

PREFERENCE_COLUMNS = [
    "pref_1",
    "pref_2",
    "pref_3",
    "pref_4",
    "pref_5",
]


def load_responses(
    path: str | Path,
) -> pd.DataFrame:
    """
    Carrega e valida o arquivo de respostas.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}"
        )

    df = pd.read_csv(path)

    required_columns = {
        "participant_id",
        "team",
        *PREFERENCE_COLUMNS,
        "importance",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            f"{sorted(missing)}"
        )

    if df["participant_id"].duplicated().any():
        duplicates = (
            df.loc[
                df["participant_id"].duplicated(),
                "participant_id",
            ]
            .tolist()
        )

        raise ValueError(
            "IDs duplicados encontrados: "
            f"{duplicates}"
        )

    invalid_teams = (
        set(df["team"].unique())
        - VALID_TEAMS
    )

    if invalid_teams:
        raise ValueError(
            "Equipes inválidas encontradas: "
            f"{sorted(invalid_teams)}"
        )

    expected_days = set(DAYS)

    for _, row in df.iterrows():
        ranking = [
            row[column]
            for column in PREFERENCE_COLUMNS
        ]

        if set(ranking) != expected_days:
            raise ValueError(
                "Ranking inválido para "
                f"{row['participant_id']}: "
                f"{ranking}"
            )

        if len(ranking) != len(set(ranking)):
            raise ValueError(
                "Ranking contém dias repetidos para "
                f"{row['participant_id']}: "
                f"{ranking}"
            )

    return df


def load_restrictions(
    path: str | Path,
    responses: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Carrega as restrições individuais.

    O arquivo pode estar vazio.
    """

    path = Path(path)

    if not path.exists():
        return pd.DataFrame(
            columns=[
                "participant_id",
                "day",
            ]
        )

    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "day",
            ]
        )

    if df.empty:
        return pd.DataFrame(
            columns=[
                "participant_id",
                "day",
            ]
        )

    required_columns = {
        "participant_id",
        "day",
    }

    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(
            "Colunas obrigatórias ausentes em restrictions.csv: "
            f"{sorted(missing)}"
        )

    invalid_days = set(df["day"]) - set(DAYS)

    if invalid_days:
        raise ValueError(
            "Dias inválidos nas restrições: "
            f"{sorted(invalid_days)}"
        )

    if responses is not None:
        valid_ids = set(
            responses["participant_id"]
        )

        unknown_ids = (
            set(df["participant_id"])
            - valid_ids
        )

        if unknown_ids:
            raise ValueError(
                "Restrições associadas a IDs desconhecidos: "
                f"{sorted(unknown_ids)}"
            )

    df = (
        df[
            [
                "participant_id",
                "day",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    return df


def responses_to_long(
    responses: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte o ranking para representação longa.

    Exemplo:

    EMP0001 | sexta | rank 1 | score 5
    EMP0001 | segunda | rank 2 | score 4
    ...
    """

    rows: list[dict] = []

    for _, person in responses.iterrows():
        for rank in range(1, 6):
            day = person[f"pref_{rank}"]

            score = 6 - rank

            rows.append(
                {
                    "participant_id": (
                        person["participant_id"]
                    ),
                    "team": person["team"],
                    "day": day,
                    "rank": rank,
                    "score": score,
                    "importance": (
                        person["importance"]
                    ),
                }
            )

    return pd.DataFrame(rows)


def preference_lookup(
    responses: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """
    Retorna:

    (participant_id, day) -> score
    """

    long_df = responses_to_long(
        responses
    )

    return {
        (
            row.participant_id,
            row.day,
        ): int(row.score)
        for row in long_df.itertuples(
            index=False
        )
    }


def rank_lookup(
    responses: pd.DataFrame,
) -> dict[tuple[str, str], int]:
    """
    Retorna:

    (participant_id, day) -> rank
    """

    long_df = responses_to_long(
        responses
    )

    return {
        (
            row.participant_id,
            row.day,
        ): int(row.rank)
        for row in long_df.itertuples(
            index=False
        )
    }