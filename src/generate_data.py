from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd


DAYS = [
    "segunda",
    "terca",
    "quarta",
    "quinta",
    "sexta",
]

TEAMS = {
    "PD": 20,
    "PLATAFORMA": 50,
}

DEFAULT_DAY_WEIGHTS = {
    "segunda": 0.30,
    "terca": 0.10,
    "quarta": 0.10,
    "quinta": 0.15,
    "sexta": 0.35,
}

IMPORTANCE_LEVELS = [
    "pouco_importante",
    "indiferente",
    "importante",
    "muito_importante",
]

IMPORTANCE_PROBABILITIES = [
    0.10,
    0.15,
    0.45,
    0.30,
]


def _validate_day_weights(
    day_weights: Mapping[str, float],
) -> np.ndarray:
    """
    Valida e normaliza os pesos utilizados para gerar os rankings.
    """

    missing = set(DAYS) - set(day_weights)

    if missing:
        raise ValueError(
            f"Pesos ausentes para os dias: {sorted(missing)}"
        )

    weights = np.array(
        [float(day_weights[day]) for day in DAYS],
        dtype=float,
    )

    if np.any(weights < 0):
        raise ValueError(
            "Os pesos dos dias não podem ser negativos."
        )

    total = weights.sum()

    if total <= 0:
        raise ValueError(
            "A soma dos pesos dos dias deve ser maior que zero."
        )

    return weights / total


def _generate_ranking(
    rng: np.random.Generator,
    probabilities: np.ndarray,
) -> list[str]:
    """
    Gera uma permutação dos cinco dias.

    A seleção é feita sequencialmente e sem reposição.

    Dessa forma, segunda e sexta possuem maior probabilidade de
    aparecer nas primeiras posições, mas todos os rankings continuam
    sendo possíveis.
    """

    remaining_days = DAYS.copy()
    remaining_probabilities = probabilities.copy()

    ranking: list[str] = []

    while remaining_days:
        normalized = (
            remaining_probabilities
            / remaining_probabilities.sum()
        )

        index = rng.choice(
            len(remaining_days),
            p=normalized,
        )

        ranking.append(
            remaining_days.pop(index)
        )

        remaining_probabilities = np.delete(
            remaining_probabilities,
            index,
        )

    return ranking


def _generate_participants() -> list[dict]:
    """
    Cria os 70 participantes:

    - EMP0001 até EMP0020 -> P&D
    - EMP0021 até EMP0070 -> Plataforma
    """

    participants: list[dict] = []

    current_id = 1

    for team, quantity in TEAMS.items():
        for _ in range(quantity):
            participants.append(
                {
                    "participant_id": f"EMP{current_id:04d}",
                    "team": team,
                }
            )

            current_id += 1

    return participants


def _generate_restrictions(
    responses: pd.DataFrame,
    rng: np.random.Generator,
    restriction_rate: float,
) -> pd.DataFrame:
    """
    Gera restrições individuais simuladas.

    restriction_rate representa a proporção aproximada de pessoas
    que terão exatamente um dia inviável.

    Exemplo:
        restriction_rate = 0.05

    Em 70 pessoas, aproximadamente 3 ou 4 terão uma restrição.

    A restrição nunca será aplicada à primeira preferência da pessoa
    na geração sintética, pois queremos simular impedimentos
    operacionais e não respostas contraditórias propositalmente.
    """

    if not 0 <= restriction_rate <= 1:
        raise ValueError(
            "restriction_rate deve estar entre 0 e 1."
        )

    rows: list[dict] = []

    for _, person in responses.iterrows():
        if rng.random() >= restriction_rate:
            continue

        first_choice = person["pref_1"]

        possible_blocked_days = [
            day
            for day in DAYS
            if day != first_choice
        ]

        blocked_day = rng.choice(
            possible_blocked_days
        )

        rows.append(
            {
                "participant_id": person["participant_id"],
                "day": blocked_day,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "participant_id",
            "day",
        ],
    )


def generate_simulated_dataset(
    output_dir: str | Path,
    seed: int = 42,
    restriction_rate: float = 0.05,
    day_weights: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gera o conjunto de respostas simuladas para 70 colaboradores.

    Retorna
    -------
    responses:
        DataFrame com equipe, ranking e importância.

    restrictions:
        DataFrame com restrições individuais simuladas.
    """

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    rng = np.random.default_rng(seed)

    if day_weights is None:
        day_weights = DEFAULT_DAY_WEIGHTS

    probabilities = _validate_day_weights(
        day_weights
    )

    participants = _generate_participants()

    response_rows: list[dict] = []

    for participant in participants:
        ranking = _generate_ranking(
            rng=rng,
            probabilities=probabilities,
        )

        importance = rng.choice(
            IMPORTANCE_LEVELS,
            p=IMPORTANCE_PROBABILITIES,
        )

        response_rows.append(
            {
                "participant_id": participant["participant_id"],
                "team": participant["team"],
                "pref_1": ranking[0],
                "pref_2": ranking[1],
                "pref_3": ranking[2],
                "pref_4": ranking[3],
                "pref_5": ranking[4],
                "importance": importance,
            }
        )

    responses = pd.DataFrame(response_rows)

    restrictions = _generate_restrictions(
        responses=responses,
        rng=rng,
        restriction_rate=restriction_rate,
    )

    responses_path = (
        output_dir
        / "responses_70.csv"
    )

    restrictions_path = (
        output_dir
        / "restrictions.csv"
    )

    responses.to_csv(
        responses_path,
        index=False,
    )

    restrictions.to_csv(
        restrictions_path,
        index=False,
    )

    return responses, restrictions


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Gera dados simulados para o experimento "
            "de alocação de home office."
        )
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/simulated",
        help="Diretório onde os CSVs serão salvos.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed para reprodução do experimento.",
    )

    parser.add_argument(
        "--restriction-rate",
        type=float,
        default=0.05,
        help=(
            "Proporção de participantes com uma "
            "restrição individual simulada."
        ),
    )

    args = parser.parse_args()

    responses, restrictions = (
        generate_simulated_dataset(
            output_dir=args.output_dir,
            seed=args.seed,
            restriction_rate=args.restriction_rate,
        )
    )

    print("=" * 70)
    print("DADOS SIMULADOS GERADOS")
    print("=" * 70)

    print(
        f"Participantes: {len(responses)}"
    )

    print()

    print(
        responses["team"]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Primeiras preferências:"
    )

    print(
        responses["pref_1"]
        .value_counts()
        .reindex(DAYS, fill_value=0)
        .to_string()
    )

    print()

    print(
        f"Restrições geradas: "
        f"{len(restrictions)}"
    )


if __name__ == "__main__":
    main()