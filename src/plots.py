from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns

from .load_data import DAYS
from .metrics import (
    day_distribution,
    rank_distribution,
)


DAY_LABELS = {
    "segunda": "Segunda",
    "terca": "Terça",
    "quarta": "Quarta",
    "quinta": "Quinta",
    "sexta": "Sexta",
}


# Cor visual usada no projeto.
# Se vocês conseguirem o HEX oficial da BemAgro,
# basta trocar esta constante.
BEMAGRO_GREEN = "#2E7D32"
BEMAGRO_LIGHT_GREEN = "#C8E6C9"
OFFICE_COLOR = "#EEEEEE"


def _prepare_output_dir(
    output_dir: str | Path,
) -> Path:

    path = Path(output_dir)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )

    return path


def _apply_clean_style() -> None:
    """
    Configuração visual comum.
    """

    sns.set_theme(
        style="whitegrid",
        context="notebook",
    )


def plot_first_choice_distribution(
    responses: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Distribuição das primeiras preferências.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    counts = (
        responses["pref_1"]
        .value_counts()
        .reindex(
            DAYS,
            fill_value=0,
        )
    )

    data = pd.DataFrame(
        {
            "day": [
                DAY_LABELS[day]
                for day in DAYS
            ],
            "count": counts.values,
        }
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=data,
        x="day",
        y="count",
        color=BEMAGRO_GREEN,
        ax=ax,
    )

    ax.set_title(
        "Primeira preferência de Home Office"
    )

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "Número de colaboradores"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
        )

    fig.tight_layout()

    path = (
        output_dir
        / "first_choice_distribution.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_assignment_distribution(
    assignment: pd.DataFrame,
    output_dir: str | Path,
    filename: str = (
        "assignment_distribution.png"
    ),
    title: str = (
        "Distribuição final de Home Office"
    ),
) -> Path:
    """
    Quantidade de pessoas em HO por dia.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    distribution = day_distribution(
        assignment
    )

    distribution["day_label"] = (
        distribution["day"]
        .map(DAY_LABELS)
    )

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=distribution,
        x="day_label",
        y="home_office",
        color=BEMAGRO_GREEN,
        ax=ax,
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "Pessoas em Home Office"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
        )

    fig.tight_layout()

    path = output_dir / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_rank_distribution(
    assignment: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "rank_distribution.png",
    title: str = "Preferência recebida",
) -> Path:
    """
    Quantidade de pessoas em cada posição
    recebida do ranking.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    distribution = rank_distribution(
        assignment
    )

    distribution["rank_label"] = [
        "1ª",
        "2ª",
        "3ª",
        "4ª",
        "5ª",
    ]

    fig, ax = plt.subplots(
        figsize=(9, 5)
    )

    sns.barplot(
        data=distribution,
        x="rank_label",
        y="count",
        color=BEMAGRO_GREEN,
        ax=ax,
    )

    ax.set_title(title)

    ax.set_xlabel(
        "Posição recebida no ranking"
    )

    ax.set_ylabel(
        "Número de colaboradores"
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%d",
        )

    fig.tight_layout()

    path = output_dir / filename

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_method_comparison(
    summary: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Compara satisfação média do sorteio
    com a otimização.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    sns.barplot(
        data=summary,
        x="method",
        y="average_score",
        color=BEMAGRO_GREEN,
        ax=ax,
    )

    ax.set_title(
        "Satisfação média por método"
    )

    ax.set_xlabel(
        "Método"
    )

    ax.set_ylabel(
        "Satisfação média"
    )

    ax.set_ylim(
        0,
        5,
    )

    for container in ax.containers:
        ax.bar_label(
            container,
            fmt="%.2f",
        )

    fig.tight_layout()

    path = (
        output_dir
        / "method_comparison.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_weekly_schedule_heatmap(
    responses: pd.DataFrame,
    assignment: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Mapa de calor dos 70 colaboradores.

    Cada linha = um colaborador.
    Cada coluna = um dia.

    Verde = Home Office.
    Cinza = Escritório.

    Como cada pessoa recebe um único dia,
    haverá exatamente uma célula verde
    por linha.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    ordered = (
        responses[
            [
                "participant_id",
                "team",
            ]
        ]
        .sort_values(
            [
                "team",
                "participant_id",
            ]
        )
        .reset_index(drop=True)
    )

    assigned_lookup = dict(
        zip(
            assignment["participant_id"],
            assignment["assigned_day"],
        )
    )

    matrix_rows: list[list[int]] = []
    annotations: list[list[str]] = []
    y_labels: list[str] = []

    for person in ordered.itertuples(
        index=False
    ):

        participant_id = (
            person.participant_id
        )

        team = person.team

        assigned_day = assigned_lookup[
            participant_id
        ]

        row: list[int] = []
        annotation_row: list[str] = []

        for day in DAYS:

            is_home_office = (
                day == assigned_day
            )

            row.append(
                1 if is_home_office else 0
            )

            annotation_row.append(
                "HO"
                if is_home_office
                else ""
            )

        matrix_rows.append(row)
        annotations.append(
            annotation_row
        )

        y_labels.append(
            f"{participant_id} · {team}"
        )

    matrix = pd.DataFrame(
        matrix_rows,
        columns=[
            DAY_LABELS[day]
            for day in DAYS
        ],
        index=y_labels,
    )

    annotation_df = pd.DataFrame(
        annotations,
        columns=matrix.columns,
        index=matrix.index,
    )

    cmap = ListedColormap(
        [
            OFFICE_COLOR,
            BEMAGRO_GREEN,
        ]
    )

    # Com 70 pessoas o gráfico precisa
    # deliberadamente ser alto.
    fig, ax = plt.subplots(
        figsize=(8, 24)
    )

    sns.heatmap(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=1,
        cbar=False,
        linewidths=0.5,
        linecolor="white",
        annot=annotation_df,
        fmt="",
        annot_kws={
            "fontsize": 7,
            "fontweight": "bold",
        },
        ax=ax,
    )

    ax.set_title(
        "Mapa semanal de alocação — Home Office",
        fontsize=15,
        pad=15,
    )

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "Colaborador"
    )

    ax.tick_params(
        axis="y",
        labelsize=7,
    )

    ax.tick_params(
        axis="x",
        labelsize=10,
        rotation=0,
    )

    legend = [
        Patch(
            facecolor=BEMAGRO_GREEN,
            label="Home Office",
        ),
        Patch(
            facecolor=OFFICE_COLOR,
            label="Escritório",
        ),
    ]

    ax.legend(
        handles=legend,
        loc="upper left",
        bbox_to_anchor=(
            1.02,
            1.0,
        ),
        borderaxespad=0,
    )

    fig.tight_layout()

    path = (
        output_dir
        / "weekly_schedule_heatmap.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_occupancy_percentage(
    occupancy_by_day: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Percentual em HO e no escritório por dia.

    Barras empilhadas totalizando 100%.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    data = (
        occupancy_by_day
        .set_index("day_label")
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        data.index,
        data["office_pct"],
        label="Escritório",
        color=OFFICE_COLOR,
    )

    ax.bar(
        data.index,
        data["home_office_pct"],
        bottom=data["office_pct"],
        label="Home Office",
        color=BEMAGRO_GREEN,
    )

    for index, row in enumerate(
        data.itertuples()
    ):

        ax.text(
            index,
            row.office_pct / 2,
            f"{row.office_pct:.1f}%\n"
            f"({row.office})",
            ha="center",
            va="center",
            fontsize=9,
        )

        ax.text(
            index,
            (
                row.office_pct
                + row.home_office_pct / 2
            ),
            f"{row.home_office_pct:.1f}%\n"
            f"({row.home_office})",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_ylim(
        0,
        100,
    )

    ax.set_title(
        "Ocupação diária do escritório"
    )

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "Percentual dos colaboradores"
    )

    ax.legend(
        loc="upper right"
    )

    fig.tight_layout()

    path = (
        output_dir
        / "occupancy_percentage.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_team_home_office_percentage(
    occupancy_by_team: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Percentual de cada equipe em HO por dia.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=occupancy_by_team,
        x="day_label",
        y="home_office_pct",
        hue="team",
        ax=ax,
    )

    ax.set_title(
        "Home Office por equipe"
    )

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "% da equipe em Home Office"
    )

    ax.set_ylim(
        0,
        max(
            30,
            occupancy_by_team[
                "home_office_pct"
            ].max()
            + 5,
        ),
    )

    ax.legend(
        title="Equipe"
    )

    fig.tight_layout()

    path = (
        output_dir
        / "team_home_office_percentage.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path


def plot_preference_vs_assignment(
    preference_vs_assignment: pd.DataFrame,
    output_dir: str | Path,
) -> Path:
    """
    Compara demanda de 1ª escolha
    com a quantidade finalmente alocada.
    """

    _apply_clean_style()

    output_dir = _prepare_output_dir(
        output_dir
    )

    data = preference_vs_assignment[
        [
            "day_label",
            "first_choice_demand",
            "assigned_home_office",
        ]
    ].copy()

    long_data = data.melt(
        id_vars="day_label",
        var_name="metric",
        value_name="people",
    )

    long_data["metric"] = (
        long_data["metric"].map(
            {
                "first_choice_demand": (
                    "Queriam como 1ª opção"
                ),
                "assigned_home_office": (
                    "Receberam Home Office"
                ),
            }
        )
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    sns.barplot(
        data=long_data,
        x="day_label",
        y="people",
        hue="metric",
        ax=ax,
    )

    ax.set_title(
        "Preferência inicial × alocação final"
    )

    ax.set_xlabel(
        "Dia da semana"
    )

    ax.set_ylabel(
        "Número de colaboradores"
    )

    ax.legend(
        title=""
    )

    fig.tight_layout()

    path = (
        output_dir
        / "preference_vs_assignment.png"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.close(fig)

    return path