# Otimização de Home Office

Projeto em Python para otimizar a distribuição dos dias de home office,
conciliando as preferências dos colaboradores, restrições individuais e limites
de ocupação por equipe.

O modelo utiliza programação linear inteira com PuLP e compara a alocação
otimizada com uma distribuição aleatória. A avaliação inclui experimentos de
Monte Carlo, métricas, relatórios operacionais e visualizações.

## Funcionalidades

- geração de uma população sintética de colaboradores;
- otimização da alocação semanal com base no ranking de preferências;
- respeito às restrições individuais e aos limites por equipe e por dia;
- comparação entre a solução otimizada e uma alocação aleatória;
- execução de experimentos de Monte Carlo;
- geração de arquivos CSV, JSON e gráficos para análise dos resultados.

## Tecnologias

- Python
- pandas e NumPy
- PuLP
- Matplotlib e Seaborn
- PyYAML

## Instalação

```bash
python -m venv .venv
```

No Windows:

```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Execução

Para executar uma simulação individual:

```bash
python scripts/run_simulation.py
```

Para executar os experimentos de Monte Carlo:

```bash
python scripts/run_experiments.py
```

Os parâmetros do experimento podem ser alterados em `config.yaml`. Os
resultados são gravados no diretório `outputs/`.

## Estrutura do projeto

```text
data/       Dados simulados
docs/       Documentação da metodologia e do formulário
outputs/    Resultados, métricas e visualizações
scripts/    Pontos de entrada das simulações
src/        Geração, otimização, métricas e relatórios
```
