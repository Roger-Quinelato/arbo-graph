"""
Roda um experimento e o baseline e0 no mesmo painel.

Uso: python scripts/run_experiment.py e0_baseline_rf_xgb [--painel dados_processados/painel.parquet]
     [--saida resultados/] [--publicar] [--sem-baseline]
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from arbo_core import config
from arbo_core.experiments import executar_experimento


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nome")
    parser.add_argument("--painel", type=Path, default=config.DADOS_PROCESSADOS_DIR / "painel.parquet")
    parser.add_argument("--saida", type=Path, default=None)
    parser.add_argument("--publicar", action="store_true", help="Copia RESULTADO.md para a pasta do experimento")
    parser.add_argument("--sem-baseline", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    painel = pd.read_parquet(args.painel)
    destino = executar_experimento(args.nome, painel, saida=args.saida,
                                   com_baseline=not args.sem_baseline, publicar=args.publicar)
    print(destino)


if __name__ == "__main__":
    main()
