"""
Validação de equivalência (T9): repo novo contra o pipeline antigo (commit 402f845).

Roda na máquina que tem as fontes. Uso:
    ARBO_DADOS_DIR=D:\\arboRoger\\dados python scripts/check_equivalence.py [--saida DIR]
        [--repo-antigo CAMINHO] [--ano-teste 2025]

Sai com código 1 se não houver equivalência; o relatório fica em <saida>/equivalencia.json.
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from arbo_core import config
from arbo_core.equivalence import REPO_ANTIGO_PADRAO, verificar


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--saida", type=Path, default=config.RESULTADOS_DIR / "equivalencia")
    parser.add_argument("--repo-antigo", type=Path, default=REPO_ANTIGO_PADRAO)
    parser.add_argument("--ano-teste", type=int, default=2025)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(name)s | %(levelname)s | %(message)s")
    relatorio = verificar(args.saida, ano_teste=args.ano_teste, repo_antigo=args.repo_antigo)
    print(json.dumps(relatorio.to_dict(), indent=2, ensure_ascii=False, default=str))
    sys.exit(0 if relatorio.equivalente else 1)


if __name__ == "__main__":
    main()
