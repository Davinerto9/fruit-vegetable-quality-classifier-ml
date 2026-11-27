"""
Punto de entrada del proyecto: cada paso es un comando.

Uso (desde la raíz del repositorio, con el ambiente activado):
    python -m src.main todo          # pasos 1 a 6 seguidos; se salta los que ya están al día

    python -m src.main inventario    # 1. lista todas las fotos con su fruta, estado y grupo
    python -m src.main recortar      # 2. separa la fruta del fondo y guarda un recorte 128x128
    python -m src.main dividir       # 3. reparte en train / val / test por grupos
    python -m src.main atributos     # 4. calcula los 359 atributos de cada recorte
    python -m src.main entrenar      # 5. entrena y guarda los modelos en models/
    python -m src.main evaluar       # 6. mide los modelos en val

    python -m src.main fuga          # (opcional) cuánto se infla la nota sin agrupar
    python -m src.main congelar      # al final: huella de modelos y reparto
    python -m src.main evaluar --test --confirmo    # test, una sola vez
    python -m src.main celular <paso>               # prueba con fotos de celular

    streamlit run app/app.py         # la aplicación

Cada paso lee lo que dejó el anterior:
    data/raw/ -> index.csv -> crops/ + manifest.csv -> splits.csv -> features.npz -> models/
"""
from __future__ import annotations

import argparse

from src import config


def step_inventory() -> None:
    from src.data import inventory

    records = inventory.build_inventory()
    inventory.write_inventory(records)
    print(f"{len(records)} fotos registradas en {config.INDEX_CSV.relative_to(config.ROOT)}")


def step_crop() -> None:
    from src.data import prepare

    prepare.run(pilot=False)


def step_split() -> None:
    from src.data import split

    split.run()


def step_features() -> None:
    from src.data import features

    features.build()


def step_train(only: str | None = None) -> None:
    from src.training import train_models

    train_models.run(only)


def step_evaluate(test: bool = False, confirm: bool = False) -> None:
    from src.evaluation import evaluate_models

    evaluate_models.run("test" if test else "val", confirm)


def up_to_date(outputs: list, inputs: list) -> bool:
    """True si todas las salidas existen y son más nuevas que todas las entradas que existen."""
    if not all(path.is_file() for path in outputs):
        return False
    oldest = min(path.stat().st_mtime for path in outputs)
    return all(oldest >= path.stat().st_mtime for path in inputs if path.is_file())


MANIFEST = config.CROPS_DIR / "manifest.csv"
MODELS = [config.MODELS_DIR / name for name in config.MODEL_FILES.values()]

# (nombre, función, ¿ya está hecho?). "todo" se salta un paso si su salida existe y es más
# nueva que su entrada; así, al agregar fotos de celular etiquetadas, rehace desde "dividir".
PIPELINE = (
    ("inventario", step_inventory, lambda: config.INDEX_CSV.is_file()),
    ("recortar", step_crop, lambda: MANIFEST.is_file()),
    ("dividir", step_split, lambda: up_to_date([config.SPLITS_CSV], [MANIFEST, config.PHONE_TRAIN_CSV])),
    ("atributos", step_features, lambda: up_to_date([config.FEATURES_NPZ], [config.SPLITS_CSV])),
    ("entrenar", step_train, lambda: up_to_date(MODELS, [config.FEATURES_NPZ])),
    ("evaluar", step_evaluate, lambda: False),
)


def run_all() -> None:
    """
    Corre los pasos en orden y se salta los que ya están al día (para rehacer un paso a la
    fuerza, córrelo solo; para rehacer todo, borra data/processed/ y models/).
    """
    for number, (name, function, done) in enumerate(PIPELINE, start=1):
        if done():
            print(f"[{number}/{len(PIPELINE)}] {name}: ya está al día, se salta.")
            continue
        print(f"[{number}/{len(PIPELINE)}] {name}…")
        function()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.main", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="paso", required=True)
    sub.add_parser("todo", help="pasos 1 a 6 en orden")
    sub.add_parser("inventario", help="1. index.csv con todas las fotos")
    sub.add_parser("recortar", help="2. segmentación y recortes 128x128")
    sub.add_parser("dividir", help="3. train / val / test por grupos")
    sub.add_parser("atributos", help="4. 359 atributos de color, textura y forma")
    train = sub.add_parser("entrenar", help="5. entrena y guarda en models/")
    train.add_argument("--solo", choices=("clasicos", "cnn"))
    evaluate = sub.add_parser("evaluar", help="6. métricas en val (o test con --test)")
    evaluate.add_argument("--test", action="store_true")
    evaluate.add_argument("--confirmo", action="store_true")
    sub.add_parser("congelar", help="huella de modelos y reparto antes de test")
    sub.add_parser("fuga", help="prueba de fuga por sesiones de AgriFreshNET")
    phone = sub.add_parser("celular", help="prueba con fotos de celular etiquetadas a ciegas")
    phone.add_argument("accion", choices=("ingerir", "hojas", "acuerdo", "finalizar", "evaluar"))
    phone.add_argument("--confirmo", action="store_true")
    args = parser.parse_args(argv)

    if args.paso == "todo":
        run_all()
    elif args.paso == "entrenar":
        step_train(args.solo)
    elif args.paso == "evaluar":
        step_evaluate(args.test, args.confirmo)
    elif args.paso == "congelar":
        from src.evaluation.evaluate_models import freeze
        freeze()
    elif args.paso == "fuga":
        from src.evaluation.leak_check import main as leak_check
        leak_check()
    elif args.paso == "celular":
        from src.evaluation.phone_test import run
        run(args.accion, confirm=args.confirmo)
    else:
        dict((name, function) for name, function, _ in PIPELINE)[args.paso]()


if __name__ == "__main__":
    main()
