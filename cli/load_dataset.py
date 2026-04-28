from __future__ import annotations

import argparse
from pathlib import Path

from src.config import get_settings
from src.repositories.mlflow import MLFlowRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Скачать части SFT-датасета из MLflow в одну папку."
    )
    parser.add_argument(
        "--run-ids",
        nargs="*",
        default=None,
        help="Список run_id. Если не передано, используются MLFLOW_DATASET_RUN_IDS из .env",
    )
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--artifact-path", type=str, default=None)
    parser.add_argument("--tracking-uri", type=str, default=None)
    parser.add_argument("--registry-uri", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()

    run_ids = args.run_ids or settings.dataset_run_ids
    if not run_ids:
        raise ValueError("Не переданы run_id. Укажите --run-ids или MLFLOW_DATASET_RUN_IDS.")

    output_dir = Path(args.output_dir) if args.output_dir else settings.dataset_download_dir
    artifact_path = args.artifact_path or settings.mlflow_dataset_artifact_path

    repository = MLFlowRepository(
        tracking_uri=args.tracking_uri,
        registry_uri=args.registry_uri,
    )
    files = repository.download_sft_dataset_parts(
        run_ids=run_ids,
        output_dir=output_dir,
        artifact_subpath=artifact_path,
    )
    print(f"Downloaded {len(files)} .sft files into: {output_dir}")


if __name__ == "__main__":
    main()
