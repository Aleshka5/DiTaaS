from __future__ import annotations

import argparse
import json
import os

import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient

from src.config import get_settings


def parse_args() -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Получить список run_id для эксперимента MLflow по имени."
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        required=True,
        help="Название эксперимента в MLflow.",
    )
    parser.add_argument(
        "--tracking-uri",
        type=str,
        default=settings.mlflow_tracking_uri,
        help="MLflow Tracking URI. По умолчанию берётся из MLFLOW_TRACKING_URI.",
    )
    parser.add_argument(
        "--include-deleted",
        action="store_true",
        help="Включить удалённые run в результаты.",
    )
    return parser.parse_args()


def _search_runs_all_pages(
    *,
    client: MlflowClient,
    experiment_id: str,
    run_view_type: ViewType,
) -> list[str]:
    run_ids: list[str] = []
    page_token: str | None = None

    while True:
        runs_page = client.search_runs(
            experiment_ids=[experiment_id],
            run_view_type=run_view_type,
            max_results=50000,
            page_token=page_token,
        )
        run_ids.extend(run.info.run_id for run in runs_page)
        page_token = getattr(runs_page, "token", None)
        if not page_token:
            break

    return run_ids


def main() -> None:
    args = parse_args()
    settings = get_settings()
    if not args.tracking_uri:
        raise ValueError("Не задан tracking URI. Укажите --tracking-uri или MLFLOW_TRACKING_URI.")

    if settings.mlflow_tracking_username and not os.getenv("MLFLOW_TRACKING_USERNAME"):
        os.environ["MLFLOW_TRACKING_USERNAME"] = settings.mlflow_tracking_username
    if settings.mlflow_tracking_password and not os.getenv("MLFLOW_TRACKING_PASSWORD"):
        os.environ["MLFLOW_TRACKING_PASSWORD"] = settings.mlflow_tracking_password

    mlflow.set_tracking_uri(args.tracking_uri)
    client = MlflowClient()
    experiment = client.get_experiment_by_name(args.experiment_name)
    if experiment is None:
        raise ValueError(f"Эксперимент '{args.experiment_name}' не найден.")

    run_view_type = ViewType.ALL if args.include_deleted else ViewType.ACTIVE_ONLY
    run_ids = _search_runs_all_pages(
        client=client,
        experiment_id=experiment.experiment_id,
        run_view_type=run_view_type,
    )
    print(json.dumps(run_ids, ensure_ascii=False))


if __name__ == "__main__":
    main()
