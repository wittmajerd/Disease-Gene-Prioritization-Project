import json
import yaml
import logging
from datetime import datetime
from pathlib import Path


def _pretty(obj) -> str:
    return json.dumps(obj, indent=4)


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _setup_run_dir(run_name: str | None) -> Path:
    runs_root = _project_root() / "saves"
    runs_root.mkdir(parents=True, exist_ok=True)
    if run_name is None or not run_name.strip():
        run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = runs_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _setup_logger(run_dir: Path) -> logging.Logger:
    logger = logging.getLogger("biokg.pipeline")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    return logger


def _save_yaml(path: Path, payload) -> None:
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

def _append_jsonl(path: Path, payload) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")