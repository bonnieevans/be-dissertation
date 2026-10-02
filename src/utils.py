"""Shared helpers for the MSOA-year panel pipeline: config loading, paths,
logging, and a DuckDB connection factory. Kept dependency-light on purpose."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import duckdb
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config() -> dict:
    with open(PROJECT_ROOT / "config" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_variable_mappings() -> dict:
    with open(PROJECT_ROOT / "config" / "variable_mappings.yaml") as f:
        return yaml.safe_load(f)


def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)


def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    log_dir = PROJECT_ROOT / "outputs" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / f"{name}.log", mode="w")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    return logger


def get_duckdb_connection(memory_limit: str = "8GB", threads: int | None = None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET memory_limit='{memory_limit}'")
    if threads:
        con.execute(f"SET threads={threads}")
    con.execute("SET preserve_insertion_order=false")
    return con
