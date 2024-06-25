#!/usr/bin/env python

from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.get_vocabulary import (
    VOCABULARY_ORDERING,
    VOCABULARY_ORDERING_METHODS,
)
from MEDS_polars_functions.utils import hydra_loguru_init


@hydra.main(version_base=None, config_path="../../configs", config_name="preprocess")
def main(cfg: DictConfig):
    """TODO."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    metadata_input_dir = Path(cfg.stage_cfg.metadata_input_dir)
    output_dir = Path(cfg.stage_cfg.output_dir)

    code_metadata = pl.read_parquet(metadata_input_dir.parent / "code_metadata.parquet", use_pyarrow=True)

    ordering_method = cfg.stage_cfg.get("ordering_method", VOCABULARY_ORDERING.LEXICOGRAPHIC)

    if ordering_method not in VOCABULARY_ORDERING_METHODS:
        raise ValueError(
            f"Invalid ordering method: {ordering_method}. "
            f"Expected one of {', '.join(VOCABULARY_ORDERING_METHODS.keys())}"
        )

    logger.info(f"Assigning code vocabulary indices via a {ordering_method} order.")
    ordering_fn = VOCABULARY_ORDERING_METHODS[ordering_method]

    code_modifiers = cfg.get("code_modifier_columns", None)
    if code_modifiers is None:
        code_modifiers = []

    code_metadata = ordering_fn(code_metadata, code_modifiers)

    output_fp = output_dir / "code_metadata.parquet"
    logger.info(f"Indices assigned. Writing to {output_fp}")

    code_metadata.write_parquet(output_fp, use_pyarrow=True)

    logger.info(f"Done with {cfg.stage}")


if __name__ == "__main__":
    main()
