#!/usr/bin/env python

import copy
import json
import random
from functools import partial
from pathlib import Path

import hydra
import polars as pl
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from MEDS_polars_functions.event_conversion import convert_to_events
from MEDS_polars_functions.mapper import wrap as rwlock_wrap
from MEDS_polars_functions.utils import hydra_loguru_init, write_lazyframe


@hydra.main(version_base=None, config_path="../../configs", config_name="extraction")
def main(cfg: DictConfig):
    """Converts the sub-sharded or raw data into events which are sharded by patient X input shard."""

    hydra_loguru_init()

    logger.info(
        f"Running with config:\n{OmegaConf.to_yaml(cfg)}\n"
        f"Stage: {cfg.stage}\n\n"
        f"Stage config:\n{OmegaConf.to_yaml(cfg.stage_cfg)}"
    )

    shards = json.loads((Path(cfg.stage_cfg.metadata_input_dir) / "splits.json").read_text())

    event_conversion_cfg_fp = Path(cfg.event_conversion_config_fp)
    if not event_conversion_cfg_fp.exists():
        raise FileNotFoundError(f"Event conversion config file not found: {event_conversion_cfg_fp}")

    logger.info("Starting event conversion.")

    logger.info(f"Reading event conversion config from {event_conversion_cfg_fp}")
    event_conversion_cfg = OmegaConf.load(event_conversion_cfg_fp)
    logger.info(f"Event conversion config:\n{OmegaConf.to_yaml(event_conversion_cfg)}")

    default_patient_id_col = event_conversion_cfg.pop("patient_id_col", "patient_id")

    patient_subsharded_dir = Path(cfg.stage_cfg.output_dir)
    patient_subsharded_dir.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(event_conversion_cfg, patient_subsharded_dir / "event_conversion_config.yaml")

    patient_splits = list(shards.items())
    random.shuffle(patient_splits)

    event_configs = list(event_conversion_cfg.items())
    random.shuffle(event_configs)

    # Here, we'll be reading files directly, so we'll turn off globbing
    read_fn = partial(pl.scan_parquet, glob=False)

    for sp, patients in patient_splits:
        for input_prefix, event_cfgs in event_configs:
            event_cfgs = copy.deepcopy(event_cfgs)
            input_patient_id_column = event_cfgs.pop("patient_id_col", default_patient_id_col)

            event_shards = list((Path(cfg.stage_cfg.data_input_dir) / input_prefix).glob("*.parquet"))
            random.shuffle(event_shards)

            for shard_fp in event_shards:
                out_fp = patient_subsharded_dir / sp / input_prefix / shard_fp.name
                logger.info(f"Converting {shard_fp} to events and saving to {out_fp}")

                def compute_fn(df: pl.LazyFrame) -> pl.LazyFrame:
                    typed_patients = pl.Series(patients, dtype=df.schema[input_patient_id_column], strict=False)

                    if input_patient_id_column != "patient_id":
                        df = df.rename({input_patient_id_column: "patient_id"})

                    try:
                        logger.info(f"Extracting events for {input_prefix}/{shard_fp.name}")
                        return convert_to_events(
                            df.filter(pl.col("patient_id").is_in(typed_patients)),
                            event_cfgs=copy.deepcopy(event_cfgs),
                        )
                    except Exception as e:
                        raise ValueError(
                            f"Error converting {str(shard_fp.resolve())} for {sp}/{input_prefix}: {e}"
                        ) from e

                rwlock_wrap(
                    shard_fp, out_fp, read_fn, write_lazyframe, compute_fn, do_overwrite=cfg.do_overwrite
                )

    logger.info("Subsharded into converted events.")


if __name__ == "__main__":
    main()
