"""Base helper code and data inputs for all transforms integration tests.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import json
import os
import tempfile
from io import StringIO
from pathlib import Path

import numpy as np
import polars as pl
import rootutils
from nested_ragged_tensors.ragged_numpy import JointNestedRaggedTensorDict

from .utils import assert_df_equal, parse_meds_csvs, run_command

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

code_root = root / "src" / "MEDS_transforms"
transforms_root = code_root / "transforms"
filters_root = code_root / "filters"

if os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1":
    # Root Source
    FIT_VOCABULARY_INDICES_SCRIPT = code_root / "fit_vocabulary_indices.py"

    # Filters
    FILTER_MEASUREMENTS_SCRIPT = filters_root / "filter_measurements.py"
    FILTER_PATIENTS_SCRIPT = filters_root / "filter_patients.py"

    # Transforms
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT = transforms_root / "add_time_derived_measurements.py"
    REORDER_MEASUREMENTS_SCRIPT = transforms_root / "reorder_measurements.py"
    EXTRACT_VALUES_SCRIPT = transforms_root / "extract_values.py"
    NORMALIZATION_SCRIPT = transforms_root / "normalization.py"
    OCCLUDE_OUTLIERS_SCRIPT = transforms_root / "occlude_outliers.py"
    TENSORIZATION_SCRIPT = transforms_root / "tensorization.py"
    TOKENIZATION_SCRIPT = transforms_root / "tokenization.py"
else:
    # Root Source
    FIT_VOCABULARY_INDICES_SCRIPT = "MEDS_transform-fit_vocabulary_indices"

    # Filters
    FILTER_MEASUREMENTS_SCRIPT = "MEDS_transform-filter_measurements"
    FILTER_PATIENTS_SCRIPT = "MEDS_transform-filter_patients"

    # Transforms
    ADD_TIME_DERIVED_MEASUREMENTS_SCRIPT = "MEDS_transform-add_time_derived_measurements"
    REORDER_MEASUREMENTS_SCRIPT = "MEDS_transform-reorder_measurements"
    EXTRACT_VALUES_SCRIPT = "MEDS_transform-extract_values"
    NORMALIZATION_SCRIPT = "MEDS_transform-normalization"
    OCCLUDE_OUTLIERS_SCRIPT = "MEDS_transform-occlude_outliers"
    TENSORIZATION_SCRIPT = "MEDS_transform-tensorization"
    TOKENIZATION_SCRIPT = "MEDS_transform-tokenization"

# Test MEDS data (inputs)

SPLITS = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

MEDS_TRAIN_0 = """
patient_id,time,code,numeric_value
239684,,EYE_COLOR//BROWN,
239684,,HEIGHT,175.271115221764
239684,"12/28/1980, 00:00:00",DOB,
239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
239684,"05/11/2010, 17:41:51",HR,102.6
239684,"05/11/2010, 17:41:51",TEMP,96.0
239684,"05/11/2010, 17:48:48",HR,105.1
239684,"05/11/2010, 17:48:48",TEMP,96.2
239684,"05/11/2010, 18:25:35",HR,113.4
239684,"05/11/2010, 18:25:35",TEMP,95.8
239684,"05/11/2010, 18:57:18",HR,112.6
239684,"05/11/2010, 18:57:18",TEMP,95.5
239684,"05/11/2010, 19:27:19",DISCHARGE,
1195293,,EYE_COLOR//BLUE,
1195293,,HEIGHT,164.6868838269085
1195293,"06/20/1978, 00:00:00",DOB,
1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
1195293,"06/20/2010, 19:23:52",HR,109.0
1195293,"06/20/2010, 19:23:52",TEMP,100.0
1195293,"06/20/2010, 19:25:32",HR,114.1
1195293,"06/20/2010, 19:25:32",TEMP,100.0
1195293,"06/20/2010, 19:45:19",HR,119.8
1195293,"06/20/2010, 19:45:19",TEMP,99.9
1195293,"06/20/2010, 20:12:31",HR,112.5
1195293,"06/20/2010, 20:12:31",TEMP,99.8
1195293,"06/20/2010, 20:24:44",HR,107.7
1195293,"06/20/2010, 20:24:44",TEMP,100.0
1195293,"06/20/2010, 20:41:33",HR,107.5
1195293,"06/20/2010, 20:41:33",TEMP,100.4
1195293,"06/20/2010, 20:50:04",DISCHARGE,
"""

MEDS_TRAIN_1 = """
patient_id,time,code,numeric_value
68729,,EYE_COLOR//HAZEL,
68729,,HEIGHT,160.3953106166676
68729,"03/09/1978, 00:00:00",DOB,
68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
68729,"05/26/2010, 02:30:56",HR,86.0
68729,"05/26/2010, 02:30:56",TEMP,97.8
68729,"05/26/2010, 04:51:52",DISCHARGE,
814703,,EYE_COLOR//HAZEL,
814703,,HEIGHT,156.48559093209357
814703,"03/28/1976, 00:00:00",DOB,
814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
814703,"02/05/2010, 05:55:39",HR,170.2
814703,"02/05/2010, 05:55:39",TEMP,100.1
814703,"02/05/2010, 07:02:30",DISCHARGE,
"""

MEDS_TUNING_0 = """
patient_id,time,code,numeric_value
754281,,EYE_COLOR//BROWN,
754281,,HEIGHT,166.22261567137025
754281,"12/19/1988, 00:00:00",DOB,
754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
754281,"01/03/2010, 06:27:59",HR,142.0
754281,"01/03/2010, 06:27:59",TEMP,99.8
754281,"01/03/2010, 08:22:13",DISCHARGE,
"""

MEDS_HELD_OUT_0 = """
patient_id,time,code,numeric_value
1500733,,EYE_COLOR//BROWN,
1500733,,HEIGHT,158.60131573580904
1500733,"07/20/1986, 00:00:00",DOB,
1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
1500733,"06/03/2010, 14:54:38",HR,91.4
1500733,"06/03/2010, 14:54:38",TEMP,100.0
1500733,"06/03/2010, 15:39:49",HR,84.4
1500733,"06/03/2010, 15:39:49",TEMP,100.3
1500733,"06/03/2010, 16:20:49",HR,90.1
1500733,"06/03/2010, 16:20:49",TEMP,100.1
1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""

MEDS_SHARDS = parse_meds_csvs(
    {
        "train/0": MEDS_TRAIN_0,
        "train/1": MEDS_TRAIN_1,
        "tuning/0": MEDS_TUNING_0,
        "held_out/0": MEDS_HELD_OUT_0,
    }
)


MEDS_CODE_METADATA_CSV = """
code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_code
,44,4,28,3198.8389005974336,382968.28937288234,,
ADMISSION//CARDIAC,2,2,0,,,,
ADMISSION//ORTHOPEDIC,1,1,0,,,,
ADMISSION//PULMONARY,1,1,0,,,,
DISCHARGE,4,4,0,,,,
DOB,4,4,0,,,,
EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
"""

MEDS_CODE_METADATA_SCHEMA = {
    "code": pl.Utf8,
    "code/n_occurrences": pl.UInt8,
    "code/n_patients": pl.UInt8,
    "values/n_occurrences": pl.UInt8,
    "values/sum": pl.Float32,
    "values/sum_sqd": pl.Float32,
    "description": pl.Utf8,
    "parent_code": pl.Utf8,
    "code/vocab_index": pl.UInt8,
}


def parse_code_metadata_csv(csv_str: str) -> pl.DataFrame:
    cols = csv_str.strip().split("\n")[0].split(",")
    schema = {col: dt for col, dt in MEDS_CODE_METADATA_SCHEMA.items() if col in cols}
    return pl.read_csv(StringIO(csv_str), schema=schema)


MEDS_CODE_METADATA = parse_code_metadata_csv(MEDS_CODE_METADATA_CSV)


def check_NRT_output(
    output_fp: Path,
    want_nrt: JointNestedRaggedTensorDict,
    stderr: str,
    stdout: str,
):
    assert output_fp.is_file(), f"Expected {output_fp} to exist."

    got_nrt = JointNestedRaggedTensorDict.load(output_fp)

    # assert got_nrt.schema == want_nrt.schema, (
    #    f"Expected the schema of the NRT at {output_fp} to be equal to the target.\n"
    #    f"Script stdout:\n{stdout}\n"
    #    f"Script stderr:\n{stderr}\n"
    #    f"Wanted:\n{want_nrt.schema}\n"
    #    f"Got:\n{got_nrt.schema}"
    # )

    want_tensors = want_nrt.tensors
    got_tensors = got_nrt.tensors

    assert got_tensors.keys() == want_tensors.keys(), (
        f"Expected the keys of the NRT at {output_fp} to be equal to the target.\n"
        f"Script stdout:\n{stdout}\n"
        f"Script stderr:\n{stderr}\n"
        f"Wanted:\n{list(want_tensors.keys())}\n"
        f"Got:\n{list(got_tensors.keys())}"
    )

    for k in want_tensors.keys():
        want_v = want_tensors[k]
        got_v = got_tensors[k]

        assert type(want_v) is type(got_v), (
            f"Expected tensor {k} of the NRT at {output_fp} to be of the same type as the target.\n"
            f"Script stdout:\n{stdout}\n"
            f"Script stderr:\n{stderr}\n"
            f"Wanted:\n{type(want_v)}\n"
            f"Got:\n{type(got_v)}"
        )

        if isinstance(want_v, list):
            assert len(want_v) == len(got_v), (
                f"Expected list {k} of the NRT at {output_fp} to be of the same length as the target.\n"
                f"Script stdout:\n{stdout}\n"
                f"Script stderr:\n{stderr}\n"
                f"Wanted:\n{len(want_v)}\n"
                f"Got:\n{len(got_v)}"
            )
            for i, (want_i, got_i) in enumerate(zip(want_v, got_v)):
                assert np.array_equal(want_i, got_i, equal_nan=True), (
                    f"Expected tensor {k}[{i}] of the NRT at {output_fp} to be equal to the target.\n"
                    f"Script stdout:\n{stdout}\n"
                    f"Script stderr:\n{stderr}\n"
                    f"Wanted:\n{want_i}\n"
                    f"Got:\n{got_i}"
                )
        else:
            assert np.array_equal(want_v, got_v, equal_nan=True), (
                f"Expected tensor {k} of the NRT at {output_fp} to be equal to the target.\n"
                f"Script stdout:\n{stdout}\n"
                f"Script stderr:\n{stderr}\n"
                f"Wanted:\n{want_v}\n"
                f"Got:\n{got_v}"
            )


def check_df_output(
    output_fp: Path,
    want_df: pl.DataFrame,
    stderr: str,
    stdout: str,
    check_column_order: bool = False,
    check_row_order: bool = True,
    **kwargs,
):
    assert output_fp.is_file(), f"Expected {output_fp} to exist."

    got_df = pl.read_parquet(output_fp, glob=False)
    assert_df_equal(
        want_df,
        got_df,
        (
            f"Expected the dataframe at {output_fp} to be equal to the target.\n"
            f"Script stdout:\n{stdout}\n"
            f"Script stderr:\n{stderr}"
        ),
        check_column_order=check_column_order,
        check_row_order=check_row_order,
        **kwargs,
    )


def single_stage_transform_tester(
    transform_script: str | Path,
    stage_name: str,
    transform_stage_kwargs: dict[str, str] | None,
    want_outputs: pl.DataFrame | dict[str, pl.DataFrame],
    code_metadata: pl.DataFrame | str | None = None,
    input_shards: dict[str, pl.DataFrame] | None = None,
    do_pass_stage_name: bool = False,
    file_suffix: str = ".parquet",
):
    with tempfile.TemporaryDirectory() as d:
        MEDS_dir = Path(d) / "MEDS_cohort"
        cohort_dir = Path(d) / "output_cohort"

        MEDS_data_dir = MEDS_dir / "data"
        MEDS_metadata_dir = MEDS_dir / "metadata"
        cohort_metadata_dir = cohort_dir / "metadata"

        # Create the directories
        MEDS_data_dir.mkdir(parents=True)
        MEDS_metadata_dir.mkdir(parents=True)
        cohort_dir.mkdir(parents=True)

        # Write the splits
        splits_fp = MEDS_dir / "splits.json"
        splits_fp.write_text(json.dumps(SPLITS))

        if input_shards is None:
            input_shards = MEDS_SHARDS

        # Write the shards
        for shard_name, df in input_shards.items():
            fp = MEDS_data_dir / f"{shard_name}.parquet"
            fp.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(fp, use_pyarrow=True)

        code_metadata_fp = MEDS_metadata_dir / "codes.parquet"
        if code_metadata is None:
            code_metadata = MEDS_CODE_METADATA
        elif isinstance(code_metadata, str):
            code_metadata = parse_code_metadata_csv(code_metadata)
        code_metadata.write_parquet(code_metadata_fp, use_pyarrow=True)

        pipeline_config_kwargs = {
            "input_dir": str(MEDS_dir.resolve()),
            "cohort_dir": str(cohort_dir.resolve()),
            "stages": [stage_name],
            "hydra.verbose": True,
        }

        if do_pass_stage_name:
            pipeline_config_kwargs["stage"] = stage_name
        if transform_stage_kwargs:
            pipeline_config_kwargs["stage_configs"] = {stage_name: transform_stage_kwargs}

        # Run the transform
        stderr, stdout = run_command(
            transform_script,
            pipeline_config_kwargs,
            f"Single stage transform: {stage_name}",
        )

        # Check the output
        if isinstance(want_outputs, pl.DataFrame):
            # The want output is a code_metadata file in the root directory in this case.
            check_df_output(cohort_metadata_dir / "codes.parquet", want_outputs, stderr, stdout)
        else:
            for shard_name, want in want_outputs.items():
                output_fp = cohort_dir / "data" / f"{shard_name}{file_suffix}"
                if file_suffix == ".parquet":
                    check_df_output(output_fp, want, stderr, stdout)
                elif file_suffix == ".nrt":
                    check_NRT_output(output_fp, want, stderr, stdout)
                else:
                    raise ValueError(f"Unknown file suffix: {file_suffix}")
