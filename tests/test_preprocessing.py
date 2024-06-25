import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

import json
import subprocess
from io import StringIO
from pathlib import Path

import polars as pl

SPLITS_JSON = """{"train/0": [239684, 1195293], "train/1": [68729, 814703], "tuning/0": [754281], "held_out/0": [1500733]}"""  # noqa: E501

MEDS_TRAIN_0 = """
patient_id,code,timestamp,numerical_value
239684,HEIGHT,,175.271115221764
239684,EYE_COLOR//BROWN,,
239684,DOB,1980-12-28T00:00:00.000000,
239684,TEMP,2010-05-11T17:41:51.000000,96.0
239684,ADMISSION//CARDIAC,2010-05-11T17:41:51.000000,
239684,HR,2010-05-11T17:41:51.000000,102.6
239684,TEMP,2010-05-11T17:48:48.000000,96.2
239684,HR,2010-05-11T17:48:48.000000,105.1
239684,TEMP,2010-05-11T18:25:35.000000,95.8
239684,HR,2010-05-11T18:25:35.000000,113.4
239684,HR,2010-05-11T18:57:18.000000,112.6
239684,TEMP,2010-05-11T18:57:18.000000,95.5
239684,DISCHARGE,2010-05-11T19:27:19.000000,
1195293,HEIGHT,,164.6868838269085
1195293,EYE_COLOR//BLUE,,
1195293,DOB,1978-06-20T00:00:00.000000,
1195293,TEMP,2010-06-20T19:23:52.000000,100.0
1195293,ADMISSION//CARDIAC,2010-06-20T19:23:52.000000,
1195293,HR,2010-06-20T19:23:52.000000,109.0
1195293,TEMP,2010-06-20T19:25:32.000000,100.0
1195293,HR,2010-06-20T19:25:32.000000,114.1
1195293,HR,2010-06-20T19:45:19.000000,119.8
1195293,TEMP,2010-06-20T19:45:19.000000,99.9
1195293,HR,2010-06-20T20:12:31.000000,112.5
1195293,TEMP,2010-06-20T20:12:31.000000,99.8
1195293,HR,2010-06-20T20:24:44.000000,107.7
1195293,TEMP,2010-06-20T20:24:44.000000,100.0
1195293,TEMP,2010-06-20T20:41:33.000000,100.4
1195293,HR,2010-06-20T20:41:33.000000,107.5
1195293,DISCHARGE,2010-06-20T20:50:04.000000,
"""
MEDS_TRAIN_1 = """
patient_id,code,timestamp,numerical_value
68729,EYE_COLOR//HAZEL,,
68729,HEIGHT,,160.3953106166676
68729,DOB,1978-03-09T00:00:00.000000,
68729,HR,2010-05-26T02:30:56.000000,86.0
68729,ADMISSION//PULMONARY,2010-05-26T02:30:56.000000,
68729,TEMP,2010-05-26T02:30:56.000000,97.8
68729,DISCHARGE,2010-05-26T04:51:52.000000,
814703,EYE_COLOR//HAZEL,,
814703,HEIGHT,,156.48559093209357
814703,DOB,1976-03-28T00:00:00.000000,
814703,TEMP,2010-02-05T05:55:39.000000,100.1
814703,HR,2010-02-05T05:55:39.000000,170.2
814703,ADMISSION//ORTHOPEDIC,2010-02-05T05:55:39.000000,
814703,DISCHARGE,2010-02-05T07:02:30.000000,
"""
MEDS_HELD_OUT_0 = """
patient_id,code,timestamp,numerical_value
1500733,HEIGHT,,158.60131573580904
1500733,EYE_COLOR//BROWN,,
1500733,DOB,1986-07-20T00:00:00.000000,
1500733,TEMP,2010-06-03T14:54:38.000000,100.0
1500733,HR,2010-06-03T14:54:38.000000,91.4
1500733,ADMISSION//ORTHOPEDIC,2010-06-03T14:54:38.000000,
1500733,HR,2010-06-03T15:39:49.000000,84.4
1500733,TEMP,2010-06-03T15:39:49.000000,100.3
1500733,HR,2010-06-03T16:20:49.000000,90.1
1500733,TEMP,2010-06-03T16:20:49.000000,100.1
1500733,DISCHARGE,2010-06-03T16:44:26.000000,
"""
MEDS_TUNING_0 = """
patient_id,code,timestamp,numerical_value
754281,EYE_COLOR//BROWN,,
754281,HEIGHT,,166.22261567137025
754281,DOB,1988-12-19T00:00:00.000000,
754281,ADMISSION//PULMONARY,2010-01-03T06:27:59.000000,
754281,TEMP,2010-01-03T06:27:59.000000,99.8
754281,HR,2010-01-03T06:27:59.000000,142.0
754281,DISCHARGE,2010-01-03T08:22:13.000000,
"""

MEDS_OUTPUTS = {
    "train/0": MEDS_TRAIN_0,
    "train/1": MEDS_TRAIN_1,
    "held_out/0": MEDS_HELD_OUT_0,
    "tuning/0": MEDS_TUNING_0,
}

MEDS_OUTPUT_CODE_METADATA_FILE = """
code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd
,44,4,28,3198.8389005974336,382968.28937288234
ADMISSION//CARDIAC,2,2,0,,
ADMISSION//ORTHOPEDIC,1,1,0,,
ADMISSION//PULMONARY,1,1,0,,
DISCHARGE,4,4,0,,
DOB,4,4,0,,
EYE_COLOR//BLUE,1,1,0,,
EYE_COLOR//BROWN,1,1,0,,
EYE_COLOR//HAZEL,2,2,0,,
HEIGHT,4,4,4,656.8389005974336,108056.12937288235
HR,12,4,12,1360.5000000000002,158538.77
TEMP,12,4,12,1181.4999999999998,116373.38999999998
"""


def list_subdir_files(root: Path | str, ext: str) -> list[Path]:
    """List files in subdirectories of a directory with a given extension."""
    return sorted(list(Path(root).glob(f"**/*.{ext}")))


def run_command(script: Path, hydra_kwargs: dict[str, str], test_name: str):
    script = str(script.resolve())
    command_parts = ["python", script] + [f"{k}={v}" for k, v in hydra_kwargs.items()]
    command = " ".join(command_parts)
    command_out = subprocess.run(command, shell=True, capture_output=True)
    stderr = command_out.stderr.decode()
    stdout = command_out.stdout.decode()
    # raise ValueError(command)
    if command_out.returncode != 0:
        raise AssertionError(f"{test_name} failed!\nstdout:\n{stdout}\nstderr:\n{stderr}")
    return stderr, stdout


def test_tokenize(tmp_path):
    MEDS_cohort_dir = tmp_path / "processed" / "final_cohort"

    # Create the directories
    MEDS_cohort_dir.mkdir(parents=True, exist_ok=True)

    # Store MEDS outputs
    for split, data in MEDS_OUTPUTS.items():
        file_path = MEDS_cohort_dir / f"{split}.parquet"
        file_path.parent.mkdir(exist_ok=True)
        df = pl.read_csv(StringIO(data))
        df.with_columns(pl.col("timestamp").str.to_datetime("%Y-%m-%dT%H:%M:%S%.f")).write_parquet(file_path)

    # Check the files are not empty
    meds_files = list_subdir_files(Path(MEDS_cohort_dir), "parquet")
    assert (
        len(list_subdir_files(Path(MEDS_cohort_dir).parent, "parquet")) == 4
    ), "MEDS train split Data Files Should be 4!"
    for f in meds_files:
        assert pl.read_parquet(f).shape[0] > 0, "MEDS Data Tabular Dataframe Should not be Empty!"
    split_json = json.load(StringIO(SPLITS_JSON))
    splits_fp = MEDS_cohort_dir.parent / "splits.json"
    json.dump(split_json, splits_fp.open("w"))

    pl.read_csv(source=StringIO(MEDS_OUTPUT_CODE_METADATA_FILE)).write_parquet(
        MEDS_cohort_dir / "code_metadata.parquet"
    )

    stage = "add_time_derived_measurements"
    config_kwargs = {
        # MEDS_cohort_dir=str(MEDS_cohort_dir.resolve()),
        "input_dir": str(MEDS_cohort_dir.parent.resolve()),
        "cohort_dir": str(MEDS_cohort_dir.resolve()),
        "stage_configs.add_time_derived_measurements.age.DOB_code": "DOB",
    }
    # config_kwargs["stage_cfg.preliminary_counts"] = str(MEDS_cohort_dir.resolve())
    preprocess_root = root / "scripts" / "preprocessing"
    stderr, stdout = run_command(preprocess_root / "filter_patients.py", config_kwargs, stage)
    stderr, stdout = run_command(preprocess_root / "add_time_derived_measurements.py", config_kwargs, stage)
    stderr, stdout = run_command(
        preprocess_root / "preliminary_counts.py", config_kwargs, "preliminary_counts"
    )
    stderr, stdout = run_command(preprocess_root / "filter_codes.py", config_kwargs, "filter_codes")
    stderr, stdout = run_command(preprocess_root / "filter_outliers.py", config_kwargs, "filter_outliers")
    stderr, stdout = run_command(
        preprocess_root / "fit_vocabulary_indices.py", config_kwargs, "fit_vocabulary_indices"
    )
    stderr, stdout = run_command(preprocess_root / "normalize.py", config_kwargs, "normalize")
    stderr, stdout = run_command(preprocess_root / "tokenization.py", config_kwargs, "normalize")
    stderr, stdout = run_command(preprocess_root / "tensorize.py", config_kwargs, "normalize")
