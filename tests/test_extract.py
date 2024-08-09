"""Tests the full end-to-end extraction process.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""

import os

import rootutils

root = rootutils.setup_root(__file__, dotenv=True, pythonpath=True, cwd=True)

code_root = root / "src" / "MEDS_transforms"
extraction_root = code_root / "extract"

if os.environ.get("DO_USE_LOCAL_SCRIPTS", "0") == "1":
    SHARD_EVENTS_SCRIPT = extraction_root / "shard_events.py"
    SPLIT_AND_SHARD_SCRIPT = extraction_root / "split_and_shard_patients.py"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = extraction_root / "convert_to_sharded_events.py"
    MERGE_TO_MEDS_COHORT_SCRIPT = extraction_root / "merge_to_MEDS_cohort.py"
    AGGREGATE_CODE_METADATA_SCRIPT = code_root / "aggregate_code_metadata.py"
    EXTRACT_CODE_METADATA_SCRIPT = extraction_root / "extract_code_metadata.py"
    FINALIZE_DATA_SCRIPT = extraction_root / "finalize_MEDS_data.py"
    FINALIZE_METADATA_SCRIPT = extraction_root / "finalize_MEDS_metadata.py"
else:
    SHARD_EVENTS_SCRIPT = "MEDS_extract-shard_events"
    SPLIT_AND_SHARD_SCRIPT = "MEDS_extract-split_and_shard_patients"
    CONVERT_TO_SHARDED_EVENTS_SCRIPT = "MEDS_extract-convert_to_sharded_events"
    MERGE_TO_MEDS_COHORT_SCRIPT = "MEDS_extract-merge_to_MEDS_cohort"
    AGGREGATE_CODE_METADATA_SCRIPT = "MEDS_transform-aggregate_code_metadata"
    EXTRACT_CODE_METADATA_SCRIPT = "MEDS_extract-extract_code_metadata"
    FINALIZE_DATA_SCRIPT = "MEDS_extract-finalize_MEDS_data"
    FINALIZE_METADATA_SCRIPT = "MEDS_extract-finalize_MEDS_metadata"

import json
import tempfile
from io import StringIO
from pathlib import Path

import polars as pl
from meds import __version__ as MEDS_VERSION

from .utils import assert_df_equal, run_command

# Test data (inputs)

SUBJECTS_CSV = """
MRN,dob,eye_color,height
1195293,06/20/1978,BLUE,164.6868838269085
239684,12/28/1980,BROWN,175.271115221764
1500733,07/20/1986,BROWN,158.60131573580904
814703,03/28/1976,HAZEL,156.48559093209357
754281,12/19/1988,BROWN,166.22261567137025
68729,03/09/1978,HAZEL,160.3953106166676
"""

ADMIT_VITALS_CSV = """
patient_id,admit_date,disch_date,department,vitals_date,HR,temp
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:57:18",112.6,95.5
754281,"01/03/2010, 06:27:59","01/03/2010, 08:22:13",PULMONARY,"01/03/2010, 06:27:59",142.0,99.8
814703,"02/05/2010, 05:55:39","02/05/2010, 07:02:30",ORTHOPEDIC,"02/05/2010, 05:55:39",170.2,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 18:25:35",113.4,95.8
68729,"05/26/2010, 02:30:56","05/26/2010, 04:51:52",PULMONARY,"05/26/2010, 02:30:56",86.0,97.8
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:12:31",112.5,99.8
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 16:20:49",90.1,100.1
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:48:48",105.1,96.2
239684,"05/11/2010, 17:41:51","05/11/2010, 19:27:19",CARDIAC,"05/11/2010, 17:41:51",102.6,96.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:25:32",114.1,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 14:54:38",91.4,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:41:33",107.5,100.4
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 20:24:44",107.7,100.0
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:45:19",119.8,99.9
1195293,"06/20/2010, 19:23:52","06/20/2010, 20:50:04",CARDIAC,"06/20/2010, 19:23:52",109.0,100.0
1500733,"06/03/2010, 14:54:38","06/03/2010, 16:44:26",ORTHOPEDIC,"06/03/2010, 15:39:49",84.4,100.3
"""

INPUT_METADATA_FILE = """
lab_code,title,loinc
HR,Heart Rate,8867-4
temp,Body Temperature,8310-5
"""

DEMO_METADATA_FILE = """
eye_color,description
BROWN,"Brown Eyes. The most common eye color."
BLUE,"Blue Eyes. Less common than brown."
HAZEL,"Hazel eyes. These are uncommon"
GREEN,"Green eyes. These are rare."
"""

EVENT_CFGS_YAML = """
subjects:
  patient_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    time: null
    _metadata:
      demo_metadata:
        description: description
  height:
    code: HEIGHT
    time: null
    numeric_value: height
  dob:
    code: DOB
    time: col(dob)
    time_format: "%m/%d/%Y"
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    time: col(admit_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  discharge:
    code: DISCHARGE
    time: col(disch_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
  HR:
    code: HR
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: HR
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "HR"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "HR"}}
  temp:
    code: TEMP
    time: col(vitals_date)
    time_format: "%m/%d/%Y, %H:%M:%S"
    numeric_value: temp
    _metadata:
      input_metadata:
        description: {"title": {"lab_code": "temp"}}
        parent_codes: {"LOINC/{loinc}": {"lab_code": "temp"}}
"""

# Test data (expected outputs) -- ALL OF THIS MAY CHANGE IF THE SEED OR DATA CHANGES
EXPECTED_SPLITS = {
    "train/0": [239684, 1195293],
    "train/1": [68729, 814703],
    "tuning/0": [754281],
    "held_out/0": [1500733],
}

PATIENT_SPLITS_DF = pl.DataFrame(
    {
        "patient_id": [239684, 1195293, 68729, 814703, 754281, 1500733],
        "split": ["train", "train", "train", "train", "tuning", "held_out"],
    }
)


def get_expected_output(df: str) -> pl.DataFrame:
    return (
        pl.read_csv(source=StringIO(df))
        .select(
            "patient_id",
            pl.col("time").str.strptime(pl.Datetime, "%m/%d/%Y, %H:%M:%S").alias("time"),
            pl.col("code"),
            "numeric_value",
        )
        .sort(by=["patient_id", "time"])
    )


MEDS_OUTPUT_TRAIN_0_SUBJECTS = """
patient_id,time,code,numeric_value
239684,,EYE_COLOR//BROWN,
239684,,HEIGHT,175.271115221764
239684,"12/28/1980, 00:00:00",DOB,
1195293,,EYE_COLOR//BLUE,
1195293,,HEIGHT,164.6868838269085
1195293,"06/20/1978, 00:00:00",DOB,
"""

MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS = """
patient_id,time,code,numeric_value
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

MEDS_OUTPUT_TRAIN_1_SUBJECTS = """
patient_id,time,code,numeric_value
68729,,EYE_COLOR//HAZEL,
68729,,HEIGHT,160.3953106166676
68729,"03/09/1978, 00:00:00",DOB,
814703,,EYE_COLOR//HAZEL,
814703,,HEIGHT,156.48559093209357
814703,"03/28/1976, 00:00:00",DOB,
"""

MEDS_OUTPUT_TRAIN_1_ADMIT_VITALS = """
patient_id,time,code,numeric_value
68729,"05/26/2010, 02:30:56",ADMISSION//PULMONARY,
68729,"05/26/2010, 02:30:56",HR,86.0
68729,"05/26/2010, 02:30:56",TEMP,97.8
68729,"05/26/2010, 04:51:52",DISCHARGE,
814703,"02/05/2010, 05:55:39",ADMISSION//ORTHOPEDIC,
814703,"02/05/2010, 05:55:39",HR,170.2
814703,"02/05/2010, 05:55:39",TEMP,100.1
814703,"02/05/2010, 07:02:30",DISCHARGE,
"""

MEDS_OUTPUT_TUNING_0_SUBJECTS = """
patient_id,time,code,numeric_value
754281,,EYE_COLOR//BROWN,
754281,,HEIGHT,166.22261567137025
754281,"12/19/1988, 00:00:00",DOB,
"""

MEDS_OUTPUT_TUNING_0_ADMIT_VITALS = """
patient_id,time,code,numeric_value
754281,"01/03/2010, 06:27:59",ADMISSION//PULMONARY,
754281,"01/03/2010, 06:27:59",HR,142.0
754281,"01/03/2010, 06:27:59",TEMP,99.8
754281,"01/03/2010, 08:22:13",DISCHARGE,
"""

MEDS_OUTPUT_HELD_OUT_0_SUBJECTS = """
patient_id,time,code,numeric_value
1500733,,EYE_COLOR//BROWN,
1500733,,HEIGHT,158.60131573580904
1500733,"07/20/1986, 00:00:00",DOB,
"""

MEDS_OUTPUT_HELD_OUT_0_ADMIT_VITALS = """
patient_id,time,code,numeric_value
1500733,"06/03/2010, 14:54:38",ADMISSION//ORTHOPEDIC,
1500733,"06/03/2010, 14:54:38",HR,91.4
1500733,"06/03/2010, 14:54:38",TEMP,100.0
1500733,"06/03/2010, 15:39:49",HR,84.4
1500733,"06/03/2010, 15:39:49",TEMP,100.3
1500733,"06/03/2010, 16:20:49",HR,90.1
1500733,"06/03/2010, 16:20:49",TEMP,100.1
1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""

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

MEDS_OUTPUT_CODE_METADATA_FILE_WITH_DESC = """
code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_codes
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

MEDS_OUTPUT_DATASET_METADATA_JSON = {
    "dataset_name": "TEST",
    "dataset_version": "1.0",
    "etl_name": "MEDS_transforms",
    # "etl_version": None,  # We don't test this as it changes with the commits.
    "meds_version": MEDS_VERSION,
}

SUB_SHARDED_OUTPUTS = {
    "train/0": {
        "subjects": MEDS_OUTPUT_TRAIN_0_SUBJECTS,
        "admit_vitals": MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS,
    },
    "train/1": {
        "subjects": MEDS_OUTPUT_TRAIN_1_SUBJECTS,
        "admit_vitals": MEDS_OUTPUT_TRAIN_1_ADMIT_VITALS,
    },
    "tuning/0": {
        "subjects": MEDS_OUTPUT_TUNING_0_SUBJECTS,
        "admit_vitals": MEDS_OUTPUT_TUNING_0_ADMIT_VITALS,
    },
    "held_out/0": {
        "subjects": MEDS_OUTPUT_HELD_OUT_0_SUBJECTS,
        "admit_vitals": MEDS_OUTPUT_HELD_OUT_0_ADMIT_VITALS,
    },
}


MEDS_OUTPUTS = {
    "train/0": [MEDS_OUTPUT_TRAIN_0_SUBJECTS, MEDS_OUTPUT_TRAIN_0_ADMIT_VITALS],
    "train/1": [MEDS_OUTPUT_TRAIN_1_SUBJECTS, MEDS_OUTPUT_TRAIN_1_ADMIT_VITALS],
    "tuning/0": [MEDS_OUTPUT_TUNING_0_SUBJECTS, MEDS_OUTPUT_TUNING_0_ADMIT_VITALS],
    "held_out/0": [MEDS_OUTPUT_HELD_OUT_0_SUBJECTS, MEDS_OUTPUT_HELD_OUT_0_ADMIT_VITALS],
}


def test_extraction():
    with tempfile.TemporaryDirectory() as d:
        raw_cohort_dir = Path(d) / "raw_cohort"
        MEDS_cohort_dir = Path(d) / "MEDS_cohort"

        # Create the directories
        raw_cohort_dir.mkdir()
        MEDS_cohort_dir.mkdir()

        subjects_csv = raw_cohort_dir / "subjects.csv"
        admit_vitals_csv = raw_cohort_dir / "admit_vitals.csv"
        event_cfgs_yaml = raw_cohort_dir / "event_cfgs.yaml"

        demo_metadata_csv = raw_cohort_dir / "demo_metadata.csv"
        input_metadata_csv = raw_cohort_dir / "input_metadata.csv"

        # Write the CSV files
        subjects_csv.write_text(SUBJECTS_CSV.strip())
        admit_vitals_csv.write_text(ADMIT_VITALS_CSV.strip())
        demo_metadata_csv.write_text(DEMO_METADATA_FILE.strip())
        input_metadata_csv.write_text(INPUT_METADATA_FILE.strip())

        # Mix things up -- have one CSV be also in parquet format.
        admit_vitals_parquet = raw_cohort_dir / "admit_vitals.parquet"
        df = pl.read_csv(admit_vitals_csv)

        df.write_parquet(admit_vitals_parquet, use_pyarrow=True)

        # Write the event config YAML
        event_cfgs_yaml.write_text(EVENT_CFGS_YAML)

        # Run the extraction script
        #   1. Sub-shard the data (this will be a null operation in this case, but it is worth doing just in
        #      case.
        #   2. Collect the patient splits.
        #   3. Extract the events and sub-shard by patient.
        #   4. Merge to the final output.

        extraction_config_kwargs = {
            "input_dir": str(raw_cohort_dir.resolve()),
            "cohort_dir": str(MEDS_cohort_dir.resolve()),
            "event_conversion_config_fp": str(event_cfgs_yaml.resolve()),
            "stage_configs.split_and_shard_patients.split_fracs.train": 4 / 6,
            "stage_configs.split_and_shard_patients.split_fracs.tuning": 1 / 6,
            "stage_configs.split_and_shard_patients.split_fracs.held_out": 1 / 6,
            "stage_configs.shard_events.row_chunksize": 10,
            "stage_configs.split_and_shard_patients.n_patients_per_shard": 2,
            "hydra.verbose": True,
            "etl_metadata.dataset_name": "TEST",
            "etl_metadata.dataset_version": "1.0",
        }

        all_stderrs = []
        all_stdouts = []

        # Stage 1: Sub-shard the data
        stderr, stdout = run_command(SHARD_EVENTS_SCRIPT, extraction_config_kwargs, "shard_events")

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        subsharded_dir = MEDS_cohort_dir / "shard_events"

        try:
            out_files = list(subsharded_dir.glob("**/*.parquet"))
            assert len(out_files) == 3, f"Expected 3 output files, got {len(out_files)}."

            # Checking specific out files:
            #   1. subjects.parquet
            subjects_out = subsharded_dir / "subjects" / "[0-6).parquet"
            assert subjects_out.is_file(), f"Expected {subjects_out} to exist. Files include {out_files}."

            assert_df_equal(
                pl.read_csv(subjects_csv),
                pl.read_parquet(subjects_out, glob=False),
                "Subjects should be equal after sub-sharding",
                check_column_order=False,
                check_row_order=False,
            )
        except AssertionError as e:
            full_stderr = "\n".join(all_stderrs)
            print("Sub-sharding failed")
            print(f"stderr:\n{full_stderr}")
            raise e

        #   2. admit_vitals.parquet
        df_chunks = []
        for chunk in ["[0-10)", "[10-16)"]:
            admit_vitals_chunk_fp = subsharded_dir / "admit_vitals" / f"{chunk}.parquet"
            assert admit_vitals_chunk_fp.is_file(), f"Expected {admit_vitals_chunk_fp} to exist."

            df_chunks.append(pl.read_parquet(admit_vitals_chunk_fp, glob=False))

        assert_df_equal(
            pl.read_csv(admit_vitals_csv),
            pl.concat(df_chunks),
            "Admit vitals should be equal after sub-sharding",
            check_column_order=False,
            check_row_order=False,
        )

        # Stage 2: Collect the patient splits
        stderr, stdout = run_command(
            SPLIT_AND_SHARD_SCRIPT,
            extraction_config_kwargs,
            "split_and_shard_patients",
        )

        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        try:
            shards_fp = MEDS_cohort_dir / "metadata" / ".shards.json"
            assert shards_fp.is_file(), f"Expected splits @ {str(shards_fp.resolve())} to exist."

            splits = json.loads(shards_fp.read_text())
            expected_keys = ["train/0", "train/1", "tuning/0", "held_out/0"]

            expected_keys_str = ", ".join(f"'{k}'" for k in expected_keys)
            got_keys_str = ", ".join(f"'{k}'" for k in splits.keys())

            assert set(splits.keys()) == set(expected_keys), (
                f"Expected splits to have keys {expected_keys_str}.\n" f"Got keys: {got_keys_str}"
            )

            assert splits == EXPECTED_SPLITS, (
                f"Expected splits to be {EXPECTED_SPLITS}, got {splits}. NOTE THIS MAY CHANGE IF THE SEED OR "
                "DATA CHANGES -- FAILURE HERE MAY BE JUST DUE TO A NON-DETERMINISTIC SPLIT AND THE TEST "
                "NEEDING TO BE UPDATED."
            )
        except AssertionError as e:
            print("Failed to split patients")
            print(f"stderr:\n{stderr}")
            print(f"stdout:\n{stdout}")
            raise e

        # Stage 3: Extract the events and sub-shard by patient
        stderr, stdout = run_command(
            CONVERT_TO_SHARDED_EVENTS_SCRIPT,
            extraction_config_kwargs,
            "convert_events",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        patient_subsharded_folder = MEDS_cohort_dir / "convert_to_sharded_events"
        assert patient_subsharded_folder.is_dir(), f"Expected {patient_subsharded_folder} to be a directory."

        for split, expected_outputs in SUB_SHARDED_OUTPUTS.items():
            for prefix, expected_df_L in expected_outputs.items():
                if not isinstance(expected_df_L, list):
                    expected_df_L = [expected_df_L]

                expected_df = pl.concat([get_expected_output(df) for df in expected_df_L])

                fps = list((patient_subsharded_folder / split / prefix).glob("*.parquet"))
                assert len(fps) > 0

                # We add a "unique" here as there may be some duplicates across the row-group sub-shards.
                got_df = pl.concat([pl.read_parquet(fp, glob=False) for fp in fps]).unique()
                try:
                    assert_df_equal(
                        expected_df,
                        got_df,
                        f"Expected output for split {split}/{prefix} to be equal to the expected output.",
                        check_column_order=False,
                        check_row_order=False,
                    )
                except AssertionError as e:
                    print(f"Failed on split {split}/{prefix}")
                    print(f"stderr:\n{stderr}")
                    print(f"stdout:\n{stdout}")
                    raise e

        # Stage 4: Merge to the final output
        stderr, stdout = run_command(
            MERGE_TO_MEDS_COHORT_SCRIPT,
            extraction_config_kwargs,
            "merge_to_MEDS_cohort",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        # Check the final output
        output_folder = MEDS_cohort_dir / "merge_to_MEDS_cohort"
        try:
            for split, expected_df_L in MEDS_OUTPUTS.items():
                if not isinstance(expected_df_L, list):
                    expected_df_L = [expected_df_L]

                expected_df = pl.concat([get_expected_output(df) for df in expected_df_L])

                fp = output_folder / f"{split}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist.\nstderr:\n{stderr}\nstdout:\n{stdout}"

                got_df = pl.read_parquet(fp, glob=False)
                assert_df_equal(
                    expected_df,
                    got_df,
                    f"Expected output for split {split} to be equal to the expected output.",
                    check_column_order=False,
                    check_row_order=False,
                )

                assert got_df["patient_id"].is_sorted(), f"Patient IDs should be sorted for split {split}."
                for subj in splits[split]:
                    got_df_subj = got_df.filter(pl.col("patient_id") == subj)
                    assert got_df_subj[
                        "time"
                    ].is_sorted(), f"Times should be sorted for patient {subj} in split {split}."

        except AssertionError as e:
            print(f"Failed on split {split}")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e

        # Stage 5: Aggregate preliminary code metadata
        stderr, stdout = run_command(
            AGGREGATE_CODE_METADATA_SCRIPT,
            extraction_config_kwargs,
            "aggregate_code_metadata",
            config_name="extract",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        output_file = MEDS_cohort_dir / "aggregate_code_metadata" / "codes.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False)

        want_df = pl.read_csv(source=StringIO(MEDS_OUTPUT_CODE_METADATA_FILE)).with_columns(
            pl.col("code"),
            pl.col("code/n_occurrences").cast(pl.UInt8),
            pl.col("code/n_patients").cast(pl.UInt8),
            pl.col("values/n_occurrences").cast(pl.UInt8),
            pl.col("values/sum").cast(pl.Float32).fill_null(0),
            pl.col("values/sum_sqd").cast(pl.Float32).fill_null(0),
        )

        assert_df_equal(
            want=want_df,
            got=got_df,
            msg="Code metadata differs!",
            check_column_order=False,
            check_row_order=False,
        )

        # Stage 6: Extract code metadata
        stderr, stdout = run_command(
            EXTRACT_CODE_METADATA_SCRIPT,
            extraction_config_kwargs,
            "extract_code_metadata",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        output_file = MEDS_cohort_dir / "extract_code_metadata" / "codes.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False)

        want_df = pl.read_csv(source=StringIO(MEDS_OUTPUT_CODE_METADATA_FILE_WITH_DESC)).with_columns(
            pl.col("code"),
            pl.col("code/n_occurrences").cast(pl.UInt8),
            pl.col("code/n_patients").cast(pl.UInt8),
            pl.col("values/n_occurrences").cast(pl.UInt8),
            pl.col("values/sum").cast(pl.Float32).fill_null(0),
            pl.col("values/sum_sqd").cast(pl.Float32).fill_null(0),
            pl.col("parent_codes").cast(pl.List(pl.Utf8)),
        )

        # We collapse the list type as it throws an error in the assert_df_equal otherwise
        got_df = got_df.with_columns(pl.col("parent_codes").list.join("||"))
        want_df = want_df.with_columns(pl.col("parent_codes").list.join("||"))

        assert_df_equal(
            want=want_df,
            got=got_df,
            msg="Code metadata with descriptions differs!",
            check_column_order=False,
            check_row_order=False,
        )

        # Stage 7: Finalize the MEDS data
        stderr, stdout = run_command(
            FINALIZE_DATA_SCRIPT,
            extraction_config_kwargs,
            "finalize_MEDS_data",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        # Check the final output
        output_folder = MEDS_cohort_dir / "data"
        try:
            for split, expected_df_L in MEDS_OUTPUTS.items():
                if not isinstance(expected_df_L, list):
                    expected_df_L = [expected_df_L]

                expected_df = pl.concat([get_expected_output(df) for df in expected_df_L]).with_columns(
                    pl.col("numeric_value").cast(pl.Float32)
                )

                fp = output_folder / f"{split}.parquet"
                assert fp.is_file(), f"Expected {fp} to exist."

                got_df = pl.read_parquet(fp, glob=False)
                assert_df_equal(
                    expected_df,
                    got_df,
                    f"Expected output for split {split} to be equal to the expected output.",
                    check_column_order=False,
                    check_row_order=False,
                )

                assert got_df["patient_id"].is_sorted(), f"Patient IDs should be sorted for split {split}."
                for subj in splits[split]:
                    got_df_subj = got_df.filter(pl.col("patient_id") == subj)
                    assert got_df_subj[
                        "time"
                    ].is_sorted(), f"Times should be sorted for patient {subj} in split {split}."

        except AssertionError as e:
            print(f"Failed on split {split}")
            print(f"stderr:\n{full_stderr}")
            print(f"stdout:\n{full_stdout}")
            raise e

        # Stage 8: Finalize the metadata
        stderr, stdout = run_command(
            FINALIZE_METADATA_SCRIPT,
            extraction_config_kwargs,
            "finalize_metadata",
        )
        all_stderrs.append(stderr)
        all_stdouts.append(stdout)

        full_stderr = "\n".join(all_stderrs)
        full_stdout = "\n".join(all_stdouts)

        # Check code metadata
        output_file = MEDS_cohort_dir / "metadata" / "codes.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False, use_pyarrow=True)

        want_df = pl.read_csv(source=StringIO(MEDS_OUTPUT_CODE_METADATA_FILE_WITH_DESC)).with_columns(
            pl.col("code"),
            pl.col("code/n_occurrences").cast(pl.UInt8),
            pl.col("code/n_patients").cast(pl.UInt8),
            pl.col("values/n_occurrences").cast(pl.UInt8),
            pl.col("values/sum").cast(pl.Float32).fill_null(0),
            pl.col("values/sum_sqd").cast(pl.Float32).fill_null(0),
            pl.col("parent_codes").cast(pl.List(pl.Utf8)),
        )

        # We collapse the list type as it throws an error in the assert_df_equal otherwise
        got_df = got_df.with_columns(pl.col("parent_codes").list.join("||"))
        want_df = want_df.with_columns(pl.col("parent_codes").list.join("||"))

        assert_df_equal(
            want=want_df,
            got=got_df,
            msg=f"Finalized code metadata differs:\nstderr:\n{stderr}\nstdout:\n{stdout}",
            check_column_order=False,
            check_row_order=False,
        )

        # Check dataset metadata
        output_file = MEDS_cohort_dir / "metadata" / "dataset.json"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_json = json.loads(output_file.read_text())
        assert "etl_version" in got_json, "Expected 'etl_version' to be in the dataset metadata."
        got_json.pop("etl_version")  # We don't test this as it changes with the commits.
        assert got_json == MEDS_OUTPUT_DATASET_METADATA_JSON, f"Dataset metadata differs: {got_json}"

        # Check the splits parquet
        output_file = MEDS_cohort_dir / "metadata" / "patient_splits.parquet"
        assert output_file.is_file(), f"Expected {output_file} to exist: stderr:\n{stderr}\nstdout:\n{stdout}"

        got_df = pl.read_parquet(output_file, glob=False, use_pyarrow=True)
        assert_df_equal(
            PATIENT_SPLITS_DF,
            got_df,
            "Patient splits should be equal to the expected splits.",
            check_column_order=False,
            check_row_order=False,
        )
