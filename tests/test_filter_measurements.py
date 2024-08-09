"""Tests the filter measurements script.

Set the bash env variable `DO_USE_LOCAL_SCRIPTS=1` to use the local py files, rather than the installed
scripts.
"""


from .transform_tester_base import (
    FILTER_MEASUREMENTS_SCRIPT,
    single_stage_transform_tester,
)
from .utils import parse_meds_csvs

# This is the code metadata
# MEDS_CODE_METADATA_CSV = """
# code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_code
# ,44,4,28,3198.8389005974336,382968.28937288234,,
# ADMISSION//CARDIAC,2,2,0,,,,
# ADMISSION//ORTHOPEDIC,1,1,0,,,,
# ADMISSION//PULMONARY,1,1,0,,,,
# DISCHARGE,4,4,0,,,,
# DOB,4,4,0,,,,
# EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
# EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
# EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
# HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
# HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
# TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
# """
#
# We'll keep only the codes that occur for at least 2 patients, which are: ADMISSION//CARDIAC, DISCHARGE, DOB,
# EYE_COLOR//HAZEL, HEIGHT, HR, TEMP

WANT_TRAIN_0 = """
patient_id,time,code,numeric_value
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

WANT_TRAIN_1 = """
patient_id,time,code,numeric_value
68729,,EYE_COLOR//HAZEL,
68729,,HEIGHT,160.3953106166676
68729,"03/09/1978, 00:00:00",DOB,
68729,"05/26/2010, 02:30:56",HR,86.0
68729,"05/26/2010, 02:30:56",TEMP,97.8
68729,"05/26/2010, 04:51:52",DISCHARGE,
814703,,EYE_COLOR//HAZEL,
814703,,HEIGHT,156.48559093209357
814703,"03/28/1976, 00:00:00",DOB,
814703,"02/05/2010, 05:55:39",HR,170.2
814703,"02/05/2010, 05:55:39",TEMP,100.1
814703,"02/05/2010, 07:02:30",DISCHARGE,
"""

WANT_TUNING_0 = """
patient_id,time,code,numeric_value
754281,,HEIGHT,166.22261567137025
754281,"12/19/1988, 00:00:00",DOB,
754281,"01/03/2010, 06:27:59",HR,142.0
754281,"01/03/2010, 06:27:59",TEMP,99.8
754281,"01/03/2010, 08:22:13",DISCHARGE,
"""

WANT_HELD_OUT_0 = """
patient_id,time,code,numeric_value
1500733,,HEIGHT,158.60131573580904
1500733,"07/20/1986, 00:00:00",DOB,
1500733,"06/03/2010, 14:54:38",HR,91.4
1500733,"06/03/2010, 14:54:38",TEMP,100.0
1500733,"06/03/2010, 15:39:49",HR,84.4
1500733,"06/03/2010, 15:39:49",TEMP,100.3
1500733,"06/03/2010, 16:20:49",HR,90.1
1500733,"06/03/2010, 16:20:49",TEMP,100.1
1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""

WANT_SHARDS = parse_meds_csvs(
    {
        "train/0": WANT_TRAIN_0,
        "train/1": WANT_TRAIN_1,
        "tuning/0": WANT_TUNING_0,
        "held_out/0": WANT_HELD_OUT_0,
    }
)


def test_filter_measurements():
    single_stage_transform_tester(
        transform_script=FILTER_MEASUREMENTS_SCRIPT,
        stage_name="filter_measurements",
        transform_stage_kwargs={"min_patients_per_code": 2},
        want_outputs=WANT_SHARDS,
    )


# This is the code metadata
# MEDS_CODE_METADATA_CSV = """
# code,code/n_occurrences,code/n_patients,values/n_occurrences,values/sum,values/sum_sqd,description,parent_code
# ,44,4,28,3198.8389005974336,382968.28937288234,,
# ADMISSION//CARDIAC,2,2,0,,,,
# ADMISSION//ORTHOPEDIC,1,1,0,,,,
# ADMISSION//PULMONARY,1,1,0,,,,
# DISCHARGE,4,4,0,,,,
# DOB,4,4,0,,,,
# EYE_COLOR//BLUE,1,1,0,,,"Blue Eyes. Less common than brown.",
# EYE_COLOR//BROWN,1,1,0,,,"Brown Eyes. The most common eye color.",
# EYE_COLOR//HAZEL,2,2,0,,,"Hazel eyes. These are uncommon",
# HEIGHT,4,4,4,656.8389005974336,108056.12937288235,,
# HR,12,4,12,1360.5000000000002,158538.77,"Heart Rate",LOINC/8867-4
# TEMP,12,4,12,1181.4999999999998,116373.38999999998,"Body Temperature",LOINC/8310-5
# """
#
# In the test that applies to the match and revise framework, we'll filter codes in the following manner:
#   - Codes that start with ADMISSION// will be filtered to occur at least 2 times, which are:
#     ADMISSION//CARDIAC
#   - Codes in [HR] will be filtered to occur at least 15 times, which are:
#     (no codes)
#   - Codes that start with EYE_COLOR// will be filtered to occur at least 4 times, which are:
#     (no codes)
#   - Other codes won't be filtered, so we will retain HEIGHT, DISCHARGE, DOB, TEMP

MR_WANT_TRAIN_0 = """
patient_id,time,code,numeric_value
239684,,HEIGHT,175.271115221764
239684,"12/28/1980, 00:00:00",DOB,
239684,"05/11/2010, 17:41:51",ADMISSION//CARDIAC,
239684,"05/11/2010, 17:41:51",TEMP,96.0
239684,"05/11/2010, 17:48:48",TEMP,96.2
239684,"05/11/2010, 18:25:35",TEMP,95.8
239684,"05/11/2010, 18:57:18",TEMP,95.5
239684,"05/11/2010, 19:27:19",DISCHARGE,
1195293,,HEIGHT,164.6868838269085
1195293,"06/20/1978, 00:00:00",DOB,
1195293,"06/20/2010, 19:23:52",ADMISSION//CARDIAC,
1195293,"06/20/2010, 19:23:52",TEMP,100.0
1195293,"06/20/2010, 19:25:32",TEMP,100.0
1195293,"06/20/2010, 19:45:19",TEMP,99.9
1195293,"06/20/2010, 20:12:31",TEMP,99.8
1195293,"06/20/2010, 20:24:44",TEMP,100.0
1195293,"06/20/2010, 20:41:33",TEMP,100.4
1195293,"06/20/2010, 20:50:04",DISCHARGE,
"""

MR_WANT_TRAIN_1 = """
patient_id,time,code,numeric_value
68729,,HEIGHT,160.3953106166676
68729,"03/09/1978, 00:00:00",DOB,
68729,"05/26/2010, 02:30:56",TEMP,97.8
68729,"05/26/2010, 04:51:52",DISCHARGE,
814703,,HEIGHT,156.48559093209357
814703,"03/28/1976, 00:00:00",DOB,
814703,"02/05/2010, 05:55:39",TEMP,100.1
814703,"02/05/2010, 07:02:30",DISCHARGE,
"""

MR_WANT_TUNING_0 = """
patient_id,time,code,numeric_value
754281,,HEIGHT,166.22261567137025
754281,"12/19/1988, 00:00:00",DOB,
754281,"01/03/2010, 06:27:59",TEMP,99.8
754281,"01/03/2010, 08:22:13",DISCHARGE,
"""

MR_WANT_HELD_OUT_0 = """
patient_id,time,code,numeric_value
1500733,,HEIGHT,158.60131573580904
1500733,"07/20/1986, 00:00:00",DOB,
1500733,"06/03/2010, 14:54:38",TEMP,100.0
1500733,"06/03/2010, 15:39:49",TEMP,100.3
1500733,"06/03/2010, 16:20:49",TEMP,100.1
1500733,"06/03/2010, 16:44:26",DISCHARGE,
"""

MR_WANT_SHARDS = parse_meds_csvs(
    {
        "train/0": MR_WANT_TRAIN_0,
        "train/1": MR_WANT_TRAIN_1,
        "tuning/0": MR_WANT_TUNING_0,
        "held_out/0": MR_WANT_HELD_OUT_0,
    }
)

MATCH_REVISE_KEY = "_match_revise"
MATCHER_KEY = "_matcher"


def test_match_revise_filter_measurements():
    single_stage_transform_tester(
        transform_script=FILTER_MEASUREMENTS_SCRIPT,
        stage_name="filter_measurements",
        transform_stage_kwargs={
            "_match_revise": [
                {"_matcher": {"code": "ADMISSION//CARDIAC"}, "min_patients_per_code": 2},
                {"_matcher": {"code": "ADMISSION//ORTHOPEDIC"}, "min_patients_per_code": 2},
                {"_matcher": {"code": "ADMISSION//PULMONARY"}, "min_patients_per_code": 2},
                {"_matcher": {"code": "HR"}, "min_patients_per_code": 15},
                {"_matcher": {"code": "EYE_COLOR//BLUE"}, "min_patients_per_code": 4},
                {"_matcher": {"code": "EYE_COLOR//BROWN"}, "min_patients_per_code": 4},
                {"_matcher": {"code": "EYE_COLOR//HAZEL"}, "min_patients_per_code": 4},
            ],
        },
        want_outputs=MR_WANT_SHARDS,
        do_use_config_yaml=True,
    )
