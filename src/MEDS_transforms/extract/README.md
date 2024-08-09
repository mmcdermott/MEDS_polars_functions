# MEDS Extract

This directory contains the scripts and functions used to extract raw data into a MEDS dataset. If your
dataset is:

1. Arranged in a series of files on disk of an allowed format (e.g., `.csv`, `.csv.gz`, `.parquet`)...
2. Such that each file stores a dataframe containing data about patients such that each row of any given
    table corresponds to zero or more observations about a patient at a given time...
3. And you can configure how to extract those observations in the time, code, and numeric value
    format of MEDS in the event conversion `yaml` file format specified below, then...
    this tool can automatically extract your raw data into a MEDS dataset for you in an efficient, reproducible,
    and communicable way.

TODO: figure

You can see examples of this tool in action for MIMIC-IV or eICU in TODO.

## Installing MEDS Extract

MEDS Extract is installed alongside the broader MEDS Transform package. See the main package installation
instructions for details. MEDS Extract is merely a specialization and a sub-namespace of the broader MEDS
Transform tool.

## Using MEDS Extract

To use MEDS extract, you must take the following steps:
0\. **Pre-MEDS** First, you must perform any customized data massaging necessary to ensure your data obeys
the three assumptions above. This step is called "Pre-MEDS" and is not standardized across datasets. We
provide some guidance on this below.

1. **Configuration** Second, once your data obeys the assumptions above, you need to specify your extraction
    configuration options. This step is broken down into two parts:
    \- _Event Configuration_ First, and most importantly, you must specify the event conversion configuration
    file. This file specifies how to convert your raw data into MEDS events.
    \- _Pipeline Configuration_ Second, you must specify any non-event-conversion pipeline configuration
    variables, either through the command line or through a configuration file.
2. **MEDS Extract CLI** Third, you must run the MEDS Extract tool with the specified configuration options.
    This will extract your raw data into a MEDS dataset. You can run each stage of the MEDS Extract tool
    individually, if you want greater control over the parallelism or management of each stage, or you can
    run the full pipeline at once via TODO.
3. **Data Cleaning** Finally, and optionally, you can also use MEDS Transform to configure additional data
    cleaning steps that can be applied to the MEDS dataset after it has been extracted. This is not required
    for MEDS compliance, but can be useful for downstream users of the dataset. We will provide greater
    details on this below.

In the next few sections of the documentation, we will provide greater details on [steps
1](#step-1-configuring-meds-extract) and [2](#step-2-running-meds-extract) of this process, as these are the
core steps of the MEDS Extract tool. Then, we will close with more details on the [Pre-MEDS
step](#step-0-pre-meds) and the [Data Cleaning step](#step-3-data-cleanup), for interested users.

## Step 1. Configuring MEDS Extract:

### Event Conversion Configuration

The event conversion configuration file tells MEDS Extract how to convert each row of a file among your raw
data files into one or more MEDS measurements (meaning a tuple of a patient ID, a time, a categorical
code, and/or various other value or properties columns, most commonly a numeric value). This file is written
in yaml and has the following format:

```yaml
relative_table_file_stem:
  event_name:
    code: list[str | col(COLUMN_NAME)] | str | col(COLUMN_NAME)
    time: null | col(COLUMN_NAME)
    time_format: null | str | list[str]
    output_column_name_1: input_column_name_1 (str)
    ...
  ...
...
```

When you run MEDS Extract, you will specify the path to this file as a command line argument as well as a path
to the directory containing the raw data files. Supposing that input directory is `$INPUT_DIR`, then when
processing the block specified above, MEDS Extract will look for a file in `$INPUT_DIR` with the name
`$INPUT_DIR/relative_table_file_stem.$SUFFIX`, where suffix is any of the allowed suffixes, in priority order.
If no such file exists for any valid suffix, an error will be raised. Otherwise, the file will be read in and
each row of the file will be converted into a MEDS event according to the logic specified, as follows:

1. The code of the output MEDS observation will be constructed based on the
    `relative_table_file_stem.event_name.code` field. This field can be a string literal, a reference to an
    input column (denoted by the `col(...)` syntax), or a list of same. If it is a list, the output code will
    be a `"//"` separated string of each field in the list. Each field in the list (again either a string
    literal or a input column reference) is interpreted either as the specified string literal or as the
    value present in the input column. If an input column is missing in the file, an error will be raised. If
    a row has a null value for a specified input column, that field will be converted to the string `"UNK"`
    in the output code.
2. The time of the output MEDS observation will either be `null` (corresponding to static events) or
    will be read from the column specified via the input. Time columns must either be in a datetime or date
    format in the input data, or a string format that can be converted to a time via the optional `time_format`
    key, which is either a string literal format or a list of formats to try in priority order.
3. All subsequent keys and values in the event conversion block will be extracted as MEDS output column
    names by directly copying from the input data columns given. There is no need to use a `col(...)` syntax
    here, as string literals _cannot_ be used for these columns.

There are several more nuanced aspects to the configuration file that have not yet been discussed. First, the
configuration file also specifies how to identify the patient ID from the raw data. This can be done either by
specifying a global `patient_id_col` field at the top level of the configuration file, or by specifying a
`patient_id_col` field at the per-file or per-event level. Multiple specifications can be used simultaneously,
with the most local taking precedent. If no patient ID column is specified, the patient ID will be assumed to
be stored in a `patient_id` column. If the patient ID column is not found, an error will be raised.

Second, you can also specify how to link the codes constructed for each event block to code-specific metadata
in these blocks. This is done by specifying a `_metadata` block in the event block. The format of this block
is detailed in the `../parser.py` file in this directory; see there for more details. You can also see
configuration options for this block in the `tests/test_extract.py` file and in the
`MIMIC-IV_Example/configs/event_config.yaml` file.

This block tells the system to read the file `$INPUT_DIR/metadata_table_file_stem.$SUFFIX`, to collect the
columns necessary to construct the `code` field for `event_name`, potentially renaming columns according to
`_code_name_map` first, alongside any input columns specified in the block, then to construct a
`metadata/codes.parquet` dataframe which links the metadata columns to the realized codes for the dataset. See
the [Partial MIMIC-IV Example](#partial-mimic-iv-example) below for an example of this in action.

#### Examples

##### Synthetic Data Example

```yaml
subjects:
  patient_id_col: MRN
  eye_color:
    code:
      - EYE_COLOR
      - col(eye_color)
    time:
  dob:
    code: DOB
    time: col(dob)
    time_format: '%m/%d/%Y'
admit_vitals:
  admissions:
    code:
      - ADMISSION
      - col(department)
    time: col(admit_date)
    time_format: '%m/%d/%Y, %H:%M:%S'
  HR:
    code: HR
    time: col(vitals_date)
    time_format: '%m/%d/%Y, %H:%M:%S'
    numeric_value: HR
```

##### Partial MIMIC-IV Example

```yaml
patient_id_col: subject_id
hosp/admissions:
  admission:
    code:
      - HOSPITAL_ADMISSION
      - col(admission_type)
      - col(admission_location)
    time: col(admittime)
    time_format: '%Y-%m-%d %H:%M:%S'
    insurance: insurance
    language: language
    marital_status: marital_status
    race: race
    hadm_id: hadm_id
  discharge:
    code:
      - HOSPITAL_DISCHARGE
      - col(discharge_location)
    time: col(dischtime)
    time_format: '%Y-%m-%d %H:%M:%S'
    hadm_id: hadm_id

hosp/diagnoses_icd:
  diagnosis:
    code:
      - DIAGNOSIS
      - ICD
      - col(icd_version)
      - col(icd_code)
    hadm_id: hadm_id
    time: col(hadm_discharge_time)
    time_format: '%Y-%m-%d %H:%M:%S'

hosp/labevents:
  lab:
    code:
      - LAB
      - col(itemid)
      - col(valueuom)
    hadm_id: hadm_id
    time: col(charttime)
    time_format: '%Y-%m-%d %H:%M:%S'
    numeric_value: valuenum
    text_value: value
    priority: priority
```

### Pipeline Configuration

## Step 2. Running MEDS Extract:

## Step 0. Pre-MEDS:

This step is responsible for performing the minimal set of transformations required to make the raw data
source ingestible by the later steps of MEDS Extract. This step must be written by the local data owner, are
not standardized across datasets, and are not officially part of the MEDS Extract tool. You can see some
examples of these in the provided MIMIC-IV and eICU examples (todo: links)

## Step 3. Data Cleanup:

Sometimes, there are additional transformations that can be applied to a dataset that would make it
_universally_ more useful and would _not_ fundamentally modify or make assumptions about the raw data. These
transformations are not required for MEDS compliance, but can be useful for downstream users of the dataset.
This step is also optional. To implement this step, users would assemble the requisite transformations from
the MEDS Transform pre-processing tool as additional stages of their extraction pipeline, to be performed on
the data after it is already in MEDS format, and include them in their full extraction pipeline. By moving
these steps to be _after_ the core dataset is extracted into the MEDS format, users can use the same data
cleaning transformations on multiple different datasets and leverage the full MEDS Transform ecosystem to
perform these data cleaning steps effectively and efficiently. Examples of possible data cleaning steps
include:

1. Extracting numeric values from free-text values in the dataset.
2. Splitting compound measurements into their constituent parts (e.g., splitting a "blood pressure"
    measurement that is recorded in the raw data as "120/80" into separate "systolic" and "diastolic" blood
    pressure measurements).
3. Removing known technical errors in the raw data based on local data expertise.

## FAQ

### What is "Extraction"?

As defined in the [MEDS Terminology](terminology.md), "extraction" is the process of converting raw data into
a compliant MEDS representation of that raw data. This is distinct from "model-specific pre-processing"
(a.k.a. pre-processing) which can entail further modifications to a compliant MEDS dataset to
facilitate downstream model training and evaluation. While there will be many MEDS pre-processing pipelines
for different models, there should only be one MEDS extraction pipeline for a given raw data source, owned,
validated, and maintained by the local data owner.

### What is this tool (a.k.a. "MEDS Extract")?

This tool, which is a collection of transformation operations and a pre-specified pipeline configuration for
running those operations, is designed to help make it easy to produce compliant MEDS datasets in an efficient,
modular, maintainable, reproducible, and _communicable_ way. The communicability of the extraction pipeline
used to derive a MEDS dataset is _critical_ to ensure reliability and correctness of the MEDS dataset and its
downstream usage, as effective communication of what the extraction ETL entails enables effective
collaboration between local data experts and data engineers with those developing and finalizing the MEDS
extraction pipeline.

Note that this tool is _not_:

1. A specialized tool for a particular raw data source or a particular source common data model (e.g., OMOP,
    i2b2, etc.). It is a general-purpose tool that can be used to extract general raw data sources into a
    MEDS dataset. There may be more specialized tools available for dedicated CDMs or public data sources.
    See TODO for a detailed list.
2. A universal tool that can be used to extract _any_ raw data source into a MEDS dataset. It is a tool that
    can be used to extract _many_ raw data sources into a MEDS datasets, but no tool can be universally
    applicable to _all_ raw data sources. If you think your raw data source does not sufficiently conform to
    the assumptions of the MEDS Extract tool (see below), you may need to write a custom extraction tool for
    your raw data. Feel free to reach out if you have any questions or concerns about this.

## Future Improvements and Roadmap

TODO: Add issues for all of these.

1. Single event blocks for files should be specifiable directly, without an event block name.
2. Time format should be specifiable at the file or global level, like patient ID.
