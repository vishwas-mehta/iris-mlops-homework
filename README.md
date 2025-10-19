# Week 4 - PySpark Click Count Analysis

**Name:** Vishwas Mehta  
**Roll Number:** 22F3001150  
**Course:** Introduction to Big Data

## Overview
This project analyzes user click data to compute click counts by time-of-day bins using Google Cloud Dataproc and PySpark, implemented with both RDD and DataFrame APIs.

## File Descriptions

### Input Files
- **textfile.txt.txt** - Input dataset containing user_id and timestamp columns for click events

### PySpark Scripts
- **rdd_solution.py** - PySpark implementation using RDD API with map-reduce operations to count clicks in time bins (0-6, 6-12, 12-18, 18-24 hours)

- **dataframe_solution.py** - PySpark implementation using DataFrame API with high-level transformations to achieve the same click count analysis

### Output Files
- **rdd_output-part-00000** - Results from RDD implementation showing time bins and corresponding click counts in text format

- **rdd_output-part-00001** - Additional partition output from RDD processing

- **dataframe_output.csv** - Results from DataFrame implementation in CSV format with headers (time_bin, count)

## Execution
Both scripts were executed on a Google Cloud Dataproc cluster with data stored in Google Cloud Storage bucket `week4_ibd`.

## Results
Both implementations successfully computed click counts across four time intervals, demonstrating equivalence between RDD and DataFrame approaches.
