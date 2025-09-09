# Graph Neural Network for HPC Job Scheduling
A Graph Attention Network (GAT) implementation for predicting High Performance Computing (HPC) job runtimes using historical scheduling data from Argonne National Laboratory's computing facilities.
## Overview
This project implements a Graph Neural Network approach to model and predict HPC job scheduling patterns across three major computing systems: Polaris, Mira, and Cooley. The model uses Graph Attention Networks to capture complex relationships between jobs based on resource requirements and execution characteristics.
## Features
•	Processes data from Polaris, Mira, and Cooley HPC systems
•	Creates job dependency graphs based on resource similarity
•	Uses GAT layers to focus on relevant job relationships
•	Includes batch normalization, dropout, and proper train/validation splits
•	Generates loss curves for model comparison across systems
### Data Format
#### Expected data files:
•	ANL-ALCF-DJC-POLARIS_20240101_20241031.csv.gz
•	ANL-ALCF-DJC-COOLEY_20190101_20191231.csv.gz
•	ANL-ALCF-DJC-MIRA_20190101_20191231.csv.gz
#### Usage
#### Basic Execution
Simply run the main script with your data files in the same directory:
python gnn_rl_implementation.py
