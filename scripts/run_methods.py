from helpers import *

data_direc = "/Users/johnparker/Parker_et_al_2023_methods/"

baseline = [10, 10]
stim = [0, 10]
binwidth = 0.5

run_cmd(
    f"python3 analyze_data.py -d {data_direc}/examples -r {data_direc}/methods_results -b {baseline[0]} {baseline[1]} -l {stim[0]} {stim[1]} -g -ar -pd -bw {binwidth}"
)
