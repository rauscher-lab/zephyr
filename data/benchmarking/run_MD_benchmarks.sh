#!/bin/bash
# Script to benchmark the sampling rate of MD simulations

# Directory structure
# A directory named 'runs' should be located in the same directory as the given script.
# 'runs' should contain a subdirectory for each system that needs to be benchmarked.
# Finally, each subdirectory of each benchmarked system should contain a .tpr file named "{subdir}_prod1.tpr" where
# 'subdir' is the name of the subdirectory.
# The .tpr files need to contain the entire system, including solvent molecules.

# For example, the following subdirectories will benchmark the RS and AAQAA3 systems.
# ./runs/RS/RS_prod1.tpr
# ./runs/AAQAA3/AAQAA3_prod1.tpr

# # Sleep until given date
# start_date='2026-01-18 05:00:00'
# current_epoch=$(date +%s)
# target_epoch=$(date -d "$start_date" +%s)

# sleep_seconds=$(( $target_epoch - $current_epoch ))
# echo "sleeping for $sleep_seconds seconds until $start_date."
# sleep "$sleep_seconds" | exit 1

script_path=$(realpath "${0}")
root_dir=$(dirname "${script_path}")

ns_to_benchmark=1 #Input the number of ns to benchmark each sim with
ps_to_benchmark=$((ns_to_benchmark * 1000))

# Prefixes of each simulation subdirectory. Change to benchmark other systems.
declare -a sims=("RS" "nup98_24mer" "nup98_12mer" "AAQAA3" "CLN025")

truncate -s 0 performance.txt #Clear performance.txt each time script is run
echo "ns/day hour/ns" >> performance.txt

for sim in "${sims[@]}"; do

    benchmark_prefix="${sim}_prod1_${ns_to_benchmark}ns_benchmark"
    cpt_file=${benchmark_prefix}".cpt"
    log_file=${benchmark_prefix}".log"
    base_tpr_file=${sim}"_prod1.tpr"
    new_tpr_file=${benchmark_prefix}".tpr"
    output_gro_file=${benchmark_prefix}".gro"
    
    echo "Starting on system: ${sim}"

    run_subdir="${root_dir}/runs/${sim}/"
    cd "${run_subdir}" || { echo "Exiting"; exit 1; }

    #If there is no benchmark tpr, make one
    if [ ! -f "${benchmark_prefix}.tpr" ]; then
        #If no base tpr is present, exit
        if [ -f "${base_tpr_file}" ]; then
            echo "Converting tpr"
            #Set the simulation to start at t=0 and run for the specified amount of time
            gmx_mpi convert-tpr -s "${base_tpr_file}" -until "${ps_to_benchmark}" -o "${new_tpr_file}"
        else
            { echo "Missing original tpr file, exiting"; exit 1; }
            continue
        fi
    fi

    #If the final gro file hasn't been written, run the simulation
    if [ ! -f "${output_gro_file}" ]; then
        #If there are existing log and cpt files, continue from them
        if [[ -f "${log_file}" && -f "${cpt_file}" ]]; then
            echo "Resuming simulation from checkpoint"
            gmx_mpi mdrun -deffnm "${benchmark_prefix}" -cpi "${cpt_file}" -cpo "${cpt_file}" -append
        #If no existing simulation files, start from scratch
        else
            echo "Starting benchmark simulation"
            gmx_mpi mdrun -deffnm "${benchmark_prefix}"
        fi
    fi
    
    #Write performance metrics to performance.txt
    if [ -f "${log_file}" ]; then
        perf_line=$(grep "Performance:" "${log_file}" | awk '{print $2, $3}')
        echo "${sim} ${perf_line}" >> "${root_dir}/performance.txt"
    else
        echo "No log files found for ${sim}" >&2
    fi
    
    cd "${root_dir}"
    
done



    
