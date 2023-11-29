#!/bin/bash

printf "%s\n" "------------------------"

# Default Simulation parameters
experiment_name="test"
K=5
protocol="checkerboard"
num_threads=1
mode="data"
area="V1"
sigma=1

# Default Training parameters
method="SNPE"
device="cpu"
ratio=0.8

# save start time
start=$(date +%s)

# Read the argument values
while [[ "$#" -gt 0 ]]
do
    case $1 in
        -n|--name) experiment_name="${2:-test}"; shift;;
        -k|--depths) K="${2:-5}"; shift;;
        -p|--protocol) protocol="${2:-checkerboard}"; shift;;
        -t|--threads) num_threads="${2:-1}"; shift;;
        -m|--mode) mode="${2:-data}"; shift;;
        -a|--area) area="${2:-V1}"; shift;;
        -s|--sigma) sigma="${2:-1}"; shift;;
        -d|--device) device="${2:-cpu}"; shift;;
        -r|--ratio) ratio="${2:-0.8}"; shift;;
        -h|--help) echo "Usage: ./run.sh [options]"
                   echo "Options:"
                   echo "  -n, --name              Experiment name (default: test)"
                   echo "  -k, --depths            Number of depths (default: 5)"
                   echo "  -p, --protocol          Protocol (default: checkerboard)"
                   echo "  -t, --threads           Number of threads (default: 1)"
                   echo "  -m, --mode              Mode (default: data)"
                   echo "  -a, --area              Area (default: V1)"
                   echo "  -s, --sigma             Sigma (default: 1)"
                   echo "  -d, --device            Device (default: cpu)"
                   echo "  -r, --ratio             Training/Test ratio (default: 0.8)"
                   exit 0;;
        *) echo "Unknown parameter passed: $1"; exit 1;;
    esac
    shift
done

# Show settings
echo -e "Simulation Settings:"
echo -e " \t Experiment:            \t $experiment_name"
echo -e " \t Mode:                  \t $mode"
echo -e " \t Number of Depths (K):  \t $K"
echo -e " \t Protocol:              \t $protocol"
echo -e " \t Number of Threads:     \t $num_threads"
echo -e " \t Area:                  \t $area"
printf "%s\n" "------------------------"
echo -e " "


# Create experiment directory
echo "Creating experiment directory: data/$experiment_name"
if [ -d "data/$experiment_name" ]; then
    read -p "Experiment directory already exists. Do you want to overwrite? (y/n) " overwrite
    if [ "$overwrite" == "y" ]; then
        rm -rf data/$experiment_name
        mkdir -p data/$experiment_name
    fi
fi



####### SIMULATION AND MAPPING ####### cpu accelerated
# Generate data
if [ "$mode" == "generate" ] || [ "$mode" == "data" ]; then
    echo "--------Generate--------"
    read -p "Number of simulations: " num_sims
    mpirun -n $num_threads python3 generate.py \
    --name $experiment_name \
    --num_sims $num_sims \
    --area $area
fi

# Map currents to synaptic locations in cortex
if [ "$mode" == "map" ] || [ "$mode" == "data" ]; then
    echo "--------Mapping currents to synaptic locations--------"
    python3 map.py \
    --name $experiment_name \
    --K $K \
    --sigma $sigma
fi

# Build protocol
if [ "$mode" == "protocol" ] || [ "$mode" == "data" ]; then
    echo "--------Building protocol--------"
    python3 protocol.py \
    --name $experiment_name \
    --protocol $protocol
fi

# Generate hemodynamic response and obtain beta values
if [ "$mode" == "hemodynamic" ] || [ "$mode" == "data" ]; then
    echo "--------Generating hemodynamic response--------"
    mpirun -n $num_threads python3 hemodynamics.py \
    --name $experiment_name \
    --show
fi




####### TRAINGING AND INFERENCE ######## gpu accelerated
# Train neural density estimator
if [ "$mode" == "train" ] || [ "$mode" == "infer" ]; then

    echo -e "Training Settings:"
    echo -e " \t Experiment:            \t $experiment_name"
    echo -e " \t Device:                \t $device"
    echo -e " \t Threads:               \t $num_threads"
    echo -e " \t Method:                \t $method"
    echo -e " \t Training/Test Ratio:   \t $ratio"
    printf "%s\n" "------------------------"
    echo -e " "


    echo "--------Training neural density estimator--------"
    python3 train.py \
    --name $experiment_name \
    --threads $num_threads \
    --device $device \
    --method $method \
    --ratio $ratio
fi

# Analyze results
if [ "$mode" == "analyze" ] || [ "$mode" == "infer" ]; then
    echo "--------Analyzing results--------"
    python3 analysis.py \
    --name $experiment_name
fi



# save end time
end=$(date +%s)
printf "%s\n" "------------------------"
echo -e " Procssing complete. Time elapsed: $((end-start)) seconds" 
printf "%s\n" "------------------------"
