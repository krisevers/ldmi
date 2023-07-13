echo "Laminar Dynamic Model Inference Pipeline"

$path = DCM_NVC_LBR
$models = DCM NVC LBR
$nsim = 400

$nthreads = 4

# run simulations
mpirun -np $nthreads python3 explore.py -p $path -m $models -n $nsim 

# extract summary statistics
python3 analyze.py -p $path # obtain summary statistics

# run inference
python3 posterior.py -p $path -m $models  # obtain posterior