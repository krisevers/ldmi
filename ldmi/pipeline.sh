# header
echo "Laminar Dynamic Model Inference - Kris Evers (2023)"
echo "Executing forward model simulation and inference pipeline"

while getopts m:gui flag
do
    case "${flag}" in
        m) method=${OPTARG};;
        g) gui=${OPTARG};;
    esac
done

echo "Method: $method";
echo "Gui: $gui";

# forward model simulation calls default parameters and 