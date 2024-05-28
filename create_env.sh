echo "Sourcing conda.sh" 
source /c/Users/Admin/anaconda3/etc/profile.d/conda.sh
# Replace path conda.sh with your own path

echo "Initializing conda"
conda init bash

echo "Activating conda environment"
conda activate tnmt39

echo "Installing packages from requirements.txt"
pip install -r requirements.txt

echo "Checking installed packages"
pip check