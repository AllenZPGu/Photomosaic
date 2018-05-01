module load anaconda3
conda create --name myenv -y
source activate myenv
conda install opencv -y
pip install progress
pip install matplotlib
pip install pandas
python generateCells.py