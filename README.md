
# USER MANUAL

How to install, train and eval the project:

```
# SSH to olympen
ssh -X -t username@ssh.edu.liu.se ssh -X olympen1-1xx # xx = 01-17

# Download datasets
./download.sh

# Enter enviroment that has correct Keras version
module add courses/TBMI26/2019-02-20.3

# Install additional dependencies
pip install --user -r requirements.txt

# Look at available options
python3 main.py --help

# Example of possible way to train segnet on camvid dataset:
python main.py --net segnet --dataset camvid --lr 0.0001 --batch 10 --epochs 30 --plot yes --crf_iter 0 --eval yes --patience 4 --decay step --decayFactor 0.25 --decayInterval 3 --weighting yes
```
