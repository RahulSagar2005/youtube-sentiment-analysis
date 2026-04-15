# youtube-sentiment-analysis 
conda create -n youtube python=3.11-y 
conda activate youtube 
pip install -r requirements.txt 

## DVC
dvc init 
dvc repro
dvc dag 

## AWS 
aws configure

ecr uri 
703705584004.dkr.ecr.us-east-1.amazonaws.com/mlproj