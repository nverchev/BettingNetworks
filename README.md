# BettingNetworks
Implementation of the algorithm described in *coming soon*

## EfficientNet
Original code written as a ipynb file which can be executed in Google Colab. The required packages are installed n the top code cell.
The models mentioned in the paper accompany the notebook file.

## Pointnet2++
Implementation from https://github.com/yanx27/Pointnet_Pointnet2_pytorch
Minor modification were required on test_classification.py and train_classification.py
A new model with the Betting loss is added.
To replicate the experiments of the paper, replace/add these files to the original git implementation and follow the instruction from there

Command used in the experiment:

python train_classification.py --model pointnet2_cls_ssg --log_dir pointnet2_ssg_no_uniform
python test_classification.py --log_dir pointnet2_ssg_no_uniform
python train_classification.py --model pointnet2_betting_ssg --log_dir pointnet2_betting_ssg_no_uniform
python test_classification.py --model pointnet2_betting_ssg --log_dir pointnet2_betting_ssg_no_uniform 
