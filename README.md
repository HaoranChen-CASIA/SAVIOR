# SAVIOR
Official implementation of "SAVIOR: Assessing Volume Alignment Quality for Serial Section Electron Microscopy Images using Large Vision-Language Model"

## Using the Code
### Requirements
This code has been developed under Python 3.9.17, PyTorch 2.0.1, and Ubuntu 16.04.

The python environment can be set as follows:
To set the environment for LVLM-based branch, please use:
```shell
git clone https://github.com/Q-Future/Q-Align.git
cd Q-Align
pip install -e .
```
To prepare SAM for MAM-guided motion analysis branch, please use:
```shell
pip install git+https://github.com/facebookresearch/segment-anything.git
```
The remaining environments can be set by:
```shell
conda create -n SAVIOR
conda activate SAVIOR
pip install -r requirements.txt
```

### Validation and Test Datasets
To re-implement the experiments in the paper, it is recommended to download the dataset used in this paper.
Test: Ranked volumes generated from Lucchi++, with combined distortion:
[GoogleDrive](https://drive.google.com/file/d/1EoeLCeYjoac_ASdTV2Wloi0sTY6l8PJM/view?usp=sharing)
All datasets required for training, testing and comparison experiments will be made available after the paper's official publication.

### Pretrained Weights
Pretrained Weights of SAVIOR can be downloaded from:

[CSTCloud]()

### Quantitative Experiments
To re-implement the quantitative experiments:
```Register
python train_Gating_and_eval.py
```

### Qualitative Experiments
To re-implement the qualitative experiments:
```Register
python infer_large_scale_MAM_Gating.py
```

The training dataset and codes will be released soon.






