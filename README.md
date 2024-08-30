# MFDm$^6$ARice

## Introduction

```text
MFDm6ARice
```

## Environment

```text
pytorch=1.13.0=py3.10_cuda11.7_cudnn8.5.0_0
```

## Data

We gratefully acknowledge Wang _et al_. [1] for providing open-source rice m$^6$A data.

## Model

```path
./model
```

## Result

Performance of 5-fold cross-validation:

```path
./results/MFDm6ARice/cv/performance.txt
```

Performance of independent test sets:

```path
./results/MFDm6ARice/indeps/*/performance.txt
```

## Usage

```shell
git clone https://github.com/zhlSunLab/MFDm6ARice
cd ./MFDm6ARice/codes

# In the file param_options.py, modify the parameters as required.

# Example for prediction
python main.py
```

## Cite

```cite
[1] Wang, Yifan, et al. "A deep learning approach to automate whole‐genome prediction of diverse epigenomic modifications in plants." New Phytologist 232.2 (2021): 880-897.
```
