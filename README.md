# $MFDm^6ARice$

## Introduction

```text
MFDm6ARice
```

## Environment

```text
pytorch=1.13.0=py3.10_cuda11.7_cudnn8.5.0_0
```

### Download
>Two download links are below if users want to obtain the configured environment directly.

#### Docker image
```text
Link 1: https://pan.baidu.com/s/1iy9GWE0J6bTylPpiZi2I8Q?pwd=hhsl (etraction code: hhsl)

Link 2: https://drive.google.com/drive/folders/1ZIHekVCEVe8U_HTHHeiUiVFqCFQxRYW0?usp=sharing
```

#### Packed conda environment 
```text
Link 1: https://pan.baidu.com/s/19mg6_xusXVnWfYyBdLmaNA?pwd=o9cc (etraction code: o9cc)

Link 2: https://drive.google.com/drive/folders/1QpUjFGHC3Ak-99Iaa23KBL7DaVa-7sm3?usp=sharing
```

## Data

We gratefully acknowledge Wang _et al_. [1] for providing open-source rice $m^6A$ data.

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

## Installation and Usage
### Feature encoding
```shell
# Download codes
git clone https://github.com/zhlSunLab/MFDm6ARice

cd ~/.conda/envs/  # in the ~/conda/envs/ folder or ~/anaconda3/envs/ folder. Replace with your path.
mkdir MFDm6ARice_fea
tar -xzvf MFDm6ARice_fea_encoding.tar.gz -C ./MFDm6ARice_fea/
conda info -e
conda activate MFDm6ARice_fea

cd ./MFDm6ARice/codes
python feature_encoding.py
python save_class_fea.py
```

### Docker image
>An Ubuntu 20.04 is installed in the mfdm6arice container. It allows some basic commands to be used.
>
>Note: The program uses the CPU by default. To use the GPU, users need to call the host GPU successfully in the container.

```shell
# Load docker image of mfdm6arice
docker load -i mfdm6arice.tar
docker images

# CPU
docker run -it mfdm6arice /bin/bash
./env/bin/python main.py --help
./env/bin/python main.py

# GPU
docker run --rm --gpus all -it mfdm6arice /bin/bash
./env/bin/python main.py --help
./env/bin/python main.py -d cuda:0
```

### Conda
>Note: The device needs to have Anaconda installed.

```shell
# Load packed conda environment
cd ~/.conda/envs/  # in the ~/conda/envs/ folder or ~/anaconda3/envs/ folder. Replace with your path.
mkdir MFDm6ARice
tar -xzvf MFDm6ARice.tar.gz -C ./MFDm6ARice/
conda info -e
conda activate MFDm6ARice

# Download codes
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
