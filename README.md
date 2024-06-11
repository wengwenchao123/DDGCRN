# [Pattern Recognition] Decomposition Dynamic Graph Conolutional Recurrent Network for Traffic Forecasting  

This is a PyTorch implementation of Decomposition Dynamic Graph Conolutional Recurrent Network for Traffic Forecasting, as described in our paper: [Weng, Wenchao](https://github.com/wengwenchao123/DDGCRN/), Fan Jin ,Wu Huifeng and Hu Yujie ,Tian Hao, Zhu Fu, Wu Jia,  **[A Decomposition Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting](https://www.sciencedirect.com/science/article/pii/S0031320323003710)**,Pattern Recognition 2023.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pemsd3)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd3?p=a-decomposition-dynamic-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pemsd4)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd4?p=a-decomposition-dynamic-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pems07)](https://paperswithcode.com/sota/traffic-prediction-on-pems07?p=a-decomposition-dynamic-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pems08)](https://paperswithcode.com/sota/traffic-prediction-on-pems08?p=a-decomposition-dynamic-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pemsd7-m)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-m?p=a-decomposition-dynamic-graph-convolutional)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/a-decomposition-dynamic-graph-convolutional/traffic-prediction-on-pemsd7-l)](https://paperswithcode.com/sota/traffic-prediction-on-pemsd7-l?p=a-decomposition-dynamic-graph-convolutional)
## Update
 (2024/3/22)
* Added a hyperparameter `days_per_week` for easy adjustment of the index quantity of time embeddings.

* Added instructions for using the hyperparameters `days_per_week` and `steps_per_day`.
<details>
<summary>
 (2024/1/29)</summary>
  
* Optimized the data processing part of the model by changing the normalization process from using the mean and variance of the entire dataset to using the mean and variance of the training set.

* Improved the normalization operation when reading data, no longer normalizing the prediction target to avoid the abnormal MAPE issue on the PEMS03 dataset.

* Enhanced the dataset splitting operation, changing from initially dividing the dataset and then segmenting samples to first segmenting samples and then dividing the dataset. This has increased the number of training and testing samples.
</details>



## Table of Contents

* configs: training Configs and model configs for each dataset

* lib: contains self-defined modules for our work, such as data loading, data pre-process, normalization, and evaluate metrics.

* model: implementation of our model 

* pre-trained:  pre-trained model parameters

## Usage Instructions for Hyperparameters

`days_per_week`: The time intervals for data collection vary across different datasets. Adjust this hyperparameter based on the time intervals of the dataset being used. For example, in the PEMS04 dataset with a time interval of `5` minutes, set this parameter to `14400/5=288`. Similarly, in the NYC-Bike dataset with a time interval of `30` minutes, set this parameter to `14400/30=48`.

`steps_per_day`: The data collection scope varies across different datasets. For instance, PEMS04 collects data from Monday to Sunday, so set this parameter to `7`. Conversely, for the PEMS07(M) dataset, data is collected only from Monday to Friday, so set this parameter to `5`.


# Data Preparation

For convenience, we package these datasets used in our model in [Google Drive](https://drive.google.com/drive/folders/1OQYVddI5icsHwSVWtRHbqJ-xG7242q1r?usp=share_link).

Unzip the downloaded dataset files to the main file directory, the same directory as run.py.

# Requirements

Python 3.6.5, Pytorch 1.9.0, Numpy 1.16.3, argparse and configparser

# Model Training

```bash
python run.py --dataset {DATASET_NAME} --mode {MODE_NAME}
```
Replace `{DATASET_NAME}` with one of `PEMSD3`, `PEMSD4`, `PEMSD7`, `PEMSD8`, `PEMSD7(L)`, `PEMSD7(M)`

such as `python run.py --dataset PEMSD4`

There are two options for `{MODE_NAME}` : `train` and `test`

Selecting `train` will retrain the model and save the trained model parameters and records in the `experiment` folder.

With `test` selected, run.py will import the trained model parameters from `{DATASET_NAME}.pth` in the `pre-trained` folder.

## Cite

If you find the paper useful, please cite as following:

```
@article{weng2023decomposition,
  title={A Decomposition Dynamic Graph Convolutional Recurrent Network for Traffic Forecasting},
  author={Weng, Wenchao and Fan, Jin and Wu, Huifeng and Hu, Yujie and Tian, Hao and Zhu, Fu and Wu, Jia},
  journal={Pattern Recognition},
  pages={109670},
  year={2023},
  publisher={Elsevier}
}
```

## More Related Works

- [[Neural Networks] RGDAN: A random graph diffusion attention network for traffic prediction](https://doi.org/10.1016/j.neunet.2023.106093)

