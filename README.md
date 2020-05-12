![privatesign](images/private_sign.jpg)

# Large-Scale Distributed Private Learning with DPSGD

Harvard CS205 Final Project, Spring 2020

Tianhao Wang, Litao Yan, Ruoxi Yang, Silin Zou

## Introduction
Differential privacy is a framework for measuring the privacy guarantees provided by an algorithm. We can design differentially private machine learning algorithms to train on sensitive data. It provides provable guarantees of privacy, reducing the risk of exposing sensitive training data through the learned classifier. In the context of deep learning, differential private stochastic gradient descent, or the DPSGD, is the state-of-art algorithm to train such a privacy-preserving neural network. Nowadays, the DPSGD algorithm is in urgent need of combining with big compute and big data technology. On the one hand, due to the features of the DPSGD algorithm, such as limiting the gradient size in each step of parameter update, its convergence time will be 10 to 100 times longer than that of the original SGD algorithm. On the other hand, the datasets processed by DPSGD will be up to thousands of petabyte. In this project, we implemented data processing steps through training by Spark, and in the training process, we used distributed training. We employ Spark on AWS cluster to first preprocess the large amount of data. After processing the data, we use AWS GPU clusters to distribute the workload across multiple GPUs to speed up DPSGD training, where multi-GPU commucation is enabled by PyTorch Distributed Module with NCCL backend.

We demonstrate the details of this project [here](https://yanlitao.github.io/fastDP).

## Main Features

### Overview DPSGD Algorithm
As an optimization method, differential private SGD is developed from original SGD. In the process of parameter updating, there are two more important steps for DPSGD in each iteration: gradient clipping and noise addition. These two steps reduce the effect of one single anomaly so that the results of two similar datasets will not be too different. In our profiling analysis, these two steps are also the time-consuming hot-spot. 

### Dataset
Our data comes from American Community Survey Public Use Microdata Sample (PUMS) files. It includes useful but somehow sensitive census information such as Sex, Marrital Status, College degree. Our task is to privately train a deep learning model to predict the unemployment rate based on other demographic information using DPSGD, with the aid of big data and HPC tools so that we can both protect privacy as well as obtain a satisfiable runtime of the algorithm.

### Solutions
Our solution separates into two stages, data preprocessing and the DPSGD training. Accordingly, the levels of parallelism we are implemented are **big data** and **distributed parallelization technology**. To be more specific, within the data preprocessing stage, we use Spark to process large amount of data. This is because in our experiments, Spark runs much faster than MapReduce. For the model training stage, we specifically designed a new distributed version of DPSGD with gradient AllReduce step. We implemented two versions of distributed parallelism of DPSGD, one is based on PyTorch Distributed Data Parallel module and another one is implemented from scratch, where multi-GPU communication is based on PyTorch distributed package with NCCL backend. 

In terms of the infrastructure, since we didn't get approve from AWS to get more than 5 g3.4xlarge instances, we use at most 4 g3.4xlarge (or 2 g3.8xlarge, or 1 g3.16xlarge) instances to run the distributed version of DPSGD. 