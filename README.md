![privatesign](images/private_sign.jpg)

# Large-Scale Distributed Private Learning with DPSGD

Harvard CS205 Final Project, Spring 2020

Tianhao Wang, Litao Yan, Ruoxi Yang, Silin Zou

## Introduction
Differential privacy is a framework for measuring the privacy guarantees provided by an algorithm. We can design differentially private machine learning algorithms to train on sensitive data. It provides provable guarantees of privacy, reducing the risk of exposing sensitive training data through the learned classifier. In the context of deep learning, differential private stochastic gradient descent, or the DPSGD, is the state-of-art algorithm to train such a privacy-preserving neural network. Nowadays, the DPSGD algorithm is in urgent need of combining with big compute and big data technology. On the one hand, due to the features of the DPSGD algorithm, such as limiting the gradient size in each step of parameter update, its convergence time will be 10 to 100 times longer than that of the original SGD algorithm. On the other hand, the datasets processed by DPSGD will be up to thousands of petabyte. In this project, we implemented data processing steps through training by Spark, and in the training process, we used distributed training. 

We employ Spark on AWS cluster to first preprocess the large amount of data. After processing the data, we use AWS GPU clusters to distribute the workload across multiple GPUs to speed up DPSGD training, where multi-GPU commucation is enabled by PyTorch Distributed Module with NCCL backend.

We demonstrate the details of this project in [here](https://yanlitao.github.io/fastDP).

## Main Features
### Data Processing: Spark
### Distributed Parallelization

## 