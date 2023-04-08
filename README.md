---
Source code for paper "XXXXXXXXXXXXXXXXXXXXXXXXXXXX"
---

## Introduction
### 0. About the paper
This repo is the source code for the paper "XXXXXXXXXXXXXXXXXXXXXXXXXXXX" on IEEE Transcations on Multimedia (https://ieeexplore/xxxxx). If you have any questions about the source code, please contact: zhangchongyu22@gmail.com

### 1. Running Environment
```matlab
Matlab
```

### 2. Datasets
We use three datasets to perform our experiments, i.e., CIFAR-10, MIRFLICKR-25K, and NUS-WIDE datasets.

You can download all dataset from pan.baidu.com. The link and password are listed as follows:
- [Link](https://pan.baidu.com/s/1BXnhm00jKEveCcZCN4ixsg?pwd=0408). 


We also provide Google Drive download. The link is [Link](https://zcyueternal.github.io/).  

### 3. Run demo

Run mymain.m.

```matlab
mymain
```

#### Some important files:
mymain.m: main program.  
train_twostep.m: function to compute the hash code and hash function of training data.    
mAP.m : function to compute the mAP of hashing method.  
Kernelize.m: function to transform the original features to kernel features.  