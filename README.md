# [ACM MM 2023] Self-Distillation Dual-Memory Online Hashing with Hash Centers for Streaming Data Retrieval


## Introduction
### 0. About the paper
This repo is the source code for the paper [**Self-Distillation Dual-Memory Online Hashing with Hash Centers for Streaming Data Retrieval**](https://github.com/ZCyueternal/SDOH-HC) on ACM International Conference on Multimedia (**ACM MM**), 2023 
<br>by [Chong-Yu Zhang](https://scholar.google.com.hk/citations?hl=en&user=OsbUzCMAAAAJ), [Xin Luo](https://scholar.google.com.hk/citations?hl=en&user=ZaCsoy0AAAAJ), [Yu-Wei Zhan](https://scholar.google.com.hk/citations?user=iM_-4-sAAAAJ&hl=en&oi=sra), [Peng-Fei Zhang](https://scholar.google.com.hk/citations?hl=en&user=KTnEPf8AAAAJ), [Zhen-Duo Chen](https://scholar.google.com.hk/citations?user=v28-0D0AAAAJ&hl=en&oi=sra), [Yongxin Wang](https://scholar.google.com.hk/citations?user=0SnREAQAAAAJ&hl=en&oi=sra), [Xun Yang](https://scholar.google.com.hk/citations?user=ro8lzsUAAAAJ&hl=en&oi=ao), and [Xin-Shun Xu](https://scholar.google.com.hk/citations?user=ICzwFaIAAAAJ&hl=en&oi=ao). 

### 1. Running Environment
```matlab
Matlab
```
## :rocket: News
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


## Citation
Please cite the following paper if you find this useful in your research:
```
@InProceedings{zhang2023self,
author = {Zhang, Chong-Yu and Luo, Xin and Zhan, Yu-Wei and Zhang, Peng-Fei and Chen, Zhen-Duo and Wang Yongxin and Yang, Xun and Xu, Xin-Shun},
title = {Self-distillation dual-memory online hashing with hash centers for streaming data retrieval},
booktitle = {Proceedings of the ACM International Conference on Multimedia},
year = {2023}
}
```

## Contact
If you have any questions, feel free to contact [us](zhangchongyu22@gmail.com).
