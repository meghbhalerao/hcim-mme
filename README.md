# Handling Class Imbalance In Domain Adaptation
This repository contains the codes for one of my projects on Domain Adaptation at [Image Analysis and Computer Vision Lab](http://www.ee.iisc.ac.in/people/faculty/soma.biswas/index_IACV.html). 

## Brief Description 
In this repository, I have codes to run experiments on standard **Semi-Supervised Domain Adaptation** (SSDA) benchmarks such as **DomainNet, Office-31 and Office-Home** Datasets. There are also additional experiments for artificially imbalanced step and exponential **imbalanced datasets**. 

## Steps to run the code
### Minimax Entropy
1. This repository is built on top of [SSDA_MME](https://github.com/VisionLearningGroup/SSDA_MME) [1]. 
2. Download (one of) the datasets using `bash download_data.sh`. Modify this file appropriately to download the desired dataset. 
3. To run the entropy maximization setting for Domain Adaptation, run:
 ```{python}
python main.py
 --method MME # options: {ENT, S+T}
--dataset multi # options: {office, office_home}
--source real # source domain
--target sketch # target domain
 --net resnet34 # network architecture
 --attribute glove_50 # semantic initialization of prototypes
 --dim 50 # dimensions of semantics
 --loss CBFL # options: {CE, CBFL, FL}
--alpha 1 # CBFL parameter
--beta 0.999 # another cbfl parameter
--deep 1 # option to set classifier type
--patience 10 # early stopping patience
--mode train # train or inference mode
--save_check # save checkpoint
```
4. For a more detailed description of each of the parameters please see `./main.py`
### Target Reweighting
To run the second stage of training _i.e._, **target reweighting**, please run `python main_stagetwo.py --pretrained /path/to/model_ckpt`, by passing the saved checkpoint from stage 1 as the parameter. 
### Miscelleneous
To plot **t-SNE** plots of the feature representations please navigate to the `./tsne` folder.
## Dependencies
- [`pytorch`](https://pytorch.org)
- [`scikit-learn`](https://scikit-learn.org/stable/)
- [`tsne-cuda`](https://github.com/CannyLab/tsne-cuda)

## References
1. [Semi-Supervised Domain Adaptation vie Minimax Entropy](https://arxiv.org/abs/1904.06487). _Kuniaki Saito, Donghyun Kim, Stan Sclaroff, Trevor Darrell, Kate Saenko. ICCV 2019._
