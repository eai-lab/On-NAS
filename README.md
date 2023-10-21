# [Sensys 2023 # 364] On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems
The source paper is currently under the review process.

## Introduction

This repository provides code for *On-NAS*, which is an open-sourced implementation of Submission #364 paper named *On-NAS: On-Device Neural Architecture Search on Memory-Constrained Intelligent Embedded Systems*. 
The On-NAS is consisted of two sequential parts, Two-Fold Meta-Learning and On-Device NAS. 

In the former part, Two-Fold Meta-Learning, which we denoted as TFML in the code, aims to create *meta-cell* which contains condensed and folded parameter for the given task,
to be unfolded and work as a full-sized model in the future tasks such as few-shot classification of *miniImageNet* and *Omniglot*. 

The latter part, *On-Device NAS* contains various methodologies we proposed at our work, e.g, expectation-based operation search, step-by-step backpropagation. 
On-Device NAS aims to directly operate the Neural Architecture Search at target device such as Jetson Nano, by reducing the memory consumption 20x compared to the previous works.

Scripts (**tfml.sh, tfml_onnas.sh**)  we provided reproduces the representative settings of On-NAS (ex. miniImageNet, 5-shot)  for the users.

## Notes

*codes are being updated intermittently, commit history and github activities are being cleaned for the submission.*


## Setting up the environments
```
Environment setup : conda create -f environment_onnas.yml
```

---
## Downloading the data 


Omniglot and Miniimagenet can be automatically downloaded by setting
```download = True``` for the dataloader in ```torchmeta_loader.py```

or, you can download it manually from. 

Direct link for miniImageNet : https://lyy.mpi-inf.mpg.de/mtl/download/Lmzjm9tX.html \
Direct link for Omniglot : https://github.com/brendenlake/omniglot




---


## How to use

This implementation can be separated into 2 parts, Two-fold meta-learning and On-device architecture search. 
Shell scripts below executes 5-shot setting, miniimagenet Two-Fold Meta-Learning and On-NAS. 

To begin Two-fold meta-learning and acquire meta-cell for the next step, execute the shell below.

```
./tfml.sh
```

To evaluate with additional search steps with 20x reduced memory consumption, execute the shell below.

```
./tfml_onnas.sh
```

To test the on-device search for the full-task, execute the shell below.

```
./onnas_full_task.sh
```



Arguments mentioned below are crucial arguments of On-NAS, which will determine memory consumption of On-Device NAS. \
Excessive memory consumption at Jetson Nano may lead to the shutdown of process or unintended reboot and freeze of the target device. Modify with care. 
```
argument sampleno determines number of operations to be searched.
argument split_num determines number of micro-batches we make for gradient accumulations.
argument beta_sampling turns on & off beta sampling, default is utilizing one edge pair only.
```

## Memory reduction by methodologies
by arguments above, memory consumption reduces as,

![reduction](https://github.com/sensys364/OnNAS/blob/master/images/reduction.png)


## Found architecture
---

__Diagram below shows the result of On-NAS, TFML and On-Device NAS.__

![Results](https://github.com/sensys364/OnNAS/blob/master/images/arch_int.png)

__a ) Reduction Cell, Omniglot. b ) Normal Cell, Omniglot.__ \
__c ) Reduction Cell, miniImageNet. d ) Normal Cell, miniImageNet.__




## Purpose of this project
---
This is the implementation of Sensys Submission #364, with detailed hyperparameter settings. 
This software is solely developed for the publication, Sensys 2023 #364.




