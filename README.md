## Code of paper: Approximated Bilinear Modules for Temporal Modeling

![architecture](./imgs/iccv19_architecture.png?raw=true)

### Abstract

We consider two less-emphasized temporal properties of video: 1. Temporal 
cues are fine-grained; 2. Temporal modeling needs reasoning. To tackle 
both problems at once, we exploit approximated bilinear modules (ABMs) 
for temporal modeling. There are two main points making the modules 
effective: two-layer MLPs can be seen as a constraint approximation 
of bilinear operations, thus can be used to construct deep ABMs in 
existing CNNs while reusing pretrained parameters; frame features can 
be divided into static and dynamic parts because of visual repetition 
in adjacent frames, which enables temporal modeling to be more efficient. 
Multiple ABM variants and implementations are investigated, from high 
performance to high efficiency. Specifically, we show how two-layer subnets 
in CNNs can be converted to temporal bilinear modules by adding an 
auxiliary-branch. Besides, we introduce snippet sampling and shifting 
inference to boost sparse-frame video classification performance. 
Extensive ablation studies are conducted to show the effectiveness of 
proposed techniques. Our models can outperform most state-of-the-art 
methods on Something-Something v1 and v2 datasets without Kinetics 
pretraining, and are also competitive on other YouTube-like action 
recognition datasets.

### Environment:
* Python 3.6.8
* PyTorch 1.1.0
* Torchvison: 0.3.0

### Clone this repo

```
git clone https://github.com/zhuxinqimac/abm-pytorch.git
```

### Datasets
Download Something-Something v1 and v2 from 
https://20bn.com/datasets/something-something/v1
and 
https://20bn.com/datasets/something-something/v2
respectively.

We provide converted datalists in folders something_v1_list and 
something_v2_list. You need to modify the paths in datalists 
(e.g. something_v1_trainlist.txt) to fit your own machine. 
Each line in datalists contains a video folder of frames, 
n_frame, and class number.

For optical flow of Something-Something v1, you may need to use code from 
https://github.com/yangwangx/denseFlow_gpu to extract it. 

For Something-Somehting v2, follow 
https://github.com/metalbubble/TRN-pytorch 
to extract rgb frames and optical flow.


### Training
You can use scripts provided in res_train.sh to train different models.
For example, to train an implanted ABM model with backbone res34, 
snippet sampling 3 (denoted as new_length in code), 16 frames, 
and RGB modality, you can run:
```
python main.py something RGB \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --arch res_iabp_34 \
    --weight-decay 5e-4 \
    --eval-freq 1 \
    --new_length 3 \
    --num_segments 16 \
    --gd 20 --lr 0.001 --lr_steps 25 40 \
    --epochs 50 -b 24 -j 8 --dropout 0.5 \
    --short_len 16 \
    --result_path sth_v1_results/iabp/rgb_res34
```

### Evaluating
Use the script res_eval.sh to evaluate a trained model. 
A trained model for this configuration has been provided: 
https://drive.google.com/open?id=1jvdJzVV02GWAQa79KOU2itvjPWfSTs_a
```
python main.py something RGB \
    --evaluate \
    ./something_v1_list/something_v1_trainlist.txt \
    ./something_v1_list/something_v1_vallist.txt \
    --arch res_iabp_50 \
    --new_length 1 \
    --num_segments 16 \
    -b 24 -j 8 \
    --short_len 16 \
    --result_path sth_v1_results/iabp/rgb_res50 \
    --resume sth_v1_results/iabp/rgb_res50/_rgb_ep_35_checkpoint.pth.tar
```

### Citation
If you use this code or idea, please cite our paper:
```
@inproceedings{ABM_iccv19,
author={Xinqi Zhu and Chang Xu and Langwen Hui and Cewu Lu and Dacheng Tao},
title={Approximated Bilinear Modules for Temporal Modeling},
booktitle={ICCV},
year={2019}
}
```
