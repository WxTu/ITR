[python-img]: https://img.shields.io/github/languages/top/WxTu/ITR?color=lightgrey
[stars-img]: https://img.shields.io/github/stars/WxTu/ITR?color=yellow
[stars-url]: https://github.com/WxTu/ITR/stargazers
[fork-img]: https://img.shields.io/github/forks/WxTu/ITR?color=lightblue&label=fork
[fork-url]: https://github.com/WxTu/ITR/network/members
[visitors-img]: https://visitor-badge.glitch.me/badge?page_id=WxTu/ITR
[adgc-url]: https://github.com/WxTu/ITR


## Paper
[![Made with Python][python-img]][adgc-url]
[![GitHub stars][stars-img]][stars-url]
[![GitHub forks][fork-img]][fork-url]
[![visitors][visitors-img]][adgc-url]

Source code for the paper "Initializing Then Refining: A Simple Graph Attribute Imputation Network"<br>

W. Tu, S. Zhou, X. Liu, Y. Liu, Z. Cai, E. zhu, C. Zhang, and J. Cheng.<br>

Accepted by IJCAI 2022. [[Paper]](https://github.com/WxTu/ITR/blob/master/ITR-final.pdf) <br>



## Installation

Clone this repo.
```bash
git clone https://github.com/WxTu/ITR.git
```

* Python 3.7.11
* [Pytorch (1.9.0)](https://pytorch.org/)
* Numpy 1.25.1
* Sklearn 1.0.2
* Torchvision 0.10.0
* Matplotlib 3.5.1


## Preparation

We adopt four datasets in total, including Cora, Citeseer, Amazon Computer, and Amazon Photo. To train a model on these datasets, please download them from [Baidu Cloud](https://pan.baidu.com/s/1ykIPGLXLMtMqtgpXOq3_sQ) (access code: 4622).

## Code Structure & Usage

Here we provide an implementation of Initializing Then Refining (ITR) in PyTorch, along with an execution example on the Cora (or Citeseer) dataset (due to file size limit). The repository is organised as follows:

- `ITR.py`: defines the architecture of the whole network.
- `utils.py`: defines some functions about data processing, evaluation metrics, and others.
- `main.py`: the entry point for training and testing.
- `test_X.py` and `test_AX.py`: about downstream tasks.

Finally, `main.py` puts all of the above together and may be used to execute a full training run on Cora.

<span id="jump2"></span>

## Visualization
<div align=center><img width="800" height="330" src="./figure/1.jpg"/></div>

## Contact
[wenxuantu@163.com](wenxuantu@163.com)

Any discussions or concerns are welcomed!

## Citation & License
If you use this code for your research, please cite our paper.
```
@inproceedings{TuDeep,
  title={Deep Fusion Clustering Network},
  author={Tu, Wenxuan and Zhou, Sihang and Liu, Xinwang and Guo, Xifeng and Cai, Zhiping and zhu, En and Cheng, Jieren},
  booktitle={The Thirty-Fifth AAAI Conference on Artificial Intelligence},
  year={2021}
}
```

All rights reserved.
Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). 

The code is released for academic research use only. For commercial use, please contact [wenxuantu@163.com].

## Acknowledgement

D. Bo, X. Wang, C. Shi, et al. Structural Deep Clustering Network. In WWW, 2020.<br/> 
--[https://github.com/bdy9527/SDCN](https://github.com/bdy9527/SDCN)

X. Guo, L. Gao, X. Liu, et al. Improved Deep Embedded Clustering with Local Structure Preservation. In IJCAI, 2017.<br/>
--[https://github.com/XifengGuo/IDEC](https://github.com/XifengGuo/IDEC)

J. Xie, R. Girshick, and A. Farhadi. Unsupervised Deep Embedding for Clustering Analysis. In ICML, 2016.<br/>
--[https://github.com/vlukiyanov/pt-dec](https://github.com/vlukiyanov/pt-dec)
