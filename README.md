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

W. Tu, S. Zhou, X. Liu, Y. Liu, Z. Cai, E. Zhu, C. Zhang, and J. Cheng.<br>

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

We adopt four datasets in total, including Cora, Citeseer, Amazon Computer, and Amazon Photo. To train ITR on these datasets, please download them from [Baidu Cloud](https://pan.baidu.com/s/1mtmLy7ottCVB6fAjnlUo9A) (access code: 1234).


## Code Structure & Usage

Here we provide an implementation of Initializing Then Refining (ITR) in PyTorch, along with an execution example on Cora (or Citeseer) dataset (due to file size limit). The repository is organised as follows:

- `ITR.py`: defines the architecture of the whole network.
- `utils.py`: defines some functions about data processing, evaluation metrics, and others.
- `main.py`: the entry point for training and testing.
- `test_X.py` and `test_AX.py`: about downstream tasks.

Finally, `main.py` puts all of the above together and may be used to execute a full training run on Cora (or Citeseer).

<span id="jump2"></span>

## Architecture
<div align=center><img width="800" height="330" src="./figure/1.jpg"/></div>

## Contact
[wenxuantu@163.com](wenxuantu@163.com)

Any discussions or concerns are welcomed!

## Citation & License
If you use this code for your research, please cite our paper.
```
@inproceedings{2022ITR,
  title={Initializing Then Refining: A Simple Graph Attribute Imputation Network},
  author={Wenxuan Tu and Sihang Zhou and Xinwang Liu and Yue Liu and Zhiping Cai and En Zhu and Changwang Zhang and Jieren Cheng},
  booktitle={Proceedings of The Thirty-First International Joint Conference on Artificial Intelligence},
  pages={3494-3500},
  year={2022}
}
```

All rights reserved.
Licensed under the [Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0). 

The code is released for academic research use only. For commercial use, please contact [wenxuantu@163.com].

## Acknowledgement

X. Chen, S. Chen, J. Yao, et al. Learning on Attribute-Missing Graphs. IEEE TPAMI, 2022.<br/> 
--[https://github.com/xuChenSJTU/SAT-master-online](https://github.com/xuChenSJTU/SAT-master-online)
