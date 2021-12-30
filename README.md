# ML2021_GroupProject

- 蔡定坤 BY2110227
- 郭晨旭 SY2102509
- 郭志鹏 SY2141107
- 任海英 BZ2006104
- 王健宇 SY2106106

## Client reg_info

- sid: TEAM_6
- name: 深度不学习
- token: sdbxx1234

## How to use

1. 数据集（cars_train, cars_test）两个文件夹放在datasets/stanford_cars目录下
2. 运行train.py / 或者把已经训练好的checkpoint.pth放在当前目录
3. 运行test.py
4. 在result文件夹中查看test_result和test_result_eval文件

## Link

GitHub

<https://github.com/Oct19/ML2021_GroupProject>

checkpoint.pth

<https://bhpan.buaa.edu.cn:443/link/ED0AE3E27FC95C7BD3A22CA9808E0272>

有效期限：2022-01-27 23:59

car datasets

<http://ai.stanford.edu/~jkrause/cars/car_dataset.html>

网盘坚果云

<https://www.jianguoyun.com/p/DRekJhoQ6Ln_CRjbhp8E>

<h1 align="center"><img src="https://placekitten.com/800/250"/></h1>

## Original README from API-Net

# Learning Attentive Pairwise Interaction for Fine-Grained Classification (API-Net)
Peiqin Zhuang, Yali Wang, Yu Qiao
# Introduction:
In order to effectively identify contrastive clues among highly-confused categories, we propose a simple but effective Attentive Pairwise Interaction Network (API-Net), which can progressively recognize a pair of fine-grained images by interaction. We aim at learning a mutual vector first to capture semantic differences in the input pair, and then comparing this mutual vector with individual vectors to highlight their semantic differences respectively. Besides, we also introduce a score-ranking regularization to promote the priorities of these features. For more details, please refer to [our paper](https://www.aaai.org/Papers/AAAI/2020GB/AAAI-ZhuangP.2505.pdf).
# Framework:
![Framework](/Framework.png)
# Dependencies:
* Python 2.7
* Pytorch 0.4.1
* torchvision 0.2.0
# How to use:
```
# python train.py
# python test.py
```
# Citing:
Please kindly cite the following paper, if you find this code helpful in your work.
```
@inproceedings{zhuang2020learning,
  title={Learning Attentive Pairwise Interaction for Fine-Grained Classification.},
  author={Zhuang, Peiqin and Wang, Yali and Qiao, Yu},
  booktitle={AAAI},
  pages={13130--13137},
  year={2020}
}
```
# Contact:
Please feel free to contact zpq0316@163.com or {yl.wang, yu.qiao}@siat.ac.cn, if you have any questions.
# Acknowledgement:
Some of the codes are borrowed from [siamese-triplet](https://github.com/adambielski/siamese-triplet) and [triplet-reid-pytorch](https://github.com/CoinCheung/triplet-reid-pytorch). Many thanks to them.
