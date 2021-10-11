# RANet-Submitted-to-IEEE-JSTARS
Relation-Attention Networks for Remote Sensing Scene Classification
(This paper has been submitted to IEEE-JSTARS in 2021.)

## Usage

1. Data preparation: `datasplit.py`

```
dataset|——train
	   |——Airport
	   |——BareLand
	   |——....
	   |——Viaduct
       |——val
	   |——Airport
	   |——BareLand
	   |——....
	   |——Viaduct
```



2. run `train.py` to train the model

## Figs

![image-20210601165926181](https://github.com/WangXin81/ACRNet/blob/main/2021-06-01_171017.png)


## Datasets:

UC Merced Land Use Dataset: 

http://weegee.vision.ucmerced.edu/datasets/landuse.html

AID Dataset: 

https://captain-whu.github.io/AID/

NWPU RESISC45: 

http://www.escience.cn/people/JunweiHan/NWPU-RESISC45.html

## Environments

1. Ubuntu 16.04
2. cuda 10.0
3. pytorch 1.0.1
4. opencv 3.4
