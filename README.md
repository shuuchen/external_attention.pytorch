# external_attention.pytorch
An unofficial PyTorch implementation of external attention. Official implementation can be found [here](https://github.com/MenghaoGuo/EANet).

<img src="https://github.com/shuuchen/external_attention.pytorch/blob/main/images/ea.png" width="720" height="220" />


## Requirements
```
einops==0.3.0
torch==1.7.0+cu101
torchvision==0.8.1+cu101
```

## Differences from official version
In official implementation, the number of input channels is preserved throughout the process and must be divisible by the number of attention heads. While in this implementation, the numbers of input and output channels can be specified arbitrarily, without any constraints.


## Usage
* Simply specify the numbers of input and output channels, the numbers of memory units, attention heads etc. could be specified optionally. Details can be found [here](https://github.com/shuuchen/external_attention.pytorch/blob/main/external_attention.py#L47).
```shell
$ python
Python 3.8.3 (default, Dec  9 2020, 14:17:23)
[GCC 7.5.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import torch
>>> from externel_attntion import ExternelAttention
>>>
>>> x = torch.rand(2,2,51,1)
>>> ea = ExternelAttention(2, 78)
>>> eax = ea(x)
>>> eax.size()
torch.Size([2, 78, 51, 1])
```


## References
* [Official implementation](https://github.com/MenghaoGuo/EANet)
* [Paper (first version)](https://arxiv.org/pdf/2105.02358v1.pdf)
* [Paper (multi-head extended version)](https://arxiv.org/pdf/2105.02358.pdf)

