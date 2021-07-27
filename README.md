# external_attention.pytorch
An unofficial PyTorch implementation of external attention. Official implementation can be found [here](https://github.com/MenghaoGuo/EANet).

<img src="" width="480" height="220" />


## Requirements
```
pip install -r requirements.txt
```

## Differences from official version
In official implementation, the number of input channels is preserved throughout the process and must be divisible by the number of attention heads. While in this implementation, the numbers of input and output channels can be specified arbitrarily, without any constraints.


## Usage
* Simply replace nn.ReLU with FReLU(num_channels), details can be found [here](https://github.com/shuuchen/frelu.pytorch/blob/master/resnet.py).
```python
from frelu import FReLU

conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
bn = nn.BatchNorm2d(out_channels)
frelu = FReLU(out_channels) # ⬅️
```


## References
* [Official implementation](https://github.com/MenghaoGuo/EANet)
* [Paper (first version)](https://arxiv.org/pdf/2105.02358v1.pdf)
* [Paper (multi-head extended version)](https://arxiv.org/pdf/2105.02358.pdf)

