import torch
from torch import nn
from einops import rearrange


class ExternelAttention(nn.Module):
    def __init__(self, in_channel, out_channel, num_memory_units=64, num_heads=8, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads

        self.convert_dims = nn.Linear(in_channel, out_channel)
        self.extend_dims = nn.Linear(in_channel, out_channel * num_heads)

        self.memory_key = nn.Linear(out_channel, num_memory_units)
        self.memory_value = nn.Linear(num_memory_units, out_channel)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(out_channel * num_heads, out_channel)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        height = x.size(2)

        x = rearrange(x, 'b c h w -> b (h w) c')

        y = self.convert_dims(x)  # for residual connection
        x = self.extend_dims(x)

        x = rearrange(x, 'b n (h c) -> b h n c', h=self.num_heads)

        attn = self.memory_key(x)
        attn = attn.softmax(dim=-2)
        attn = attn / (1e-9 + attn.sum(dim=-1, keepdim=True))
        attn = self.attn_drop(attn)
        x = self.memory_value(attn)
        x = rearrange(x, 'b h n c -> b n (h c)')

        x = self.proj(x)
        x = self.proj_drop(x)

        x = rearrange(x, 'b (h w) c -> b c h w', h=height)
        y = rearrange(y, 'b (h w) c -> b c h w', h=height)

        return (x + y).contiguous()


if __name__ == '__main__':
    x = torch.rand(2,2,51,1)
    ea = ExternelAttention(2, 78)
    eax = ea(x)
    print(eax.size())
    print(eax.view(2, -1).size())
