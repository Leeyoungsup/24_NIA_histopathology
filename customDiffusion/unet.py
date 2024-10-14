import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period=10000) -> torch.Tensor:
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class Upsample(nn.Module):
    """
    an upsampling layer
    """
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        output = self.layer(x)
        return output


class Downsample(nn.Module):
    """
    a downsampling layer
    """
    def __init__(self, in_ch: int, out_ch: int, use_conv: bool):
        super().__init__()
        if use_conv:
            self.layer = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        else:
            self.layer = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer(x)


class EmbedBlock(nn.Module):
    """
    abstract class
    """
    @abstractmethod
    def forward(self, x, temb, cemb):
        """
        abstract method
        """


class EmbedSequential(nn.Sequential, EmbedBlock):
    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, EmbedBlock):
                x = layer(x, temb, cemb)
            else:
                x = layer(x)
        return x


class SEBlock(nn.Module):
    def __init__(self, ch: int, reduction: int = 16):
        super().__init__()
        self.fc1 = nn.Linear(ch, ch // reduction, bias=False)
        self.fc2 = nn.Linear(ch // reduction, ch, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, _, _ = x.size()
        y = x.mean((2, 3))  # Global Average Pooling
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(B, C, 1, 1)
        return x * y


class ResBlock(EmbedBlock):
    def __init__(self, in_ch: int, out_ch: int, tdim: int, cdim: int, droprate: float):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.GroupNorm(32, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.temb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, out_ch),
        )
        self.cemb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cdim, out_ch),
        )
        self.block_2 = nn.Sequential(
            nn.GroupNorm(32, out_ch),
            nn.SiLU(),
            nn.Dropout(p=droprate),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        )
        self.se_block = SEBlock(out_ch)
        if in_ch != out_ch:
            self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
        else:
            self.residual = nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        h = self.block_1(x)
        h += self.temb_proj(temb)[:, :, None, None]
        h += self.cemb_proj(cemb)[:, :, None, None]
        h = self.block_2(h)
        h = self.se_block(h)
        return h + self.residual(x)


class Unet(nn.Module):
    def __init__(self, in_ch=3, mod_ch=64, out_ch=3, ch_mul=[1, 2, 4, 8], num_res_blocks=2, cdim=10, use_conv=True, droprate=0.1, dtype=torch.float32):
        super().__init__()
        self.in_ch = in_ch
        self.mod_ch = mod_ch
        self.out_ch = out_ch
        self.ch_mul = ch_mul
        self.num_res_blocks = num_res_blocks
        self.cdim = cdim
        self.use_conv = use_conv
        self.droprate = droprate
        self.dtype = dtype

        tdim = mod_ch * 4
        self.temb_layer = nn.Sequential(
            nn.Linear(mod_ch, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )
        self.cemb_layer = nn.Sequential(
            nn.Linear(self.cdim, tdim),
            nn.SiLU(),
            nn.Linear(tdim, tdim),
        )

        self.downblocks = nn.ModuleList([
            EmbedSequential(nn.Conv2d(in_ch, self.mod_ch, 3, padding=1))
        ])
        now_ch = self.ch_mul[0] * self.mod_ch
        chs = [now_ch]
        for i, mul in enumerate(self.ch_mul):
            nxt_ch = mul * self.mod_ch
            for _ in range(self.num_res_blocks):
                layers = [
                    ResBlock(now_ch, nxt_ch, tdim, tdim, self.droprate)
                ]
                now_ch = nxt_ch
                self.downblocks.append(EmbedSequential(*layers))
                chs.append(now_ch)
            if i != len(self.ch_mul) - 1:
                self.downblocks.append(EmbedSequential(Downsample(now_ch, now_ch, self.use_conv)))
                chs.append(now_ch)

        self.middleblocks = EmbedSequential(
            ResBlock(now_ch, now_ch, tdim, tdim, self.droprate),
            ResBlock(now_ch, now_ch, tdim, tdim, self.droprate)
        )

        self.upblocks = nn.ModuleList([])
        for i, mul in list(enumerate(self.ch_mul))[::-1]:
            nxt_ch = mul * self.mod_ch
            for j in range(num_res_blocks + 1):
                layers = [
                    ResBlock(now_ch + chs.pop(), nxt_ch, tdim, tdim, self.droprate)
                ]
                now_ch = nxt_ch
                if i and j == self.num_res_blocks:
                    layers.append(Upsample(now_ch, now_ch))
                self.upblocks.append(EmbedSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, now_ch),
            nn.SiLU(),
            nn.Conv2d(now_ch, self.out_ch, 3, stride=1, padding=1)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor, cemb: torch.Tensor) -> torch.Tensor:
        temb = self.temb_layer(timestep_embedding(t, self.mod_ch))
        cemb = self.cemb_layer(cemb)
        hs = []
        h = x.type(self.dtype)
        for block in self.downblocks:
            h = block(h, temb, cemb)
            hs.append(h)
        h = self.middleblocks(h, temb, cemb)
        for block in self.upblocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, temb, cemb)
        h = h.type(self.dtype)
        return self.out(h)