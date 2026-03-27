from __future__ import annotations
import math
from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
class InformationSieve(nn.Module):
    def __init(
        self,
        in_features: int,
        reduction_ratio: float = 0.5,
        noise_threshold: float = 0.1,
        xi_scale_dim: int = 1,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.reduction_ratio = reduction_ratio
        self.noise_threshold = noise_threshold
        self.reduced_features = max(1, int(in_features * reduction_ratio))
        self.information_encoder = nn.Sequential(
            nn.Linear(in_features, self.reduced_features),
            nn.LayerNorm(self.reduced_features),
            nn.GELU(),
        )
        self.information_decoder = nn.Sequential(
            nn.Linear(self.reduced_features, in_features),
        )
        self.noise_gate = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.Sigmoid(),
        )
        self.xi_adapter = nn.Sequential(
            nn.Linear(xi_scale_dim, in_features // 4),
            nn.GELU(),
            nn.Linear(in_features // 4, in_features),
            nn.Sigmoid(),
        )
        self.register_buffer('information_ratio', torch.tensor(1.0))
        self.register_buffer('noise_ratio', torch.tensor(0.0))
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        noise_mask = self.noise_gate(x)
        if xi_scale is not None:
            xi_adjustment = self.xi_adapter(xi_scale.view(-1, 1) if xi_scale.dim() == 1 else xi_scale)
            noise_mask = noise_mask * xi_adjustment
        denoised_x = x * noise_mask
        encoded = self.information_encoder(denoised_x)
        decoded = self.information_decoder(encoded)
        with torch.no_grad():
            active_neurons = (noise_mask > self.noise_threshold).float().mean()
            self.information_ratio = active_neurons
            self.noise_ratio = 1.0 - active_neurons
        info = {
            : self.information_ratio,
            : self.noise_ratio,
            : self.reduced_features,
            : self.in_features / self.reduced_features,
        }
        return decoded, info
class RGBottleneckV2(nn.Module):
    def __init(
        self,
        in_channels: int,
        out_channels: int,
        expansion_ratio: float = 0.25,
        stride: int = 1,
        xi_depth: float = 7.2,
        xi_data: float = 50.0,
        xi_target: float = 1.0,
        enable_sieve: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.xi_depth = xi_depth
        self.xi_data = xi_data
        self.xi_target = xi_target
        self.expanded_channels = int(out_channels * (1 / expansion_ratio))
        self.conv1 = nn.Conv2d(
            in_channels,
            self.expanded_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(self.expanded_channels)
        self.conv2 = nn.Conv2d(
            self.expanded_channels,
            self.expanded_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=self.expanded_channels,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(self.expanded_channels)
        self.conv3 = nn.Conv2d(
            self.expanded_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.enable_sieve = enable_sieve
        if enable_sieve:
            self.information_sieve = InformationSieve(
                in_features=out_channels,
                reduction_ratio=expansion_ratio,
            )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        self.xi_controller = nn.Parameter(torch.zeros(1))
        self._init_parameters()
    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    def compute_l_min(self, xi_data: Optional[torch.Tensor] = None) -> torch.Tensor:
        if xi_data is None:
            xi_data = torch.tensor(self.xi_data, device=self.conv1.weight.device)
        ratio = xi_data / self.xi_target
        ratio = torch.clamp(ratio, min=1.0)
        l_min = self.xi_depth * torch.log(ratio)
        return l_min
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.enable_sieve and xi_scale is not None:
            batch_size, channels, height, width = out.shape
            out_flat = out.view(batch_size, channels, -1).transpose(1, 2)
            out_sieved, sieve_info = self.information_sieve(out_flat, xi_scale)
            out = out_sieved.transpose(1, 2).view(batch_size, channels, height, width)
        else:
            sieve_info = {}
        out = out + identity
        out = F.relu(out, inplace=True)
        if xi_scale is not None:
            xi_adjustment = torch.sigmoid(self.xi_controller)
            out = out * xi_adjustment
        info = {
            : self.in_channels,
            : self.out_channels,
            : self.expanded_channels,
            : self.compute_l_min(xi_scale).item() if xi_scale is not None else self.xi_depth,
        }
        info.update(sieve_info)
        return out, info
class RGBlockStack(nn.Module):
    def __init(
        self,
        in_channels: int,
        out_channels: int,
        num_blocks: int,
        stride: int = 1,
        expansion_ratio: float = 0.25,
        xi_depth: float = 7.2,
    ) -> None:
        super().__init__()
        self.num_blocks = num_blocks
        self.blocks = nn.ModuleList()
        self.blocks.append(
            RGBottleneckV2(
                in_channels=in_channels,
                out_channels=out_channels,
                expansion_ratio=expansion_ratio,
                stride=stride,
                xi_depth=xi_depth,
            )
        )
        for _ in range(1, num_blocks):
            self.blocks.append(
                RGBottleneckV2(
                    in_channels=out_channels,
                    out_channels=out_channels,
                    expansion_ratio=expansion_ratio,
                    stride=1,
                    xi_depth=xi_depth,
                )
            )
        self.xi_depth_decay = nn.Parameter(
            torch.linspace(1.0, 0.5, num_blocks),
            requires_grad=False,
        )
    def forward(
        self,
        x: torch.Tensor,
        xi_scale: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, list]:
        block_info_list = []
        for block_idx, block in enumerate(self.blocks):
            if xi_scale is not None:
                layer_xi = xi_scale * self.xi_depth_decay[block_idx]
            else:
                layer_xi = None
            x, block_info = block(x, layer_xi)
            block_info['block_idx'] = block_idx
            block_info_list.append(block_info)
        return x, block_info_list
class RGPBottleneckNetwork(nn.Module):
    def __init(
        self,
        in_channels: int = 3,
        num_classes: int = 1000,
        block_config: Tuple[int, ...] = (3, 4, 6, 3),
        channel_config: Tuple[int, ...] = (64, 128, 256, 512),
        expansion_ratio: float = 0.25,
        xi_depth: float = 7.2,
        xi_data: float = 50.0,
        xi_target: float = 1.0,
    ) -> None:
        super().__init__()
        self.xi_depth = xi_depth
        self.xi_data = xi_data
        self.xi_target = xi_target
        self.conv1 = nn.Conv2d(
            in_channels,
            channel_config[0],
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(channel_config[0])
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.stages = nn.ModuleList()
        in_ch = channel_config[0]
        for stage_idx, (num_blocks, out_ch) in enumerate(zip(block_config, channel_config)):
            stride = 1 if stage_idx == 0 else 2
            stage = RGBlockStack(
                in_channels=in_ch,
                out_channels=out_ch,
                num_blocks=num_blocks,
                stride=stride,
                expansion_ratio=expansion_ratio,
                xi_depth=xi_depth,
            )
            self.stages.append(stage)
            in_ch = out_ch
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(channel_config[-1], num_classes)
        self.xi_global = nn.Parameter(torch.tensor(xi_data))
        self._init_parameters()
    def _init_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
    def compute_complexity(self, x: torch.Tensor) -> torch.Tensor:
        feature_variance = x.var(dim=[2, 3]).mean(dim=1)
        xi_estimate = torch.log1p(feature_variance * 10)
        return xi_estimate
    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor or Tuple[torch.Tensor, list]:
        xi_input = self.compute_complexity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x, inplace=True)
        x = self.maxpool(x)
        all_block_info = []
        for stage_idx, stage in enumerate(self.stages):
            stage_xi = xi_input * (0.8 ** stage_idx)
            x, block_info = stage(x, stage_xi)
            all_block_info.extend(block_info)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.fc(x)
        if return_features:
            return logits, all_block_info
        return logits
    def get_model_complexity(self) -> Dict[str, float]:
        total_params = sum(p.numel() for p in self.parameters())
        total_flops = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                flops = (
                    m.kernel_size[0] * m.kernel_size[1] *
                    m.in_channels * m.out_channels *
                    m.stride[0] * m.stride[1]
                )
                total_flops += flops
        return {
            : total_params / 1e6,
            : total_flops / 1e9,
            : self.xi_depth,
            : self.xi_data.item() if isinstance(self.xi_data, torch.Tensor) else self.xi_data,
        }