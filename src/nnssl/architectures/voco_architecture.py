import torch
from nnssl.experiment_planning.experiment_planners.plan import ConfigurationPlan
from torch import nn

class VocoProjectionHead(nn.Module):
    def __init__(self, total_channels: int, hidden_dim: int, output_dim: int, norm_op: nn.Module = nn.InstanceNorm1d):
        super(VocoProjectionHead, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(total_channels, hidden_dim),
            norm_op(hidden_dim, affine=False, track_running_stats=False),
            nn.ReLU(inplace=False),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
         #   norm_op(hidden_dim, affine=False, track_running_stats=False),
         #   nn.ReLU(inplace=True),
        )
       # self.layer3 = nn.Sequential(
       #     nn.Linear(hidden_dim, output_dim),
       # )

    def forward(self, x: list[torch.Tensor]) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
       # x = self.layer3(x)
        return x


class VoCoArchitecture(nn.Module):
    def __init__(self, encoder: nn.Module,  features: list[int], vit: bool = False):
        super(VoCoArchitecture, self).__init__()
        self.encoder = encoder
        self.vit = vit
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))


        total_features = sum(features)
        self.projector = VocoProjectionHead(total_features, 256, 256, norm_op=nn.InstanceNorm1d)

    def forward(self, x):
       # print(x.shape, 'x')
        out = self.encoder(x)
        if isinstance(out, torch.Tensor):
            out = out.contiguous().clone()
      #  print(out.shape, 'out')

        
        if self.vit: 
            feats = out if isinstance(out, (list, tuple)) else [out]
           # print(len(feats), len(feats[0]), len(feats[0][0]), len(feats[0][0][0]), 'feats')
            #flat = torch.cat([o.mean(dim=1) for o in seqs], dim=1) 
            pooled = [o.mean(dim=1) for o in feats]   # use o[:, 0] if you prefer CLS
           # print(len(pooled), len(pooled[0]), len(pooled[0][0]), 'pooled')
            flat_out = pooled[0] if len(pooled) == 1 else torch.cat(pooled, dim=1)
           # print(flat_out.shape, 'flat_out')
           #print(out.shape)
           # print(len(feats))#.shape)
           # print(len(feats[0]))
           # print(len(feats[0][0]))
           # print(len(pooled))
           # print(len(pooled[0]))
           # print(len(pooled[0][0]))
           # print(flat_out.shape)
        else:
            flat_out = torch.concat([self.adaptive_pool(o) for o in out], dim=1)
            flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))
        x = self.projector(flat_out)
        return x

class VoCoEvaArchitecture(nn.Module):
    """
    We don't have multiple CNN stages that we can take the features from and concatenate them, so for the transformer
    we only use the features from the (last) output layer.
    """
    def __init__(self, encoder: nn.Module, embed_dim: int):
        super(VoCoEvaArchitecture, self).__init__()
        self.encoder = encoder
        self.adaptive_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.projector = VocoProjectionHead(embed_dim, 2048, 2048, norm_op=nn.InstanceNorm1d)

    def forward(self, x):
        out = self.encoder(x)
        flat_out = self.adaptive_pool(out)
        flat_out = torch.reshape(flat_out, (flat_out.shape[0], -1))
        x = self.projector(flat_out)
        return x
