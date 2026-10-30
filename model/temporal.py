import torch
import torch.nn as nn
import torch.nn.functional as F


class ShiftPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 32),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(32, 16),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        diff = x[:, 1:] - x[:, :-1]
        var = torch.var(diff, dim=2)
        var = F.pad(var, (1, 0), mode="replicate")
        shift_ratio = self.fc(var.unsqueeze(-1))
        return shift_ratio


class TemporalTransformer(nn.Module):
    def __init__(self, dim, num_heads=4, depth=2):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=num_heads, dropout=0.3, batch_first=True)
            for _ in range(depth)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class GlobalAdaptiveTSM(nn.Module):
    def __init__(self, input_dim=None, fold_div=2, num_heads=4, depth=2):
        super().__init__()
        self.fold_div = fold_div
        self.num_heads = num_heads
        self.depth = depth
        self.predictor = ShiftPredictor()
        self.input_dim = input_dim

        if input_dim is not None:

            self.transformer = TemporalTransformer(dim=input_dim, num_heads=num_heads, depth=depth)
        else:

            self.transformer = None

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False

        B, T, D = x.shape
        fold = D // self.fold_div
        out = torch.zeros_like(x)

        if self.transformer is None:
            self.transformer = TemporalTransformer(dim=D, num_heads=self.num_heads, depth=self.depth).to(x.device)

        shift_ratio = self.predictor(x).squeeze(-1)
        shift_channels = (shift_ratio * fold).long()

        for b in range(B):
            for t in range(T):
                shift_n = shift_channels[b, t].item()

                if t > 0 and shift_n > 0:
                    out[b, t, :shift_n] = x[b, t-1, :shift_n]
                if t < T - 1 and shift_n > 0:
                    out[b, t, shift_n:2*shift_n] = x[b, t+1, shift_n:2*shift_n]
                out[b, t, 2*shift_n:] = x[b, t, 2*shift_n:]

        out = self.transformer(out)

        if squeeze:
            out = out.squeeze(0)

        return out
