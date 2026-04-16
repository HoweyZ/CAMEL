import torch
import torch.nn as nn


class RecurrentCycle(torch.nn.Module):
    def __init__(self, cycle_len, channel_size):
        super(RecurrentCycle, self).__init__()
        self.cycle_len = cycle_len
        self.channel_size = channel_size
        self.data = torch.nn.Parameter(
            torch.zeros(cycle_len, channel_size), requires_grad=True
        )

    def forward(self, index, length):
        gather_index = (
            index.view(-1, 1)
            + torch.arange(length, device=index.device).view(1, -1)
        ) % self.cycle_len
        return self.data[gather_index]


class Time_seg(nn.Module):
    def __init__(self, input_len, pred_len, window_size, stride, d_model, window_type):
        super().__init__()
        self.input_len = input_len
        self.pred_len = pred_len
        self.window_size = window_size
        self.stride = stride
        self.seg_num = (input_len - window_size) // stride + 1

        gaussian_sigma = 0.4

        self.windows = {}
        for i in range(self.seg_num):
            start = i * stride
            end = start + window_size

            if window_type == 'hamming':
                window = torch.hamming_window(window_size, periodic=False)
            elif window_type == 'hann':
                window = torch.hann_window(window_size, periodic=False)
            elif window_type == 'gaussian':
                x = torch.linspace(-3, 3, window_size)
                window = torch.exp(-x**2 / (2 * gaussian_sigma**2))
                window = window / window.max()
            else:
                window = torch.ones(window_size)

            full_window = torch.zeros(self.input_len)
            full_window[start:end] = window
            self.register_buffer(f"window_{i}", full_window)

        self.time_linear = nn.Sequential(
            nn.Linear(self.input_len, d_model),
            nn.ReLU(),
            nn.Linear(d_model, self.pred_len)
        )

    def forward(self, x, Tweight):
        # x: [B, D, L]
        x = x.unsqueeze(2)  # [B, D, 1, L]
        segx = torch.zeros(
            (x.shape[0], x.shape[1], self.seg_num, x.shape[3]),
            device=x.device,
            dtype=x.dtype
        )

        for i in range(self.seg_num):
            window = getattr(self, f"window_{i}").to(x.device, dtype=x.dtype)
            nowx = x * window
            segx[:, :, i:i+1, :] = nowx

        segw = torch.fft.rfft(segx, dim=-1)  # [B, D, S, L//2+1]

        FTweight = torch.softmax(Tweight, dim=0).to(segw.dtype)
        segw = segw * FTweight.unsqueeze(0).unsqueeze(0)
        segw = torch.sum(segw, dim=2)  # [B, D, L//2+1]

        y = torch.fft.irfft(segw, n=self.input_len, dim=-1)
        y = self.time_linear(y)
        return y


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in

        self.cycle_len = configs.cycle
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin

        self.seg_window = configs.seg_window
        self.seg_stride = configs.seg_stride
        self.window_type = configs.window_type

        self.seg_num = (self.seq_len - self.seg_window) // self.seg_stride + 1

        self.cycleQueue = RecurrentCycle(
            cycle_len=self.cycle_len,
            channel_size=self.enc_in
        )

        self.Tweight = nn.Parameter(
            (1 / self.seg_num) * torch.ones(self.seg_num, self.seq_len // 2 + 1),
            requires_grad=True
        )

        assert self.model_type in ['linear', 'mlp']
        if self.model_type == 'linear':
            self.model = nn.Linear(self.seq_len, self.pred_len)
        else:
            self.model = nn.Sequential(
                nn.Linear(self.seq_len, self.d_model),
                nn.ReLU(),
                nn.Linear(self.d_model, self.pred_len)
            )

        self.seg_layer = Time_seg(
            self.seq_len,
            self.pred_len,
            self.seg_window,
            self.seg_stride,
            configs.d_model,
            self.window_type
        )

    def _build_cycle_index(self, x_mark_enc, batch_size, device):
        """
        尽量从 x_mark_enc 推一个 cycle_index。
        约定优先级：
        1. 如果 x_mark_enc 存在，并且最后一维至少有 4 列，默认第 4 列是 hour / slot
        2. 否则退化成全 0
        """
        if x_mark_enc is None:
            return torch.zeros(batch_size, dtype=torch.long, device=device)

        # 常见 timeF 格式里，最后一维通常包含 month/day/weekday/hour 之类
        if x_mark_enc.dim() == 3 and x_mark_enc.size(-1) >= 4:
            # 取输入序列起点的 hour/slot
            raw_idx = x_mark_enc[:, 0, 3]

            # 兼容整数 hour 或归一化时间特征
            if raw_idx.dtype.is_floating_point:
                # 如果是 0~1 之间，缩放到 cycle_len
                if raw_idx.min() >= 0 and raw_idx.max() <= 1.0:
                    cycle_index = torch.round(raw_idx * (self.cycle_len - 1)).long()
                else:
                    cycle_index = torch.round(raw_idx).long()
            else:
                cycle_index = raw_idx.long()

            cycle_index = cycle_index % self.cycle_len
            return cycle_index.to(device)

        return torch.zeros(batch_size, dtype=torch.long, device=device)

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        # x_enc: [B, seq_len, enc_in]
        B = x_enc.size(0)
        device = x_enc.device

        cycle_index = self._build_cycle_index(x_mark_enc, B, device)

        x = x_enc

        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        x_period = self.cycleQueue(cycle_index, self.seq_len)
        x = x - x_period

        y = self.seg_layer(
            x.permute(0, 2, 1),
            self.Tweight.to(x.device, dtype=x.dtype)
        ).permute(0, 2, 1)

        y_period = self.cycleQueue(
            (cycle_index + self.seq_len) % self.cycle_len,
            self.pred_len
        )
        y = y + y_period

        if self.use_revin:
            y = y * torch.sqrt(seq_var) + seq_mean

        return y[:, -self.pred_len:, :]