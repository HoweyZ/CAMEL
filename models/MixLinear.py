import torch
import torch.nn as nn
from layers.Embed import PositionalEmbedding
import math
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        # get parameters
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.period_len = configs.period_len

        self.kernel = self.period_len
        self.lpf = configs.lpf
        self.alpha = configs.alpha

        self.seg_num_x = math.ceil(self.seq_len / self.period_len)
        self.seg_num_y = math.ceil(self.pred_len / self.period_len)

        self.sqrt_seg_num_x = math.ceil(math.sqrt(self.seq_len / self.period_len))
        self.sqrt_seg_num_y = math.ceil(math.sqrt(self.pred_len / self.period_len))

        # TLinear
        self.TLinear1 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)
        self.TLinear2 = nn.Linear(self.sqrt_seg_num_x, self.sqrt_seg_num_y, bias=False)

        self.conv1d = nn.Conv1d(in_channels=1, out_channels=1, kernel_size=self.kernel + 1,
                                stride=1, padding=int(self.kernel / 2), padding_mode="zeros", bias=False)

        # FLinear
        self.FLinear1 = nn.Linear(self.lpf, 2, bias=False)
        self.FLinear2 = nn.Linear(2, self.seg_num_y, bias=False)

    # def forward(self, x):
    #     batch_size = x.shape[0]
    #     # normalization and permute     b,s,c -> b,c,s
    #     seq_mean = torch.mean(x, dim=1).unsqueeze(1)
    #     x = (x - seq_mean).permute(0, 2, 1)
    #     # print(x.shape)
    #     x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(-1, self.enc_in, self.seq_len) + x
    #     # print(x.shape)

    #     # ->b,e,w,n
    #     x = x.reshape(batch_size, self.enc_in, -1, self.period_len).permute(0, 1, 3, 2)

    #     # Time Domain


    #     # x_o = torch.zeros(batch_size, self.enc_in, self.period_len, self.sqrt_seg_num_x ** 2).to(x.device)
    #     # x_o[:, :, :, :x.shape[-1]] = x[:, :, :, :]

    #     x_o = F.pad(x, (0, self.sqrt_seg_num_x ** 2 - x.shape[-1], 0, 0, 0, 0))

    #     x_o = x_o.reshape(batch_size, self.enc_in, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)

    #     x_o = self.TLinear1(x_o).permute(0, 1, 2, 4, 3)
    #     x_t = self.TLinear2(x_o).permute(0, 1, 2, 4, 3)

    #     x_t = x_t.reshape(batch_size, self.enc_in, self.period_len, -1).permute(0, 1, 3, 2).reshape(batch_size,
    #                                                                                                 self.enc_in,
    #                                                                                                 -1).permute(0, 2, 1)

    #     # Frequency Domain

    #     x_fft = torch.fft.fft(x, dim=3)[:, :, :, :self.lpf]
    #     # x_fft = x_fft.view(-1,self.lpf)
    #     x_fft = self.FLinear1(x_fft)
    #     x_fft = self.FLinear2(x_fft).reshape(batch_size, self.enc_in, self.period_len, -1)

    #     x_rfft = torch.fft.ifft(x_fft, dim=3).float()
    #     x_f = x_rfft.permute(0, 1, 3, 2).reshape(batch_size, self.enc_in, -1).permute(0, 2, 1)

    #     print("shape", x_t.shape, x_f.shape)

    #     # Mix
    #     print("shape", x_t[:, :self.pred_len, :].shape, x_f[:, :self.pred_len, :].shape)

    #     x = x_t[:, :self.pred_len, :] * self.alpha + seq_mean + x_f[:, :self.pred_len, :] * (1 - self.alpha)

    #     return x[:, :self.pred_len, :]
    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        """
        Forward function compatible with DLinear/Exp_Long_Term_Forecast.

        Args:
            x_enc: [B, seq_len, enc_in] input sequence
            x_mark_enc, x_dec, x_mark_dec, mask: unused, kept for compatibility

        Returns:
            x: [B, pred_len, enc_in] predicted sequence
        """
        B, S, C = x_enc.shape
        # normalize
        seq_mean = x_enc.mean(dim=1, keepdim=True)  # [B,1,C]
        x = (x_enc - seq_mean).permute(0, 2, 1)    # [B, C, seq_len]

        # Conv1d
        x = self.conv1d(x.reshape(-1, 1, self.seq_len)).reshape(B, C, self.seq_len) + x

        # Time domain: TLinear
        x_td = x.reshape(B, C, -1, self.period_len).permute(0, 1, 3, 2)
        x_td = F.pad(x_td, (0, self.sqrt_seg_num_x ** 2 - x_td.shape[-1], 0, 0, 0, 0))
        x_td = x_td.reshape(B, C, self.period_len, self.sqrt_seg_num_x, self.sqrt_seg_num_x)
        x_td = self.TLinear1(x_td).permute(0, 1, 2, 4, 3)
        x_td = self.TLinear2(x_td).permute(0, 1, 2, 4, 3)
        x_td = x_td.reshape(B, C, self.period_len, -1).permute(0, 1, 3, 2)
        x_td = x_td.reshape(B, C, -1).permute(0, 2, 1)  # [B, seq_len, C]

        # Frequency domain: FFT + FLinear
        x_fd = torch.fft.fft(x, dim=2).real[:, :, :self.lpf]  # 只取实部
        B, C, L = x_fd.shape
        x_fd = x_fd.reshape(-1, L)          # flatten
        x_fd = self.FLinear1(x_fd)          # [-1, 2]
        x_fd = self.FLinear2(x_fd)          # [-1, seg_num_y]
        x_fd = x_fd.reshape(B, C, -1)       # reshape back
        x_fd = x_fd.permute(0, 2, 1)        # [B, seq_len, C]
        # Mix time and frequency domain
        x_fd = F.interpolate(x_fd.permute(0, 2, 1), size=self.pred_len, mode='linear', align_corners=False).permute(0, 2, 1)
        x_out = x_td[:, :self.pred_len, :] * self.alpha + seq_mean + x_fd[:, :self.pred_len, :] * (1 - self.alpha)

        return x_out[:, :self.pred_len, :]