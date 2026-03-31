import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
from sparse_spk_layers import SparseSpikingConv2D


# --- 网络模型定义 ---
class SNN_CNN_Hybrid(nn.Module):
    def __init__(self):
        super(SNN_CNN_Hybrid, self).__init__()
        # SNN 输入: 100x768。经过两次 stride=2 的卷积
        self.snn_enc1 = SparseSpikingConv2D(in_channels=1, out_channels=16, kernel=(5, 5), out_shape=(50, 184),
                                            stride=(2, 2))
        self.snn_enc2 = SparseSpikingConv2D(in_channels=16, out_channels=32, kernel=(3, 3), out_shape=(25, 92),
                                            stride=(2, 2), return_dense=True)

        # CNN 解码器 (修改为输出 2 个通道)
        self.cnn_dec = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # 使用转置卷积放大 2 倍 (从 25x92 到 50x184)
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),

            # 再使用一次转置卷积放大 2 倍 (从 50x184 恢复到 100x368)
            # 【关键修改】：out_channels 从 1 改为 2
            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=4, stride=2, padding=1),

            nn.Softplus()  # 保证提取的物理参数都是正数
        )

    def forward(self, x_seq, actual_batch_size):
        mem1, mem2 = None, None

        for x_sparse in x_seq:
            out1, mem1 = self.snn_enc1(x_sparse, mem=mem1, bs=actual_batch_size)
            out2, mem2 = self.snn_enc2(out1, mem=mem2, bs=actual_batch_size)

        # 输出双通道物理特征: [Batch, 2, 100, 368] (假设最终宽度是368)
        out = self.cnn_dec(mem2)

        # 【关键解耦】：通道 0 负责时间特征，通道 1 负责空间物理特征
        inv_tau_c_pred = out[:, 0:1, :, :]  # 去相关速率 1/tau_c
        alpha_pred = out[:, 1:2, :, :]  # 光学/多重散射空间校正因子 alpha

        return inv_tau_c_pred, alpha_pred


# --- 序列合并函数 ---
def sequence_sparse_collate(batch):
    seq_len = len(batch[0][0])
    batched_seq = []
    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.batch_sparse_collate(coords_t, feats_t)
        batched_seq.append(ME.SparseTensor(features=b_feats, coordinates=b_coords))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    return batched_seq, labels