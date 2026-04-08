import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME
from sparse_spk_layers import SparseSpikingConv2D


# --- 并行双流 SNN-CNN 混合架构 ---
class SNN_CNN_Hybrid(nn.Module):
    def __init__(self):
        super(SNN_CNN_Hybrid, self).__init__()

        # ==========================================
        # 1. 时间流分支 (SNN Temporal Stream)
        # 负责处理极度稀疏的异步事件流，提取流速的高频时域特征 (1/tau_c)
        # ==========================================
        # SNN 编码器: 降维提取脉冲特征
        self.snn_enc1 = SparseSpikingConv2D(in_channels=1, out_channels=16, kernel=(5, 5), out_shape=(50, 184),
                                            stride=(2, 2))
        self.snn_enc2 = SparseSpikingConv2D(in_channels=16, out_channels=32, kernel=(3, 3), out_shape=(25, 92),
                                            stride=(2, 2), return_dense=True)

        # SNN 解码器: 将累加后的时间特征图恢复分辨率，并分离出 [1/tau_c] 和 [动态校正项 phi]
        self.snn_dec = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            # 输出 2 个通道：通道0对应 inv_tau_c，通道1对应动态物理校正因子 phi_v
            nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=4, stride=2, padding=1)
        )

        # ==========================================
        # 2. 空间流分支 (CNN Spatial Stream)
        # 负责处理去除了流速信息的空间归一化累加图，提取静态物理背景 (alpha_0)
        # ==========================================
        self.spatial_cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            # 这里不直接用 Softplus，留到融合时再约束为正数
        )

    def forward(self, x_seq, dense_map, actual_batch_size):
        """
        x_seq: SNN的输入，稀疏张量序列
        dense_map: CNN的输入，空间二维实例归一化图 [Batch, 1, 100, 368]
        actual_batch_size: 当前批次大小
        """

        # --- 步骤 1: SNN 时间特征提取与累加 (解决时序截断问题) ---
        mem1, mem2 = None, None
        accumulated_spikes = 0  # 时间累加器

        for x_sparse in x_seq:
            out1, mem1 = self.snn_enc1(x_sparse, mem=mem1, bs=actual_batch_size)
            out2, mem2 = self.snn_enc2(out1, mem=mem2, bs=actual_batch_size)

            # out2 是 return_dense=True 返回的稠密特征
            # 在整个时间窗口内累加脉冲/特征，将时间动力学转化为空间上的特征强度
            accumulated_spikes = accumulated_spikes + out2

        # 解码 SNN 累加特征
        temporal_features = self.snn_dec(accumulated_spikes)

        # 提取散斑去相关频率代理变量 1/tau_c (必须为正，使用 softplus)
        inv_tau_c = F.softplus(temporal_features[:, 0:1, :, :])
        # 提取随流速变化的非线性散射校正项 phi_v
        phi_v = temporal_features[:, 1:2, :, :]

        # --- 步骤 2: CNN 空间特征提取 (解决虚假解耦问题) ---
        # 提取不受流速影响的静态空间背景 alpha_0
        alpha_0 = self.spatial_cnn(dense_map)

        # --- 步骤 3: 物理先验融合 (Physics-Informed Fusion) ---
        # 真正的物理散射因子 = 静态管壁背景 + 高流速红细胞形变带来的动态散射校正
        # 使用 Softplus 确保最终计算的物理因子一定是正数
        alpha_dynamic = F.softplus(alpha_0 + phi_v)

        # 基于物理公式预测最终流速场: V = alpha * (1/tau_c)
        v_pred = alpha_dynamic * inv_tau_c

        return v_pred, alpha_dynamic, inv_tau_c


# --- 序列合并函数 (适配成对采样) ---
def sequence_sparse_collate(batch):
    # 这个函数需要和我们修改后的 dataset.py 配合
    # batch 此时是已经展平的 [(anchor), (pos), (anchor), (pos)...]
    seq_len = len(batch[0][0])
    batched_seq = []

    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in batch]
        feats_t = [sample[0][t][1] for sample in batch]
        b_coords, b_feats = ME.utils.batch_sparse_collate(coords_t, feats_t)
        batched_seq.append(ME.SparseTensor(features=b_feats, coordinates=b_coords))

    labels = torch.tensor([sample[1] for sample in batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in batch], dtype=torch.float32)
    # 提取送给 CNN 的二维累加图
    dense_maps = torch.stack([sample[3] for sample in batch], dim=0)

    return batched_seq, labels, d_values, dense_maps
