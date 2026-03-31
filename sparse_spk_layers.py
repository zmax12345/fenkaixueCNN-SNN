import torch
from torch import nn
import MinkowskiEngine as ME
from spk_layers import SurrogateHeaviside


class SparseSpikingConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, out_shape, stride, return_dense=False, bias=False):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.out_shape = out_shape
        self.kernel = kernel

        self.spike_fn = SurrogateHeaviside.apply
        self.return_dense = return_dense
        self.eps = 1e-8

        # --- MSF 神经元新增超参数 ---
        self.D = 4  # 最大突触连接数（即单步最大脉冲数），论文推荐值为 4
        self.h = 1.0  # 阈值间隔，论文严格证明最优解为 1.0 以保证梯度稳定
        # -----------------------------

        self.beta = nn.Parameter(torch.FloatTensor(1))
        self.b = nn.Parameter(torch.FloatTensor(out_channels))  # 初始阈值 V_th
        self.conv = ME.MinkowskiConvolution(in_channels, out_channels, kernel,
                                            stride=stride, bias=False, dimension=2)

        self.reset_parameters()

    def forward(self, input, mem, bs, scale=1.):
        conv_sparse = self.conv(input)
        conv_dense = conv_sparse.dense(
            shape=torch.Size([bs, self.out_channels, *self.out_shape])
        )[0]

        b = self.b.unsqueeze(1).unsqueeze(1).repeat((1, *self.out_shape))
        norm = (self.conv.kernel ** 2).sum((0, 1))

        if mem is None:
            mem = torch.zeros((bs, self.out_channels, *self.out_shape))
            mem = mem.type_as(input.C)

        # 膜电位泄漏与积分 (Leaky & Integrate)
        new_mem = mem * self.beta + conv_dense * (1. - self.beta)

        # 基础膜电位与初始阈值的差值 (U - V_th)
        mthr = torch.einsum("abcd,b->abcd", new_mem, 1. / (norm + self.eps)) - b

        # --- MSF 多阈值脉冲发放 (Multi-Synaptic Firing) ---
        spk = 0
        for d in range(self.D):
            # 依次判断膜电位是否越过 V_th + d*h
            # PyTorch 会在 backward 时自动将这 D 个 SurrogateHeaviside 的梯度相加
            spk += self.spike_fn(mthr - d * self.h, scale)
        # ----------------------------------------------------

        # --- 膜电位重置 (Hard Reset) ---
        # 论文要求：一旦触发脉冲(spk > 0)，膜电位归零；否则保留 new_mem
        spk_mask = (spk > 0).float()
        final_mem = new_mem * (1. - spk_mask)
        # -------------------------------

        if self.return_dense:
            return spk, final_mem
        else:
            # 由于 MSF 脉冲现在是 0~4 的整数，依然可以用于构建稀疏张量
            p_spkF = spk.permute(1, 0, 2, 3).contiguous().view(self.out_channels, -1).t()
            spkF = p_spkF[p_spkF.sum(dim=1) != 0]

            spkC_temp = torch.nonzero(spk)[:, (0, 2, 3)]
            spkF_temp = torch.zeros((spkC_temp.shape[0],))
            spkF_temp = spkF_temp.type_as(input.C)
            torch_sparse_tensor = torch.sparse_coo_tensor(
                spkC_temp.t().to(torch.int32),
                spkF_temp.to(torch.int32),
            ).coalesce()
            spkC = torch_sparse_tensor._indices().t().contiguous().to(torch.int)

            final_spk = ME.SparseTensor(spkF, spkC)

            return final_spk, final_mem

    def reset_parameters(self):
        torch.nn.init.normal_(self.beta, mean=0.8, std=0.01)
        torch.nn.init.normal_(self.b, mean=0.1, std=0.01)
        torch.nn.init.xavier_uniform_(self.conv.kernel.data, torch.nn.init.calculate_gain('sigmoid'))

    def clamp(self, min_beta=0., max_beta=1., min_b=0.):
        self.beta.data.clamp_(min_beta, max_beta)
        self.b.data.clamp_(min=min_b)