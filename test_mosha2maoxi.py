import os
import torch

# 防卡死核心设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from collections import defaultdict

# 防卡死核心设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def generalization_test_v2():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/micro_hot_pixel_mask_strict.npy"
    FINETUNED_MODEL = "/data/zm/Moshaboli/new_data/Model/SNNENABLE_best_capillary_finetuned.pth"

    # 【核心逻辑优化】：设定稳态过滤参数
    SKIP_WINDOWS = 50  # 跳过每个文件的前50个窗口（避开机械不稳定期）

    # 【配置测试集】：建议优先放在同一组（如 2.3）内验证我们的物理一致性假设
    test_config = {
        "/data/zm/2026.1.12_testdata/2.3/0.5mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/1.0mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/1.2mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/1.5mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/1.8mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/2.2mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/2.5mm_clip.csv": 0.0101,
    }

    print(f"[*] 正在加载微调后的标定模型: {FINETUNED_MODEL}")
    model = SNN_CNN_Hybrid().to(DEVICE)
    model.load_state_dict(torch.load(FINETUNED_MODEL))
    model.eval()

    dataset = CelexBloodFlowDataset(data_config=test_config, mask_path=MASK_PATH)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=sequence_sparse_collate)

    # 阶段 A：提取并锁定 Alpha Map (原位物理标定因子)
    print("[*] 正在提取原位物理校正因子 (Alpha Map)...")
    with torch.no_grad():
        first_batch = next(iter(dataloader))
        x_seq = [ME.SparseTensor(f.to(DEVICE), c.to(DEVICE)) for c, f in first_batch[0]]
        _, alpha_fixed = model(x_seq, actual_batch_size=1)
        alpha_fixed = alpha_fixed.detach()

    # 阶段 B：聚合统计化预测
    print("[*] 开始执行统计化泛化考评...")
    velocity_groups = defaultdict(list)

    with torch.no_grad():
        for seq_data, y_true, d_values in dataloader:
            v_true_key = round(y_true.item(), 3)  # 使用真实流速作为 Key 进行分组

            x_seq = [ME.SparseTensor(f.to(DEVICE), c.to(DEVICE)) for c, f in seq_data]

            # 预测时间特征 1/tau_c
            inv_tau_c_pred, _ = model(x_seq, actual_batch_size=1)

            # 使用锁定的 Alpha 计算物理流速
            d_val = d_values[0].to(DEVICE)
            v_pred_map = (d_val / (alpha_fixed + 1e-6)) * inv_tau_c_pred
            v_pred_mean = v_pred_map.mean().item()

            velocity_groups[v_true_key].append(v_pred_mean)

    # 阶段 C：结果处理与定量分析
    final_gt = []
    final_pred_mean = []
    final_pred_std = []

    print("\n" + "=" * 60)
    print(f"{'True Vel (mm/s)':<15} | {'Mean Pred':<12} | {'Std Dev':<10} | {'Valid Samples'}")
    print("-" * 60)

    for gt in sorted(velocity_groups.keys()):
        all_preds = velocity_groups[gt]
        # 剔除起步不稳的样本
        valid_preds = all_preds[SKIP_WINDOWS:] if len(all_preds) > SKIP_WINDOWS else all_preds

        m_v = np.mean(valid_preds)
        s_v = np.std(valid_preds)

        final_gt.append(gt)
        final_pred_mean.append(m_v)
        final_pred_std.append(s_v)

        print(f"{gt:<15.3f} | {m_v:<12.4f} | {s_v:<10.4f} | {len(valid_preds)}")

    # 计算全局学术指标
    final_gt = np.array(final_gt)
    final_pred_mean = np.array(final_pred_mean)
    r2 = r2_score(final_gt, final_pred_mean)
    mape = np.mean(np.abs((final_gt - final_pred_mean) / (final_gt + 1e-8))) * 100

    print("=" * 60)
    print(f"统计后全局指标: R² = {r2:.4f} | MAPE = {mape:.2f}%")

    # 绘制带误差棒的学术散点图
    plt.figure(figsize=(7, 6))
    plt.errorbar(final_gt, final_pred_mean, yerr=final_pred_std, fmt='o', color='blue',
                 ecolor='lightblue', elinewidth=2, capsize=4, label='Predicted (Mean ± Std)')

    # 理想 y=x 线
    lims = [0, max(final_gt.max(), final_pred_mean.max()) * 1.1]
    plt.plot(lims, lims, 'r--', alpha=0.7, label='Ideal y=x')

    plt.title("Generalization Performance (Steady State Statistics)", fontsize=14)
    plt.xlabel("Ground Truth Velocity (mm/s)", fontsize=12)
    plt.ylabel("Predicted Velocity (mm/s)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.legend()

    save_path = "/data/zm/Moshaboli/new_data/Loss_curve/generalization_statistical.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n[*] 统计散点图已保存至: {save_path}")
    plt.show()


if __name__ == '__main__':
    generalization_test_v2()