import os
import matplotlib.pyplot as plt

# 防卡死核心设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def adjust_and_evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 30
    BATCH_SIZE = 4
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_strict.npy"

    # 【注意】：毛细玻璃管数据集路径
    # ============================================================
    # 建议的微调 Config (专注于 2.3 组的物理一致性)
    # ============================================================

    # 1. 训练集：选取高低两个极端，让 SNN 建立起基本的“速度-时间响应”斜率
    capillary_train_config = {
        "/data/zm/2026.1.12_testdata/2.3/0.2mm_clip.csv": 0.0101,  # 极低速
        "/data/zm/2026.1.12_testdata/2.3/2.0mm_clip.csv": 0.0101,  # 高速
    }

    # 2. 验证集：选取一个中等流速，用于监控微调是否过拟合
    capillary_val_config = {
        "/data/zm/2026.1.12_testdata/2.3/0.8mm_clip.csv": 0.0101,
    }

    # 3. 泛化集 (用于运行 generalization_test.py 时手动指定)
    # 剩下的 2.3 组数据：0.5mm/s (用于测试插值能力)
    # 以及其他组的数据 (用于测试跨环境的失效点，作为论文的对比讨论)

    print("正在加载毛细玻璃管训练集（用于物理信息微调）...")
    train_dataset = CelexBloodFlowDataset(data_config=capillary_train_config, mask_path=MASK_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_sparse_collate, num_workers=8,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    print("正在加载毛细玻璃管验证集（用于监测流速预测准确度）...")
    val_dataset = CelexBloodFlowDataset(data_config=capillary_val_config, mask_path=MASK_PATH)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_sparse_collate, num_workers=8,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    # ================= 1. 初始化模型并加载预训练权重 =================
    model = SNN_CNN_Hybrid().to(DEVICE)
    pretrained_path = "/data/zm/Moshaboli/new_data/Model/PINN_hybrid_model_0.02.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"[*] 成功加载毛玻璃预训练权重: {pretrained_path}")
    else:
        raise FileNotFoundError("未找到预训练模型，请先完成阶段二的训练！")

    # ================= 2. 彻底解冻 SNN，进行全域联合微调 =================
    print("[*] 正在解冻 SNN 编码器参数，允许其适应毛细管的时间动力学...")
    for param in model.snn_enc1.parameters():
        param.requires_grad = True
    for param in model.snn_enc2.parameters():
        param.requires_grad = True

    for param in model.cnn_dec.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 参与联合微调的总参数量: {trainable_params}")

    # ================= 3. 差异化学习率配置 =================
    # 核心逻辑：SNN 只需要微调神经元阈值适应低频事件（极小步子），CNN 继续刻画空间管壁（正常微调步子）
    optimizer = torch.optim.Adam([
        {'params': model.snn_enc1.parameters(), 'lr': 1e-6},
        {'params': model.snn_enc2.parameters(), 'lr': 1e-6},
        {'params': model.cnn_dec.parameters(), 'lr': 5e-5}
    ])

    best_val_loss = float('inf')
    history_phys_loss = []
    history_data_loss = []
    history_val_mse = []

    for epoch in range(NUM_EPOCHS):
        # ----------------- 联合微调阶段 -----------------
        model.train()

        train_loss_total = 0.0
        train_data_total = 0.0
        train_phys_total = 0.0

        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] PINN Fine-tuning")

        for batch_idx, (seq_data, y_true, d_values) in enumerate(pbar_train):
            x_seq = []
            for b_coords, b_feats in seq_data:
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)

            optimizer.zero_grad()

            inv_tau_c_pred, alpha_pred = model(x_seq, actual_batch_size=len(y_true))
            d_values_expanded = d_values.view(-1, 1, 1, 1).to(DEVICE)

            # --- 物理公式层 ---
            v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred).to(DEVICE)

            # --- 混合损失计算 ---
            # 1. 引入数据损失作为驱动力
            loss_data = F.mse_loss(v_pred, y_true_expanded)

            # 2. 物理方差约束作为解耦方向盘
            if alpha_pred.shape[0] > 1:
                loss_physics = torch.var(alpha_pred, dim=0).mean()
            else:
                loss_physics = torch.tensor(0.0).to(DEVICE)

            # --- 联合微调总损失 ---
            loss = loss_data + 0.5 * loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_data_total += loss_data.item()
            train_phys_total += loss_physics.item()

            pbar_train.set_postfix({'Total': f"{loss.item():.4f}", 'Data': f"{loss_data.item():.4f}",
                                    'Phys': f"{loss_physics.item():.4f}"})

        avg_phys_loss = train_phys_total / len(train_loader)
        avg_data_loss = train_data_total / len(train_loader)
        history_phys_loss.append(avg_phys_loss)
        history_data_loss.append(avg_data_loss)

        # ----------------- 监测验证阶段 -----------------
        model.eval()
        val_mse_total = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation Monitoring")

        with torch.no_grad():
            for seq_data, y_true, d_values in pbar_val:
                x_seq = []
                for b_coords, b_feats in seq_data:
                    sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                    x_seq.append(sparse_tensor)

                y_true = y_true.to(DEVICE)
                d_values = d_values.to(DEVICE)

                inv_tau_c_pred, alpha_pred = model(x_seq, actual_batch_size=len(y_true))
                d_values_expanded = d_values.view(-1, 1, 1, 1).to(DEVICE)

                v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred).to(DEVICE)

                loss_mse = F.mse_loss(v_pred, y_true_expanded)
                val_mse_total += loss_mse.item()
                pbar_val.set_postfix({'Val_MSE': f"{loss_mse.item():.4f}"})

        avg_val_mse = val_mse_total / len(val_loader)
        history_val_mse.append(avg_val_mse)
        print(
            f"--> Epoch {epoch + 1} | Train Data MSE: {avg_data_loss:.4f} | Phys Var Loss: {avg_phys_loss:.5f} | Val MSE: {avg_val_mse:.4f}")

        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            torch.save(model.state_dict(), "/data/zm/Moshaboli/new_data/Model/SNNENABLE_best_capillary_finetuned.pth")
            print(f"[*] 发现更优的血管微调模型，已保存 (Val MSE: {best_val_loss:.4f})")

    # 绘制曲线
    print("微调结束，正在绘制并保存微调 Loss 曲线...")
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Train Data & Val MSE', color='black', fontsize=12)
    ax1.plot(range(1, NUM_EPOCHS + 1), history_data_loss, label='Train Data MSE', marker='o', color='blue')
    ax1.plot(range(1, NUM_EPOCHS + 1), history_val_mse, label='Validation MSE', marker='s', color='orange')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.legend(loc='upper left', fontsize=10)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Physics Variance Loss', color='green', fontsize=12)
    ax2.plot(range(1, NUM_EPOCHS + 1), history_phys_loss, label='Physics Variance Loss', marker='^', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=10)

    plt.title('PINN Fine-tuning on Capillary Data', fontsize=16)
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = "/data/zm/Moshaboli/new_data/Loss_curve/adjust_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"微调曲线已成功保存至当前目录: {plot_path}")


if __name__ == '__main__':
    adjust_and_evaluate()