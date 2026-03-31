import os
import matplotlib.pyplot as plt

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


def train_and_evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 50
    BATCH_SIZE = 4
    LEARNING_RATE = 2e-4
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/hot_pixel_mask_strict.npy"

    train_config = {
        "/data/zm/Moshaboli/new_data/no5": 0.01978,
        "/data/zm/Moshaboli/new_data/no1": 0.01891,
    }

    val_config = {
        "/data/zm/Moshaboli/new_data/no2": 0.01941,
        "/data/zm/Moshaboli/new_data/no4": 0.01973
    }

    print("正在加载训练集...")
    train_dataset = CelexBloodFlowDataset(data_config=train_config, mask_path=MASK_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_sparse_collate, num_workers=8,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    print("正在加载验证集...")
    val_dataset = CelexBloodFlowDataset(data_config=val_config, mask_path=MASK_PATH)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_sparse_collate, num_workers=8,
        pin_memory=True, prefetch_factor=2, persistent_workers=True
    )

    model = SNN_CNN_Hybrid().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    best_val_loss = float('inf')

    history_train_loss = []
    history_val_loss = []

    # === PINN 超参数 (已移除 TV Loss) ===
    lambda_phys = 0.5  # 物理恒定性约束权重 (对比物理损失)

    for epoch in range(NUM_EPOCHS):
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss_total = 0.0
        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Training")

        for batch_idx, (seq_data, y_true, d_values) in enumerate(pbar_train):
            x_seq = []
            for b_coords, b_feats in seq_data:
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)

            optimizer.zero_grad()

            # 【网络输出解耦】：获取时间特征(inv_tau_c) 和 空间物理校正特征(alpha)
            inv_tau_c_pred, alpha_pred = model(x_seq, actual_batch_size=len(y_true))

            # 【物理公式层】：v = (d / alpha) * (1 / tau_c)
            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            # --- 损失计算核心区 ---
            # 1. 数据损失 (Data Loss)
            loss_data = F.mse_loss(v_pred, y_true_expanded)

            # 2. 物理恒定性约束 (Contrastive Physics Loss)
            # 迫使同一个 Batch（同一环境）内不同流速的空间特征 alpha 保持方差最小
            if alpha_pred.shape[0] > 1:
                loss_physics = torch.var(alpha_pred, dim=0).mean()
            else:
                loss_physics = torch.tensor(0.0).to(DEVICE)

            # --- 总损失 ---
            loss = loss_data + lambda_phys * loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            # 打印信息中只保留 Total, Data, Phys
            pbar_train.set_postfix({'Total': f"{loss.item():.3f}", 'Data': f"{loss_data.item():.3f}",
                                    'Phys': f"{loss_physics.item():.3f}"})

        avg_train_loss = train_loss_total / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss_total = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation")

        with torch.no_grad():
            for seq_data, y_true, d_values in pbar_val:
                x_seq = []
                for b_coords, b_feats in seq_data:
                    sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                    x_seq.append(sparse_tensor)

                y_true = y_true.to(DEVICE)
                d_values = d_values.to(DEVICE)

                inv_tau_c_pred, alpha_pred = model(x_seq, actual_batch_size=len(y_true))
                d_values_expanded = d_values.view(-1, 1, 1, 1)
                v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

                loss = F.mse_loss(v_pred, y_true_expanded)
                val_loss_total += loss.item()
                pbar_val.set_postfix({'Val_MSE': f"{loss.item():.4f}"})

        avg_val_loss = val_loss_total / len(val_loader)
        history_val_loss.append(avg_val_loss)
        print(f"--> Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "/data/zm/Moshaboli/new_data/Model/best_hybrid_model.pth")
            print(f"[*] 已保存最佳模型 (Val Loss: {best_val_loss:.4f})")

    # 绘制曲线
    print("正在绘制并保存 Loss 曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Total Loss', marker='o', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Validation MSE Loss', marker='s', color='orange')
    plt.title('PINN Training and Validation Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plot_path = "/data/zm/Moshaboli/new_data/Loss_curve/loss_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Loss 曲线已成功保存至当前目录: {plot_path}")


if __name__ == '__main__':
    train_and_evaluate()