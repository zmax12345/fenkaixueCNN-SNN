import os
import gc  # 【新增】必须导入垃圾回收模块
import matplotlib.pyplot as plt

# 防卡死核心设置
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'

import torch
import torch.multiprocessing
# 【关键修复】突破 DataLoader 的多进程文件描述符限制
torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import CelexBloodFlowDataset, sequence_sparse_collate
from model import SNN_CNN_Hybrid


def structural_similarity_loss(img1, img2):
    # 逻辑保持不变
    mu1 = img1.mean(dim=[2, 3], keepdim=True)
    mu2 = img2.mean(dim=[2, 3], keepdim=True)
    img1_zero_mean = img1 - mu1
    img2_zero_mean = img2 - mu2

    numerator = (img1_zero_mean * img2_zero_mean).sum(dim=[2, 3])
    var1 = (img1_zero_mean ** 2).sum(dim=[2, 3])
    var2 = (img2_zero_mean ** 2).sum(dim=[2, 3])
    denominator = torch.sqrt(var1 * var2 + 1e-8)

    ncc = numerator / denominator
    return (1.0 - ncc).mean()


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

    print("正在加载训练集 (启用空间成对采样)...")
    train_dataset = CelexBloodFlowDataset(data_config=train_config, mask_path=MASK_PATH)

    # 【修改】大幅降本增效的 DataLoader 设置
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_sparse_collate,
        num_workers=4,  # 降为4，避免吃光内存
        pin_memory=False,  # 关闭锁页内存，防止触发系统 Swap
        prefetch_factor=2,
        persistent_workers=False  # 关闭驻留，让每个Epoch能彻底回收内存
    )

    print("正在加载验证集...")
    val_dataset = CelexBloodFlowDataset(data_config=val_config, mask_path=MASK_PATH)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_sparse_collate,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False
    )

    model = SNN_CNN_Hybrid().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 【新增】余弦退火学习率调度器：让模型在后期收敛得更好
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)

    best_val_loss = float('inf')

    history_train_loss = []
    history_val_loss = []

    lambda_phys = 0.5

    for epoch in range(NUM_EPOCHS):
        # ----------------- 训练阶段 -----------------
        model.train()
        train_loss_total = 0.0

        # 显示当前的学习率
        current_lr = optimizer.param_groups[0]['lr']
        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Train (LR:{current_lr:.1e})")

        for batch_idx, (seq_data, y_true, d_values, dense_maps) in enumerate(pbar_train):
            x_seq = []
            for b_coords, b_feats in seq_data:
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)
            dense_maps = dense_maps.to(DEVICE)

            optimizer.zero_grad()

            v_pred_model, alpha_pred, inv_tau_c_pred = model(x_seq, dense_maps, actual_batch_size=len(y_true))

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            loss_data = F.mse_loss(v_pred, y_true_expanded)

            if alpha_pred.shape[0] >= 2:
                alpha_anchor = alpha_pred[0::2]
                alpha_pos = alpha_pred[1::2]
                loss_physics = structural_similarity_loss(alpha_anchor, alpha_pos)
            else:
                loss_physics = torch.tensor(0.0).to(DEVICE)

            loss = loss_data + lambda_phys * loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            pbar_train.set_postfix({'Total': f"{loss.item():.3f}", 'Data': f"{loss_data.item():.3f}",
                                    'Phys_SSIM': f"{loss_physics.item():.3f}"})

            # 【新增】核心防溢出与波动机制
            del x_seq, sparse_tensor, v_pred_model, alpha_pred, inv_tau_c_pred, dense_maps, v_pred, y_true_expanded
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_train_loss = train_loss_total / len(train_loader)
        history_train_loss.append(avg_train_loss)

        # ----------------- 验证阶段 -----------------
        model.eval()
        val_loss_total = 0.0
        pbar_val = tqdm(val_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Validation")

        with torch.no_grad():
            for batch_idx, (seq_data, y_true, d_values, dense_maps) in enumerate(pbar_val):
                x_seq = []
                for b_coords, b_feats in seq_data:
                    sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                    x_seq.append(sparse_tensor)

                y_true = y_true.to(DEVICE)
                d_values = d_values.to(DEVICE)
                dense_maps = dense_maps.to(DEVICE)

                v_pred_model, alpha_pred, inv_tau_c_pred = model(x_seq, dense_maps, actual_batch_size=len(y_true))

                d_values_expanded = d_values.view(-1, 1, 1, 1)
                v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
                y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

                loss = F.mse_loss(v_pred, y_true_expanded)
                val_loss_total += loss.item()
                pbar_val.set_postfix({'Val_MSE': f"{loss.item():.4f}"})

                # 【新增】严重隐患区：验证阶段也要清空显存对象
                del x_seq, sparse_tensor, v_pred_model, alpha_pred, inv_tau_c_pred, dense_maps, v_pred, y_true_expanded
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

        avg_val_loss = val_loss_total / len(val_loader)
        history_val_loss.append(avg_val_loss)
        print(f"--> Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 【新增】更新学习率
        scheduler.step()

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs("/data/zm/Moshaboli/new_data/Model", exist_ok=True)
            torch.save(model.state_dict(), "/data/zm/Moshaboli/new_data/Model/best_hybrid_model.pth")
            print(f"[*] 已保存最佳模型 (Val Loss: {best_val_loss:.4f})")

        # 【新增】Epoch 级别大扫除，防止内存泄漏积累
        gc.collect()
        torch.cuda.empty_cache()

    # 绘制曲线
    print("正在绘制并保存 Loss 曲线...")
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, NUM_EPOCHS + 1), history_train_loss, label='Train Total Loss', marker='o', color='blue')
    plt.plot(range(1, NUM_EPOCHS + 1), history_val_loss, label='Validation MSE Loss', marker='s', color='orange')
    plt.title('Physics-Informed Two-Stream Training Curve', fontsize=16)
    plt.xlabel('Epochs', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)

    os.makedirs("/data/zm/Moshaboli/new_data/Loss_curve", exist_ok=True)
    plot_path = "/data/zm/Moshaboli/new_data/Loss_curve/loss_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Loss 曲线已成功保存至: {plot_path}")


if __name__ == '__main__':
    train_and_evaluate()
