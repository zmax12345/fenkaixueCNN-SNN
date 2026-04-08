import os
import gc
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
    """
    计算基于归一化互相关 (NCC) 的结构相似度损失。
    """
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


def adjust_and_evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    NUM_EPOCHS = 30
    BATCH_SIZE = 4
    MASK_PATH = "/data/zm/Moshaboli/new_data/other_data/micro_hot_pixel_mask_strict.npy"

    # ============================================================
    # 微调 Config
    # ============================================================
    capillary_train_config = {
        "/data/zm/2026.1.12_testdata/2.3/0.2mm_clip.csv": 0.0101,
        "/data/zm/2026.1.12_testdata/2.3/1.2mm_clip.csv": 0.0101,# 极低速
        "/data/zm/2026.1.12_testdata/2.3/2.5mm_clip.csv": 0.0101,  # 高速
    }
    capillary_val_config = {
        "/data/zm/2026.1.12_testdata/2.3/1.8mm_clip.csv": 0.0101,
    }

    print("正在加载毛细玻璃管训练集（用于物理信息微调）...")
    train_dataset = CelexBloodFlowDataset(data_config=capillary_train_config, mask_path=MASK_PATH)
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=sequence_sparse_collate,
        num_workers=4,  # 降频保显存
        pin_memory=False,  # 解除内存锁定
        prefetch_factor=2,
        persistent_workers=False
    )

    print("正在加载毛细玻璃管验证集（用于监测流速预测准确度）...")
    val_dataset = CelexBloodFlowDataset(data_config=capillary_val_config, mask_path=MASK_PATH)
    val_loader = DataLoader(
        val_dataset, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=sequence_sparse_collate,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False
    )

    # ================= 1. 初始化模型并加载预训练权重 =================
    model = SNN_CNN_Hybrid().to(DEVICE)
    # 【注意路径】这里请指向你在 train.py 中跑出来的最优并行架构模型
    pretrained_path = "/data/zm/Moshaboli/new_data/Model/CNNSNN_fenkai.pth"
    if os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        print(f"[*] 成功加载并行架构预训练权重: {pretrained_path}")
    else:
        raise FileNotFoundError("未找到预训练模型，请先完成阶段一的磨砂玻璃训练！")

    # ================= 2. 全参数解冻 (Full-Parameter Fine-tuning) =================
    print("[*] 正在执行全参数解冻，允许网络扭转物理映射...")
    for param in model.parameters():
        param.requires_grad = True

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[*] 参与联合微调的总参数量: {trainable_params}")

    # ================= 3. 差异化学习率配置 =================
    # SNN 已经有极好的时空滤波基础，用极小的步长 (5e-6) 去适应 Washout 规律
    # CNN 空间网络需要面对全新的背景环境，用正常的步长 (5e-5) 快速收敛管壁 α0
    optimizer = torch.optim.Adam([
        {'params': model.snn_enc1.parameters(), 'lr': 1e-6},
        {'params': model.snn_enc2.parameters(), 'lr': 1e-6},
        {'params': model.snn_dec.parameters(), 'lr': 1e-6},
        {'params': model.spatial_cnn.parameters(), 'lr': 5e-6}
    ])

    # 同样配置余弦退火
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-7)

    best_val_loss = float('inf')
    history_phys_loss = []
    history_data_loss = []
    history_val_mse = []

    lambda_phys = 0.5  # SSIM 物理结构约束的权重

    for epoch in range(NUM_EPOCHS):
        # ----------------- 联合微调阶段 -----------------
        model.train()

        train_loss_total = 0.0
        train_data_total = 0.0
        train_phys_total = 0.0

        current_lr_snn = optimizer.param_groups[0]['lr']
        pbar_train = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{NUM_EPOCHS}] Fine-tune (LR:{current_lr_snn:.1e})")

        # 接收 dense_maps
        for batch_idx, (seq_data, y_true, d_values, dense_maps) in enumerate(pbar_train):
            x_seq = []
            for b_coords, b_feats in seq_data:
                sparse_tensor = ME.SparseTensor(features=b_feats.to(DEVICE), coordinates=b_coords.to(DEVICE))
                x_seq.append(sparse_tensor)

            y_true = y_true.to(DEVICE)
            d_values = d_values.to(DEVICE)
            dense_maps = dense_maps.to(DEVICE)

            optimizer.zero_grad()

            # 调用并行双流前向传播
            v_pred_model, alpha_pred, inv_tau_c_pred = model(x_seq, dense_maps, actual_batch_size=len(y_true))

            d_values_expanded = d_values.view(-1, 1, 1, 1)
            v_pred = (d_values_expanded / (alpha_pred + 1e-6)) * inv_tau_c_pred
            y_true_expanded = y_true.view(-1, 1, 1, 1).expand_as(v_pred)

            # 1. 数据损失
            loss_data = F.mse_loss(v_pred, y_true_expanded)

            # 2. 物理结构约束 (SSIM)
            if alpha_pred.shape[0] >= 2:
                alpha_anchor = alpha_pred[0::2]
                alpha_pos = alpha_pred[1::2]
                loss_physics = structural_similarity_loss(alpha_anchor, alpha_pos)
            else:
                loss_physics = torch.tensor(0.0).to(DEVICE)

            # --- 联合微调总损失 ---
            loss = loss_data + lambda_phys * loss_physics

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss_total += loss.item()
            train_data_total += loss_data.item()
            train_phys_total += loss_physics.item()

            pbar_train.set_postfix({'Total': f"{loss.item():.4f}", 'Data': f"{loss_data.item():.4f}",
                                    'Phys_SSIM': f"{loss_physics.item():.4f}"})

            # 清理显存防卡死
            del x_seq, sparse_tensor, v_pred_model, alpha_pred, inv_tau_c_pred, dense_maps, v_pred, y_true_expanded
            if batch_idx % 50 == 0:
                torch.cuda.empty_cache()

        avg_phys_loss = train_phys_total / len(train_loader)
        avg_data_loss = train_data_total / len(train_loader)
        history_phys_loss.append(avg_phys_loss)
        history_data_loss.append(avg_data_loss)

        # ----------------- 监测验证阶段 -----------------
        model.eval()
        val_mse_total = 0.0
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

                loss_mse = F.mse_loss(v_pred, y_true_expanded)
                val_mse_total += loss_mse.item()
                pbar_val.set_postfix({'Val_MSE': f"{loss_mse.item():.4f}"})

                del x_seq, sparse_tensor, v_pred_model, alpha_pred, inv_tau_c_pred, dense_maps, v_pred, y_true_expanded
                if batch_idx % 50 == 0:
                    torch.cuda.empty_cache()

        avg_val_mse = val_mse_total / len(val_loader)
        history_val_mse.append(avg_val_mse)
        print(
            f"--> Epoch {epoch + 1} | Train Data MSE: {avg_data_loss:.4f} | Phys SSIM: {avg_phys_loss:.5f} | Val MSE: {avg_val_mse:.4f}")

        # 更新学习率
        scheduler.step()

        if avg_val_mse < best_val_loss:
            best_val_loss = avg_val_mse
            os.makedirs("/data/zm/Moshaboli/new_data/Model", exist_ok=True)
            torch.save(model.state_dict(), "/data/zm/Moshaboli/new_data/Model/best_capillary_finetuned.pth")
            print(f"[*] 发现更优的血管微调模型，已保存 (Val MSE: {best_val_loss:.4f})")

        # Epoch级深度清理
        gc.collect()
        torch.cuda.empty_cache()

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
    ax2.set_ylabel('Physics SSIM Loss', color='green', fontsize=12)
    ax2.plot(range(1, NUM_EPOCHS + 1), history_phys_loss, label='Physics SSIM Loss', marker='^', color='green')
    ax2.tick_params(axis='y', labelcolor='green')
    ax2.legend(loc='upper right', fontsize=10)

    plt.title('Physics-Informed Fine-tuning on Capillary Data', fontsize=16)
    fig.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.7)

    os.makedirs("/data/zm/Moshaboli/new_data/Loss_curve", exist_ok=True)
    plot_path = "/data/zm/Moshaboli/new_data/Loss_curve/adjust_curve.png"
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"微调曲线已成功保存至: {plot_path}")


if __name__ == '__main__':
    adjust_and_evaluate()
