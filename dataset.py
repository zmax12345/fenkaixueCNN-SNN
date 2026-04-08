import os
import glob
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import MinkowskiEngine as ME
import random


class CelexBloodFlowDataset(Dataset):
    def __init__(self, data_config, mask_path="hot_pixel_mask.npy", T=150, seq_len=70, dt_us=20):
        self.data_config = data_config
        self.T = T
        self.seq_len = seq_len
        self.dt = dt_us
        self.hot_mask = np.load(mask_path) if os.path.exists(mask_path) else np.zeros((800, 1280), dtype=bool)

        # 核心改变：内存中只存经过 ROI 过滤后的精简 DataFrame，不存 Tensor
        self.file_dataframes = {}
        self.samples_metadata = []  # 存储样本的索引元数据

        self._build_dataset_lightweight()

        # 构建环境索引字典
        self.env_to_indices = {}
        for idx, meta in enumerate(self.samples_metadata):
            env_key = meta['env_key']
            if env_key not in self.env_to_indices:
                self.env_to_indices[env_key] = []
            self.env_to_indices[env_key].append(idx)

    def _build_dataset_lightweight(self):
        """轻量化构建：只扫描文件并记录时间窗口指针"""
        for path, d_val in self.data_config.items():
            if os.path.isdir(path):
                csv_files = glob.glob(os.path.join(path, "*_clip.csv"))
            elif os.path.isfile(path) and path.endswith('.csv'):
                csv_files = [path]
            else:
                continue

            # 使用文件夹路径作为 env_key，而不是具体文件路径
            env_key = os.path.dirname(path) if os.path.isfile(path) else path
            for file in csv_files:
                filename = os.path.basename(file)
                try:
                    v_true = float(filename.split('mm')[0])
                except:
                    continue

                try:
                    # 读取并进行初步过滤，减少内存占用
                    df = pd.read_csv(file, header=None, names=['row', 'col', 't_in', 't_off'],
                                     dtype={'row': np.int32, 'col': np.int32, 't_in': np.int64, 't_off': np.int64},
                                     on_bad_lines='skip')
                except:
                    continue

                if df.empty: continue
                # 预处理：ROI 过滤与坐标平移
                df = df[(df['row'] >= 400) & (df['row'] <= 499) & (df['col'] >= 700) & (df['col'] <= 1067)].copy()
                if df.empty: continue

                # 坏点过滤
                valid_events = ~self.hot_mask[df['row'].values, df['col'].values]
                df = df[valid_events].copy()
                if df.empty: continue

                df['row'] -= 400
                df['col'] -= 700

                # 时间量化
                t_start = df['t_in'].min()
                df['t_bin'] = (df['t_in'] - t_start) // self.dt

                # 存储精简后的 DF
                file_id = file
                self.file_dataframes[file_id] = df

                max_bin = df['t_bin'].max()
                total_frames = int(max_bin // self.T) + 1

                # 只记录元数据指针
                for seq_start_idx in range(total_frames - self.seq_len + 1):
                    self.samples_metadata.append({
                        'file_id': file_id,
                        'start_bin': seq_start_idx * self.T,
                        'end_bin': (seq_start_idx + self.seq_len) * self.T,
                        'v_true': v_true,
                        'd_val': d_val,
                        'env_key': env_key
                    })

    def _process_single_sample(self, index):
        """在运行时处理单个样本的数据生成"""
        meta = self.samples_metadata[index]
        df = self.file_dataframes[meta['file_id']]

        # 提取当前时间窗口的数据
        seq_df = df[(df['t_bin'] >= meta['start_bin']) & (df['t_bin'] < meta['end_bin'])]

        # --- 1. 生成 CNN 用的稠密归一化图 ---
        dense_map = np.zeros((100, 368), dtype=np.float32)
        if not seq_df.empty:
            np.add.at(dense_map, (seq_df['row'].values, seq_df['col'].values), 1.0)
            max_count = dense_map.max()
            if max_count > 0:
                dense_map /= max_count
        dense_map_tensor = torch.from_numpy(dense_map).unsqueeze(0)

        # --- 2. 生成 SNN 用的稀疏张量序列 ---
        sequence_data = []
        for f_idx in range(self.seq_len):
            frame_start = meta['start_bin'] + f_idx * self.T
            frame_df = seq_df[(seq_df['t_bin'] >= frame_start) & (seq_df['t_bin'] < frame_start + self.T)]

            if frame_df.empty:
                coords = torch.empty((0, 3), dtype=torch.int32)
                feats = torch.empty((0, 1), dtype=torch.float32)
            else:
                locations = torch.IntTensor(np.column_stack((frame_df['row'].values, frame_df['col'].values)))
                features = torch.ones((len(frame_df), 1), dtype=torch.float32)
                coords, feats = ME.utils.sparse_quantize(
                    coordinates=locations, features=features, quantization_size=[1, 1]
                )
            sequence_data.append((coords, feats))

        return (sequence_data, meta['v_true'], meta['d_val'], dense_map_tensor, meta['env_key'])

    def __len__(self):
        return len(self.samples_metadata)

    def __getitem__(self, index):
        # 锚点样本
        anchor_sample = self._process_single_sample(index)

        # 成对采样逻辑
        env_key = anchor_sample[4]
        pos_idx = random.choice(self.env_to_indices[env_key])
        pos_sample = self._process_single_sample(pos_idx)

        return anchor_sample, pos_sample


def sequence_sparse_collate(batch):
    # 逻辑保持不变，适配成对采样
    flat_batch = []
    for anchor, pos in batch:
        flat_batch.append(anchor)
        flat_batch.append(pos)

    seq_len = len(flat_batch[0][0])
    batched_seq_data = []

    for t in range(seq_len):
        coords_t = [sample[0][t][0] for sample in flat_batch]
        feats_t = [sample[0][t][1] for sample in flat_batch]
        b_coords, b_feats = ME.utils.sparse_collate(coords_t, feats_t)
        batched_seq_data.append((b_coords, b_feats))

    labels = torch.tensor([sample[1] for sample in flat_batch], dtype=torch.float32)
    d_values = torch.tensor([sample[2] for sample in flat_batch], dtype=torch.float32)
    dense_maps = torch.stack([sample[3] for sample in flat_batch], dim=0)

    return batched_seq_data, labels, d_values, dense_maps
