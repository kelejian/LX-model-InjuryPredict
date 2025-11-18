# -*- coding: utf-8 -*-
"""
集中管理模型训练、损失函数和网络结构的可调超参数。
"""

# 1. 优化与训练相关
training_params = {
    "Epochs": 450,
    "Batch_size": 512,
    "Learning_rate": 0.02,
    "Learning_rate_min": 1e-6,
    "weight_decay": 6e-4,
    "Patience": 1000, # 早停轮数
}

# 2. 损失函数相关
loss_params = {
    "base_loss": "mae",
    "weight_factor_classify": 1.1,
    "weight_factor_sample": 0.2,
    "loss_weights": (0.2, 1.0, 20.0), # HIC, Dmax, Nij 各自损失的权重
}

# 3. 模型结构相关
model_params = {
    "Ksize_init": 8,
    "Ksize_mid": 3,
    "num_blocks_of_tcn": 3,
    "tcn_channels_list": [64, 128, 160],  # 每个 TCN 块的输出通道数
    "num_layers_of_mlpE": 3,
    "num_layers_of_mlpD": 3,
    "mlpE_hidden": 192,
    "mlpD_hidden": 160,
    "encoder_output_dim": 128,
    "decoder_output_dim": 96,
    "dropout_MLP": 0.35,
    "dropout_TCN": 0.15,
    "use_channel_attention": True,  # 是否使用通道注意力机制
    "fixed_channel_weight": [0.69, 0.3, 0.01],  # 固定的通道注意力权重(None表示自适应学习)
}

# K-Fold 专项设置
kfold_params = {
    "K": 5 # K-Fold 折数
}