#!/bin/bash

# ========================================================================
# 模态丢弃（Modality Dropout）训练脚本
# ========================================================================
#
# 使用方法：
#   1. 选择下面的某个训练配置（取消注释）
#   2. 运行：bash train_with_modality_dropout.sh
#
# 注意：
#   - 所有命令默认被注释，防止误运行
#   - 根据需求选择一个配置，取消注释后运行
# ========================================================================

set -e  # 遇到错误立即退出

# 配置变量
DATASET="cao/so101_stack_green_on_bottom_v6"
BATCH_SIZE=16
STEPS=15000

# ========================================================================
# 选项1: 原始训练（Baseline，无dropout）
# ========================================================================
# 用途：建立对比基线
# 预期：与修改前完全一致
#
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/baseline_no_dropout \
#   --wandb.project=SO101_modality_dropout \
#   --wandb.run_name=baseline

# ========================================================================
# 选项2: 模态丢弃 p=0.3（推荐配置）
# ========================================================================
# 用途：标准实验配置
# 预期：视觉注意力提升到40-50%，成功率+8-15%
#
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.3 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/modality_dropout_p03 \
#   --wandb.project=SO101_modality_dropout \
#   --wandb.run_name=dropout_p0.3

# ========================================================================
# 选项3: 保守测试 p=0.2
# ========================================================================
# 用途：第一次尝试或任务需要较多状态反馈
# 预期：温和提升
#
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.2 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/modality_dropout_p02 \
#   --wandb.project=SO101_modality_dropout \
#   --wandb.run_name=dropout_p0.2

# ========================================================================
# 选项4: 激进测试 p=0.4
# ========================================================================
# 用途：p=0.3效果不够，或任务纯视觉主导
# 预期：最大化视觉依赖
#
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.4 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/modality_dropout_p04 \
#   --wandb.project=SO101_modality_dropout \
#   --wandb.run_name=dropout_p0.4

# ========================================================================
# 选项5: 快速验证（100 steps）
# ========================================================================
# 用途：验证代码正确性，无崩溃
# 时间：约10-15分钟
#
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.3 \
#   --batch_size=8 \
#   --steps=100 \
#   --output_dir=/tmp/test_dropout_quick \
#   --wandb.disabled=true

# ========================================================================
# 选项6: 并行运行所有实验（需要多GPU或后台运行）
# ========================================================================
# 用途：同时对比所有配置
# 时间：1-2天（如果有足够GPU）
#
# 注意：会同时启动4个训练进程，确保资源充足！
#
# echo "启动并行训练..."
#
# # Baseline
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/baseline \
#   --wandb.run_name=baseline &
# PID_BASELINE=$!
#
# # p=0.2
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.2 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/dropout_p02 \
#   --wandb.run_name=dropout_p0.2 &
# PID_P02=$!
#
# # p=0.3（推荐）
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.3 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/dropout_p03 \
#   --wandb.run_name=dropout_p0.3 &
# PID_P03=$!
#
# # p=0.4
# lerobot-train \
#   --dataset.repo_id="$DATASET" \
#   --policy.type=act \
#   --policy.use_modality_dropout=true \
#   --policy.modality_dropout_prob=0.4 \
#   --batch_size=$BATCH_SIZE \
#   --steps=$STEPS \
#   --output_dir=outputs/dropout_p04 \
#   --wandb.run_name=dropout_p0.4 &
# PID_P04=$!
#
# echo "所有训练已启动！"
# echo "  Baseline: PID $PID_BASELINE"
# echo "  Dropout p=0.2: PID $PID_P02"
# echo "  Dropout p=0.3: PID $PID_P03"
# echo "  Dropout p=0.4: PID $PID_P04"
# echo ""
# echo "等待所有训练完成..."
# wait $PID_BASELINE $PID_P02 $PID_P03 $PID_P04
# echo "✅ 所有训练完成！"

# ========================================================================
# 选项7: 从checkpoint恢复训练
# ========================================================================
# 用途：中断后继续训练，或改变配置继续训练
#
# lerobot-train \
#   --resume_from=outputs/modality_dropout_p03/checkpoint-5000 \
#   --steps=20000

# ========================================================================
# 选项8: 关闭dropout继续训练（回退测试）
# ========================================================================
# 用途：测试权重兼容性，或发现dropout效果不好时回退
#
# lerobot-train \
#   --resume_from=outputs/modality_dropout_p03/checkpoint-10000 \
#   --policy.use_modality_dropout=false \
#   --steps=20000 \
#   --output_dir=outputs/dropout_p03_then_disabled

echo ""
echo "========================================================================"
echo "提示：所有训练命令已准备好，但默认被注释以防止误运行"
echo "请编辑此脚本，取消注释你想要的配置，然后运行："
echo "  bash train_with_modality_dropout.sh"
echo "========================================================================"
echo ""
