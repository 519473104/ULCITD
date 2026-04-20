"""
PruningDiagnostics — 模型效率维度诊断
======================================
专注于验证权重掩码剪枝的效率收益，每个增量任务自动采集：

  - 总参数量 / 有效非零参数量 / 累积稀疏率
  - 逐层稀疏率分布
  - 单任务训练耗时 (wall-clock)
  - 推理延迟 (ms/batch)
  - FLOPs 估算（理论计算量，零权重不贡献有效 FLOPs）

最终任务完成后自动生成：
  - task_efficiency.csv           汇总表
  - layer_sparsity.csv            逐层稀疏明细
  - efficiency_overview.png/svg   四合一效率总图
  - layer_sparsity_heatmap.png    逐层稀疏热力图
  - params_breakdown.png          参数量堆叠柱状图

使用方式（在 foster_uda_compressed.py 中已自动集成）：
  self.diagnostics = PruningDiagnostics(args)
  self.diagnostics.start_training_timer()
  ...训练...
  train_time = self.diagnostics.stop_training_timer()
  self.diagnostics.record_task(...)
  self.diagnostics.generate_all_reports()  # 最后一个任务结束后
"""

import os
import time
import logging
import numpy as np
import pandas as pd

import torch
from torch import nn

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PruningDiagnostics:

    def __init__(self, args, save_dir="./diagnostics"):
        self.args = args
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.task_records = []
        self.layer_sparsity_records = []
        self._train_start_time = None

    # =================================================================
    # 计时
    # =================================================================

    def start_training_timer(self):
        self._train_start_time = time.time()

    def stop_training_timer(self):
        if self._train_start_time is None:
            return 0.0
        elapsed = time.time() - self._train_start_time
        self._train_start_time = None
        return elapsed

    # =================================================================
    # 推理延迟
    # =================================================================

    @staticmethod
    def measure_inference_latency(network, device, input_shape=(1, 3, 32, 32),
                                  warmup=10, repeats=50):
        network.eval()
        dummy = torch.randn(*input_shape).to(device)

        with torch.no_grad():
            for _ in range(warmup):
                network(dummy)

        if device.type == 'cuda':
            torch.cuda.synchronize()

        times = []
        with torch.no_grad():
            for _ in range(repeats):
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                t0 = time.time()
                network(dummy)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                times.append((time.time() - t0) * 1000)

        return float(np.mean(times)), float(np.std(times))

    # =================================================================
    # 参数统计
    # =================================================================

    @staticmethod
    def compute_param_stats(network):
        total = 0
        nonzero = 0
        layer_details = {}

        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                w = module.weight.data
                n = w.numel()
                nz = (w != 0).sum().item()
                total += n
                nonzero += nz
                layer_details[name] = {
                    'shape': list(w.shape),
                    'total': n,
                    'nonzero': nz,
                    'sparsity': 1 - nz / n if n > 0 else 0
                }

        sparsity = 1 - nonzero / total if total > 0 else 0
        return {
            'total_params': total,
            'nonzero_params': nonzero,
            'sparsity': sparsity,
            'layer_details': layer_details
        }

    # =================================================================
    # FLOPs 估算（只计非零权重的有效 FLOPs）
    # =================================================================

    @staticmethod
    def estimate_flops(network, input_shape=(1, 3, 32, 32)):
        """
        粗略估算 Conv2d 和 Linear 的 FLOPs
        total_flops: 理论 FLOPs（含零权重）
        effective_flops: 有效 FLOPs（排除零权重）
        """
        total_flops = 0
        effective_flops = 0

        hooks = []
        flops_dict = {}

        def make_hook(name):
            def hook_fn(module, inp, out):
                if isinstance(module, nn.Conv2d):
                    # FLOPs = 2 * K*K*C_in * C_out * H_out * W_out
                    batch, c_out, h_out, w_out = out.shape
                    k = module.kernel_size[0] * module.kernel_size[1]
                    c_in = module.in_channels // module.groups
                    layer_flops = 2 * k * c_in * c_out * h_out * w_out

                    nz_ratio = (module.weight.data != 0).float().mean().item()
                    flops_dict[name] = {
                        'total': layer_flops,
                        'effective': int(layer_flops * nz_ratio)
                    }
                elif isinstance(module, nn.Linear):
                    layer_flops = 2 * module.in_features * module.out_features
                    nz_ratio = (module.weight.data != 0).float().mean().item()
                    flops_dict[name] = {
                        'total': layer_flops,
                        'effective': int(layer_flops * nz_ratio)
                    }
            return hook_fn

        for name, module in network.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                hooks.append(module.register_forward_hook(make_hook(name)))

        device = next(network.parameters()).device
        dummy = torch.randn(*input_shape).to(device)
        network.eval()
        with torch.no_grad():
            network(dummy)

        for h in hooks:
            h.remove()

        for name, info in flops_dict.items():
            total_flops += info['total']
            effective_flops += info['effective']

        return total_flops, effective_flops, flops_dict

    # =================================================================
    # 核心记录接口
    # =================================================================

    def record_task(self, task_id, network, test_loader, device,
                    total_classes, train_time_sec, test_accuracy):
        logging.info(f"[Diagnostics] Recording efficiency for task {task_id}...")

        param_stats = self.compute_param_stats(network)

        # 推理延迟
        try:
            sample = next(iter(test_loader))
            input_shape = (1,) + tuple(sample[1][0].shape)
            lat_mean, lat_std = self.measure_inference_latency(
                network, device, input_shape=input_shape
            )
        except Exception as e:
            logging.warning(f"[Diagnostics] Latency failed: {e}")
            lat_mean, lat_std = 0.0, 0.0

        # FLOPs
        try:
            total_flops, eff_flops, _ = self.estimate_flops(network)
            flops_reduction = 1 - eff_flops / total_flops if total_flops > 0 else 0
        except Exception as e:
            logging.warning(f"[Diagnostics] FLOPs estimation failed: {e}")
            total_flops, eff_flops, flops_reduction = 0, 0, 0

        record = {
            'task_id': task_id,
            'total_classes': total_classes,
            'test_accuracy': test_accuracy,
            'total_params': param_stats['total_params'],
            'nonzero_params': param_stats['nonzero_params'],
            'pruned_params': param_stats['total_params'] - param_stats['nonzero_params'],
            'sparsity_pct': param_stats['sparsity'] * 100,
            'train_time_sec': train_time_sec,
            'latency_mean_ms': lat_mean,
            'latency_std_ms': lat_std,
            'total_flops': total_flops,
            'effective_flops': eff_flops,
            'flops_reduction_pct': flops_reduction * 100,
        }
        self.task_records.append(record)

        # 逐层稀疏
        self.layer_sparsity_records.append({
            'task_id': task_id,
            'layers': param_stats['layer_details']
        })

        logging.info(
            f"[Diagnostics] Task {task_id}: "
            f"acc={test_accuracy:.2f}%, "
            f"sparsity={param_stats['sparsity']:.2%}, "
            f"nonzero={param_stats['nonzero_params']:,}/{param_stats['total_params']:,}, "
            f"train={train_time_sec:.1f}s, latency={lat_mean:.2f}ms, "
            f"FLOPs reduction={flops_reduction:.2%}"
        )

    # =================================================================
    # 报告生成
    # =================================================================

    def generate_all_reports(self):
        if not self.task_records:
            logging.warning("[Diagnostics] No records")
            return

        logging.info(f"[Diagnostics] Generating efficiency reports -> {self.save_dir}")

        self._save_csvs()
        self._plot_efficiency_overview()
        self._plot_params_breakdown()
        self._plot_layer_sparsity_heatmap()

        logging.info(f"[Diagnostics] All reports saved to: {self.save_dir}")

    def _save_csvs(self):
        df = pd.DataFrame(self.task_records)
        df.to_csv(os.path.join(self.save_dir, 'task_efficiency.csv'), index=False)

        rows = []
        for rec in self.layer_sparsity_records:
            for layer_name, info in rec['layers'].items():
                rows.append({
                    'task_id': rec['task_id'],
                    'layer': layer_name,
                    'shape': str(info['shape']),
                    'total': info['total'],
                    'nonzero': info['nonzero'],
                    'sparsity_pct': info['sparsity'] * 100
                })
        if rows:
            pd.DataFrame(rows).to_csv(
                os.path.join(self.save_dir, 'layer_sparsity.csv'), index=False
            )

    # =================================================================
    # 四合一效率总图
    # =================================================================

    def _plot_efficiency_overview(self):
        df = pd.DataFrame(self.task_records)
        tasks = df['task_id'].values

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (1) 精度 vs 稀疏率 — 双Y轴
        ax1 = axes[0, 0]
        c_acc, c_sp = '#1565C0', '#E65100'
        ax1.plot(tasks, df['test_accuracy'], 'o-', color=c_acc, lw=2, ms=8, label='Accuracy')
        ax1.set_ylabel('Test Accuracy (%)', color=c_acc, fontsize=12)
        ax1.set_ylim([max(0, df['test_accuracy'].min() - 10), 105])
        ax1.tick_params(axis='y', labelcolor=c_acc)

        ax1b = ax1.twinx()
        ax1b.plot(tasks, df['sparsity_pct'], 's--', color=c_sp, lw=2, ms=8, label='Sparsity')
        ax1b.set_ylabel('Sparsity (%)', color=c_sp, fontsize=12)
        ax1b.set_ylim([0, max(df['sparsity_pct'].max() + 10, 50)])
        ax1b.tick_params(axis='y', labelcolor=c_sp)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1b.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=10)
        ax1.set_xlabel('Task')
        ax1.set_title('Accuracy vs Sparsity', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.2)

        # (2) 训练耗时
        ax2 = axes[0, 1]
        ax2.bar(tasks, df['train_time_sec'], color='#43A047', alpha=0.85, edgecolor='k', lw=0.5)
        for i, v in enumerate(df['train_time_sec']):
            ax2.text(tasks[i], v + 0.5, f'{v:.1f}s', ha='center', fontsize=9)
        ax2.set_xlabel('Task')
        ax2.set_ylabel('Training Time (sec)')
        ax2.set_title('Training Time per Task', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.2)

        # (3) 推理延迟
        ax3 = axes[1, 0]
        ax3.errorbar(tasks, df['latency_mean_ms'], yerr=df['latency_std_ms'],
                      fmt='D-', color='#7B1FA2', lw=2, ms=8, capsize=5)
        ax3.set_xlabel('Task')
        ax3.set_ylabel('Inference Latency (ms)')
        ax3.set_title('Inference Latency per Task', fontsize=13, fontweight='bold')
        ax3.grid(True, alpha=0.2)

        # (4) FLOPs 对比
        ax4 = axes[1, 1]
        x = np.arange(len(tasks))
        width = 0.35
        ax4.bar(x - width / 2, df['total_flops'] / 1e6, width,
                label='Total FLOPs', color='#BDBDBD', edgecolor='k', lw=0.5)
        ax4.bar(x + width / 2, df['effective_flops'] / 1e6, width,
                label='Effective FLOPs', color='#1E88E5', edgecolor='k', lw=0.5)
        # 标注 reduction
        for i in range(len(tasks)):
            red = df['flops_reduction_pct'].values[i]
            if red > 0:
                ax4.text(x[i], df['total_flops'].values[i] / 1e6 + 0.5,
                         f'-{red:.1f}%', ha='center', fontsize=9, color='red', fontweight='bold')
        ax4.set_xlabel('Task')
        ax4.set_ylabel('MFLOPs')
        ax4.set_xticks(x)
        ax4.set_xticklabels([str(t) for t in tasks])
        ax4.set_title('Theoretical vs Effective FLOPs', fontsize=13, fontweight='bold')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.2)

        fig.suptitle('Model Efficiency Diagnostics — Weight Mask Pruning',
                     fontsize=15, fontweight='bold', y=1.01)
        fig.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'efficiency_overview.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_dir, 'efficiency_overview.svg'),
                    format='svg', bbox_inches='tight')
        plt.close()

    # =================================================================
    # 参数量堆叠柱状图
    # =================================================================

    def _plot_params_breakdown(self):
        df = pd.DataFrame(self.task_records)
        tasks = df['task_id'].values

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.bar(tasks, df['nonzero_params'], color='#2196F3', alpha=0.9,
               label='Non-zero (effective)', edgecolor='k', lw=0.5)
        ax.bar(tasks, df['pruned_params'], bottom=df['nonzero_params'],
               color='#E0E0E0', alpha=0.7, label='Pruned (zeros)', edgecolor='k', lw=0.5)

        # 标注稀疏率
        for i in range(len(tasks)):
            total = df['total_params'].values[i]
            sp = df['sparsity_pct'].values[i]
            ax.text(tasks[i], total + total * 0.01,
                    f'{sp:.1f}%', ha='center', fontsize=10, fontweight='bold', color='red')

        ax.set_xlabel('Task', fontsize=13)
        ax.set_ylabel('Parameter Count', fontsize=13)
        ax.set_title('Parameter Breakdown per Task\n'
                     '(red = cumulative sparsity)', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.2, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'params_breakdown.png'), dpi=300)
        plt.savefig(os.path.join(self.save_dir, 'params_breakdown.svg'), format='svg')
        plt.close()

    # =================================================================
    # 逐层稀疏热力图
    # =================================================================

    def _plot_layer_sparsity_heatmap(self):
        if not self.layer_sparsity_records:
            return

        # 只取 Conv2d 层
        all_layers = []
        for rec in self.layer_sparsity_records:
            for name, info in rec['layers'].items():
                if len(info['shape']) == 4 and name not in all_layers:
                    all_layers.append(name)

        if not all_layers:
            return

        task_ids = [rec['task_id'] for rec in self.layer_sparsity_records]
        matrix = np.zeros((len(task_ids), len(all_layers)))

        for i, rec in enumerate(self.layer_sparsity_records):
            for j, layer_name in enumerate(all_layers):
                if layer_name in rec['layers']:
                    matrix[i, j] = rec['layers'][layer_name]['sparsity'] * 100

        fig, ax = plt.subplots(figsize=(max(10, len(all_layers) * 1.2),
                                         max(4, len(task_ids) * 0.8)))

        try:
            import seaborn as sns
            sns.heatmap(matrix, ax=ax, cmap='YlOrRd', vmin=0,
                        xticklabels=[n.replace('.', '\n') for n in all_layers],
                        yticklabels=[f'Task {t}' for t in task_ids],
                        annot=True, fmt='.1f',
                        cbar_kws={'label': 'Sparsity (%)'})
        except ImportError:
            im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0)
            ax.set_xticks(range(len(all_layers)))
            ax.set_xticklabels(all_layers, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(len(task_ids)))
            ax.set_yticklabels([f'Task {t}' for t in task_ids])
            for i in range(len(task_ids)):
                for j in range(len(all_layers)):
                    ax.text(j, i, f'{matrix[i, j]:.1f}', ha='center', va='center', fontsize=8)
            plt.colorbar(im, ax=ax, label='Sparsity (%)')

        ax.set_title('Layer-wise Sparsity Distribution (%)\n'
                      'Global pruning: shallow layers preserved, deep layers pruned more',
                      fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'layer_sparsity_heatmap.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.save_dir, 'layer_sparsity_heatmap.svg'),
                    format='svg', bbox_inches='tight')
        plt.close()
