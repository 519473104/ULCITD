"""
FOSTER_UDA with Incremental Pruning & Model Compression
========================================================
"""

import logging
import copy
import numpy as np
from tqdm import tqdm

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.base import BaseLearner
from utils.inc_net import FOSTERNet
from utils.toolkit import tensor2numpy
import torch.nn.utils.prune as torch_prune

import os
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import pandas as pd

from models.pruning_diagnostics import PruningDiagnostics

# =====================================================================
# 主类
# =====================================================================

class FOSTER_UDA(BaseLearner):
    """FOSTER with UDA + Safe Pruning & Compression"""

    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self._network = FOSTERNet(args, False)
        self._snet = None
        self.is_teacher_wa = args["is_teacher_wa"]
        self.lambda_okd = args["lambda_okd"]
        self.lambda_fkd = args.get("lambda_fkd", 1)
        self.wa_value = args["wa_value"]
        self.oofc = args["oofc"].lower()

        # ============ UDA 参数 ============
        self.use_uda = args.get("use_uda", True)
        self.lambda_con = args.get("lambda_con", 0.1)
        self._old_network = None

        # ============ 可视化参数 ============
        self.enable_visualization = args.get("enable_visualization", False)
        self.vis_save_dir = args.get("vis_save_dir", "./visualizations")

        # ============ 剪枝 & 压缩参数 ============
        self.use_pruning = args.get("use_pruning", True)
        # "global" = 全局幅值剪枝(推荐), "layerwise" = 逐层幅值剪枝, "manual" = 手动置零
        self.pruning_method = args.get("pruning_method", "global")
        self.prune_ratio = args.get("prune_ratio", 0.3)
        self.prune_ratio_decay = args.get("prune_ratio_decay", 0.9)
        self.min_prune_ratio = args.get("min_prune_ratio", 0.1)

        self._sparse_trainer = None
        self._compression_stats = []

        # ============ 剪枝诊断工具 ============
        diag_dir = args.get("diagnostics_dir", "./diagnostics")
        self.diagnostics = PruningDiagnostics(args, save_dir=diag_dir)

    # =====================================================================
    # 安全的剪枝方法（不改变网络结构）
    # =====================================================================

    def _incremental_prune(self, convnet, prune_ratio):
        """
        对指定 convnet 执行权重掩码剪枝（累积式）
        核心逻辑：
          - 根据当前已有稀疏率 + 本轮 prune_ratio，计算目标稀疏率
          - 只对当前仍非零的权重排序，裁掉其中最小的一批
          - 每轮增量任务都能真正裁掉新的权重，稀疏率逐任务递增
        """
        if prune_ratio <= 0:
            return

        total_params = sum(p.numel() for p in convnet.parameters())
        current_nonzero = sum(
            (p.data != 0).sum().item() for p in convnet.parameters()
        )
        current_sparsity = 1 - current_nonzero / total_params if total_params > 0 else 0

        # 目标：在当前非零权重中再裁掉 prune_ratio 比例
        # 即 new_nonzero = current_nonzero * (1 - prune_ratio)
        target_nonzero = int(current_nonzero * (1 - prune_ratio))
        num_to_prune = current_nonzero - target_nonzero
        target_sparsity = 1 - target_nonzero / total_params if total_params > 0 else 0

        logging.info(f"[Pruning] Method={self.pruning_method}, "
                     f"current sparsity={current_sparsity:.2%}, "
                     f"target sparsity={target_sparsity:.2%}")
        logging.info(f"[Pruning] Will prune {num_to_prune:,} weights "
                     f"from {current_nonzero:,} non-zero")

        if num_to_prune <= 0:
            logging.info("[Pruning] Nothing to prune, skipping")
            return

        # 收集所有目标层的非零权重绝对值，找全局阈值
        nonzero_abs_values = []
        target_modules = []
        for name, module in convnet.named_modules():
            if isinstance(module, nn.Conv2d):
                target_modules.append((name, module))
                w = module.weight.data
                nonzero_vals = w[w != 0].abs()
                if len(nonzero_vals) > 0:
                    nonzero_abs_values.append(nonzero_vals.view(-1))

        if len(nonzero_abs_values) == 0:
            logging.info("[Pruning] No non-zero weights found, skipping")
            return

        all_nonzero = torch.cat(nonzero_abs_values)

        # 安全检查：不能裁掉超过实际非零数量
        num_to_prune = min(num_to_prune, len(all_nonzero) - 1)
        if num_to_prune <= 0:
            return

        # 找到第 num_to_prune 小的非零权重作为阈值
        threshold = all_nonzero.kthvalue(num_to_prune).values

        # 应用掩码：将非零权重中低于阈值的置零
        total_pruned = 0
        with torch.no_grad():
            for name, module in target_modules:
                w = module.weight.data
                # 只裁非零且小于阈值的
                prune_mask = (w.abs() <= threshold) & (w != 0)
                module.weight.data[prune_mask] = 0.0
                total_pruned += prune_mask.sum().item()

        # 剪枝后统计
        after_nonzero = sum(
            (p.data != 0).sum().item() for p in convnet.parameters()
        )
        after_sparsity = 1 - after_nonzero / total_params if total_params > 0 else 0

        logging.info(f"[Pruning] Result: {current_nonzero:,} -> {after_nonzero:,} non-zero "
                     f"(pruned {total_pruned:,}), sparsity={after_sparsity:.2%}")

        # 逐层详情
        for name, module in target_modules:
            w = module.weight.data
            layer_sp = (w == 0).sum().item() / w.numel()
            logging.info(f"  {name}: shape={list(w.shape)}, sparsity={layer_sp:.2%}")

        self._compression_stats.append({
            'task': self._cur_task,
            'stage': 'weight_mask_pruning',
            'method': self.pruning_method,
            'total_params': total_params,
            'nonzero_before': current_nonzero,
            'nonzero_after': after_nonzero,
            'actually_pruned': total_pruned,
            'sparsity': after_sparsity
        })

    # =====================================================================
    # 增量训练主函数
    # =====================================================================

    def incremental_train(self, data_manager):
        self.data_manager = data_manager
        self._cur_task += 1

        if self._cur_task > 0:
            self._old_network = copy.deepcopy(self._snet)
            self._old_network.eval()
            for p in self._old_network.parameters():
                p.requires_grad = False

        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        self._network.update_fc(self._total_classes)
        self._network_module_ptr = self._network

        logging.info(f'Task {self._cur_task}: known={self._known_classes}, '
                     f'new={data_manager.get_task_size(self._cur_task)}, '
                     f'total={self._total_classes}')

        self._network.to(self._device)

        # ====== 增量任务时的剪枝 ======
        if self._cur_task > 0:
            # 冻结所有旧 convnet 和旧分类头
            for i in range(len(self._network.convnets) - 1):  # 最后一个是新网络，不冻结
                for p in self._network.convnets[i].parameters():
                    p.requires_grad = False
            for p in self._network.oldfc.parameters():
                p.requires_grad = False

            if self.use_pruning:
                # 递减剪枝率
                current_prune_ratio = max(
                    self.min_prune_ratio,
                    self.prune_ratio * (self.prune_ratio_decay ** (self._cur_task - 1))
                )
                # 对所有已冻结的旧 convnet 执行剪枝
                for i in range(len(self._network.convnets) - 1):
                    logging.info(f"[Pruning] Task {self._cur_task}, convnets[{i}]: "
                                 f"ratio={current_prune_ratio:.3f}")
                    self._incremental_prune(self._network.convnets[i], current_prune_ratio)

        self._prepare_data_loaders(data_manager)

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        # ====== 开始计时 ======
        self.diagnostics.start_training_timer()

        self._train(self.train_loader, self.test_loader)

        # ====== 停止计时 ======
        train_time = self.diagnostics.stop_training_timer()

        self.build_rehearsal_memory(data_manager, self.samples_per_class)

        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

        self._snet = copy.deepcopy(self._network)
        self._known_classes = self._total_classes

        # ====== 诊断：采集本任务效率指标 ======
        test_acc = self._compute_accuracy(self._network, self.test_loader)
        self.diagnostics.record_task(
            task_id=self._cur_task,
            network=self._snet,
            test_loader=self.test_loader,
            device=self._device,
            total_classes=self._total_classes,
            train_time_sec=train_time,
            test_accuracy=test_acc
        )

        self._log_compression_stats()
        logging.info(f'Task {self._cur_task} completed. known_classes={self._known_classes}')

        # 可视化 + 诊断报告
        total_tasks = getattr(data_manager, 'nb_tasks', 5)
        if self._cur_task == total_tasks - 1:
            # 生成剪枝诊断全套报告
            self.diagnostics.generate_all_reports()

            if self.enable_visualization:
                print('\n' + '*' * 60)
                print(f'Final task {self._cur_task} completed!')
                print('Generating visualizations...')
                print('*' * 60)
                self.visualize_target_domain_tsne()
                self.generate_confusion_matrix()

    def _log_compression_stats(self):
        if not self._compression_stats:
            return
        current_params = sum(p.numel() for p in self._network.parameters())
        current_nonzero = sum(
            (p.data != 0).sum().item() for p in self._network.parameters()
        )
        overall_sparsity = 1 - current_nonzero / current_params if current_params > 0 else 0

        logging.info(f"[Stats] Network: {current_params:,} total params, "
                     f"{current_nonzero:,} non-zero, "
                     f"overall sparsity={overall_sparsity:.2%}")

        for stat in self._compression_stats:
            if stat['task'] == self._cur_task:
                logging.info(f"[Stats] {stat}")

    # =====================================================================
    # 数据加载
    # =====================================================================

    def _prepare_data_loaders(self, data_manager):
        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source='train', mode='train', appendent=self._get_memory()
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=self.args["num_workers"], pin_memory=True
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source='test', mode='test'
        )
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"],
            shuffle=True, num_workers=self.args["num_workers"]
        )

        if self.use_uda and self.args.get('lambda_pseudo', 0) > 0:
            if hasattr(data_manager, 'get_unlabeled_dataset'):
                try:
                    unlabeled_dataset = data_manager.get_unlabeled_dataset()
                    if unlabeled_dataset and len(unlabeled_dataset) > 0:
                        self.unlabeled_loader = DataLoader(
                            unlabeled_dataset, batch_size=self.args["batch_size"],
                            shuffle=True, num_workers=self.args["num_workers"], pin_memory=True
                        )
                    else:
                        raise ValueError("Empty unlabeled dataset")
                except:
                    unlabeled_dataset = data_manager.get_dataset(
                        np.arange(0, self._total_classes), source='test', mode='test'
                    )
                    self.unlabeled_loader = DataLoader(
                        unlabeled_dataset, batch_size=self.args["batch_size"],
                        shuffle=True, num_workers=self.args["num_workers"], pin_memory=True
                    )
            else:
                unlabeled_dataset = data_manager.get_dataset(
                    np.arange(0, self._total_classes), source='test', mode='test'
                )
                self.unlabeled_loader = DataLoader(
                    unlabeled_dataset, batch_size=self.args["batch_size"],
                    shuffle=True, num_workers=self.args["num_workers"], pin_memory=True
                )
        else:
            self.unlabeled_loader = None

    # =====================================================================
    # 训练
    # =====================================================================

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if hasattr(self._network, "module"):
            self._network_module_ptr = self._network.module

        if self._cur_task == 0:
            self._unified_train_with_uda(
                train_loader, test_loader,
                lr=self.args["init_lr"], epochs=self.args["init_epochs"],
                weight_decay=self.args["init_weight_decay"]
            )
        else:
            self._unified_train_with_uda(
                train_loader, test_loader,
                lr=self.args["lr"], epochs=self.args["boosting_epochs"],
                weight_decay=self.args["weight_decay"]
            )

    def _unified_train_with_uda(self, train_loader, test_loader, lr, epochs, weight_decay):
        self.per_cls_weights = torch.ones(self._total_classes).to(self._device)

        optimizer = optim.SGD(
            filter(lambda p: p.requires_grad, self._network.parameters()),
            lr=lr, momentum=0.9, weight_decay=weight_decay
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=epochs)

        prog_bar = tqdm(range(epochs))
        unlabeled_iter = iter(self.unlabeled_loader) if self.use_uda and self.unlabeled_loader else None

        for epoch in prog_bar:
            self._network.train()
            losses = {'total': 0., 'clf': 0., 'kd': 0., 'uda': 0.}
            correct, total = 0, 0
            uda_weight = self._get_adaptive_uda_weight(epoch, epochs)

            for _, inputs, targets in train_loader:
                inputs, targets = inputs.to(self._device), targets.to(self._device)
                outputs = self._network(inputs)
                logits, features = outputs["logits"], outputs["features"]
                old_logits = outputs.get("old_logits", None)
                fe_logits = outputs.get("fe_logits", None)
                targets = targets.long()

                loss_clf = F.cross_entropy(logits, targets)
                loss_kd = torch.tensor(0.).to(self._device)

                if old_logits is not None and self._cur_task > 0 and old_logits.size(1) > 0:
                    current_old_classes = min(logits.size(1), old_logits.size(1))
                    if current_old_classes > 0:
                        loss_kd_decision = self._KD_loss(
                            logits[:, :current_old_classes],
                            old_logits[:, :current_old_classes].detach(),
                            self.args["T"]
                        )
                        loss_kd_feature = self._compute_feature_kd_loss(inputs, features)
                        # 双级知识蒸馏
                        loss_kd = self.lambda_okd * loss_kd_decision + self.lambda_fkd * loss_kd_feature
                        # loss_kd = self.lambda_fkd * loss_kd_feature

                loss_fe = torch.tensor(0.).to(self._device)
                if fe_logits is not None:
                    loss_fe = F.cross_entropy(fe_logits, targets)

                loss_uda = torch.tensor(0.).to(self._device)
                if self.use_uda and unlabeled_iter is not None:
                    raw_uda_loss = self._compute_uda_loss(features, logits, targets, unlabeled_iter)
                    loss_uda = uda_weight * raw_uda_loss

                loss = loss_clf + loss_fe + loss_kd + loss_uda
                # loss = loss_clf  + loss_kd + loss_uda
                # loss = loss_clf + loss_fe + loss_uda

                optimizer.zero_grad()
                loss.backward()

                if self._cur_task > 0 and self.oofc == "az":
                    for i, p in enumerate(self._network_module_ptr.fc.parameters()):
                        if i == 0:
                            p.grad.data[self._known_classes:, :self._network_module_ptr.out_dim] = 0.

                optimizer.step()

                # 动态稀疏 step
                if self._sparse_trainer is not None:
                    self._sparse_trainer.step()

                losses['total'] += loss.item()
                losses['clf'] += loss_clf.item()
                losses['kd'] += loss_kd.item()
                losses['uda'] += loss_uda.item()

                _, preds = torch.max(logits, 1)
                correct += preds.eq(targets).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, 2)

            if epoch % 5 == 0:
                test_acc = self._compute_accuracy(self._network, test_loader)
                info = (f'Task {self._cur_task}, Epoch {epoch + 1}/{epochs} => '
                        f'Loss {losses["total"] / len(train_loader):.3f}, '
                        f'UDA {losses["uda"] / len(train_loader):.3f} (w={uda_weight:.2f}), '
                        f'Train {train_acc:.2f}, Test {test_acc:.2f}')
                prog_bar.set_description(info)
                logging.info(info)

        # 动态稀疏结束
        if self._sparse_trainer is not None:
            stats = self._sparse_trainer.get_sparsity_stats()
            logging.info(f"[DynamicSparse] Final sparsity: {stats['overall']:.2%}")
            self._sparse_trainer.apply_masks()
            self._compression_stats.append({
                'task': self._cur_task, 'stage': 'dynamic_sparse',
                'overall_sparsity': stats['overall']
            })

        if self._cur_task > 0 and self.is_teacher_wa:
            self._network_module_ptr.weight_align(
                self._known_classes,
                self._total_classes - self._known_classes,
                self.wa_value
            )

    # =====================================================================
    # UDA 相关
    # =====================================================================

    def _get_adaptive_uda_weight(self, current_epoch, total_epochs):
        warmup_epochs = max(1, int(total_epochs * 0.1))
        return (current_epoch + 1) / warmup_epochs if current_epoch < warmup_epochs else 1.0

    def _compute_uda_loss(self, features, logits, targets, unlabeled_iter):
        if unlabeled_iter is None or not self.use_uda:
            return torch.tensor(0.).to(self._device)

        try:
            _, unlabeled_inputs, _ = next(unlabeled_iter)
        except StopIteration:
            unlabeled_iter = iter(self.unlabeled_loader)
            _, unlabeled_inputs, _ = next(unlabeled_iter)

        unlabeled_inputs = unlabeled_inputs.to(self._device)
        with torch.no_grad():
            self._network(unlabeled_inputs)

        total_uda_loss = torch.tensor(0.).to(self._device)
        if self.lambda_con > 0:
            consistency_loss = self._compute_consistency_loss(unlabeled_inputs)
            total_uda_loss = total_uda_loss + self.lambda_con * consistency_loss

        return total_uda_loss

    def _compute_consistency_loss(self, inputs, T=0.5, tau=0.8):
        with torch.no_grad():
            weak_outputs = self._network(inputs)
            weak_probs = F.softmax(weak_outputs["logits"], dim=1)
            weak_probs = weak_probs ** (1 / T)
            weak_probs = weak_probs / weak_probs.sum(dim=1, keepdim=True)

        noise = torch.randn_like(inputs) * 0.01
        strong_inputs = inputs + noise
        strong_inputs = F.dropout(strong_inputs, p=0.1, training=True)

        strong_outputs = self._network(strong_inputs)
        strong_probs = F.softmax(strong_outputs["logits"], dim=1)

        loss = F.kl_div(strong_probs.log(), weak_probs, reduction="none").sum(1)
        mask = (weak_probs.max(dim=1).values > tau).float()
        return (loss * mask).mean()

    def _KD_loss(self, pred, soft, T):
        min_dim = min(pred.size(1), soft.size(1))
        pred, soft = pred[:, :min_dim], soft[:, :min_dim]
        pred = torch.log_softmax(pred / T, 1)
        soft = torch.softmax(soft / T, 1)
        return -1 * torch.mul(soft, pred).sum() / pred.shape[0]

    def _compute_feature_kd_loss(self, inputs, current_features):
        if self._old_network is None or self._cur_task <= 1:
            return torch.tensor(0.).to(self._device)
        with torch.no_grad():
            old_features = self._old_network(inputs)["features"]
        min_dim = min(current_features.size(-1), old_features.size(-1))
        return F.mse_loss(current_features[..., :min_dim], old_features[..., :min_dim].detach())

    # =====================================================================
    # 可视化
    # =====================================================================

    def _get_seed_tag(self):
        """获取当前实验的 seed 标签，用于文件名索引"""
        seed = self.args.get('seed', None)
        if seed is not None:
            return f"_seed{seed}"
        return ""

    def extract_features(self, data_loader, network=None, max_samples=2000):
        if network is None:
            network = self._snet if self._snet is not None else self._network
        network.eval()
        features_list, labels_list = [], []
        total_samples = 0
        with torch.no_grad():
            for _, inputs, targets in data_loader:
                if total_samples >= max_samples:
                    break
                inputs = inputs.to(self._device)
                outputs = network(inputs)
                features = outputs["features"] if "features" in outputs else outputs["logits"]
                batch_size = min(features.size(0), max_samples - total_samples)
                features_list.append(features[:batch_size].cpu().numpy())
                labels_list.append(targets[:batch_size].numpy())
                total_samples += batch_size
        if not features_list:
            return np.array([]), np.array([])
        return np.concatenate(features_list), np.concatenate(labels_list)

    def visualize_target_domain_tsne(self, save_dir=None):
        print('\n' + '=' * 60)
        print(f'Generating t-SNE for Target Domain (Task {self._cur_task})...')
        print('=' * 60)

        if save_dir is None:
            save_dir = os.path.join(self.vis_save_dir, 'tsne_results')
        os.makedirs(save_dir, exist_ok=True)

        if self.test_loader is None:
            print("[Error] No test data loader!")
            return None, None

        network = self._snet if self._snet is not None else self._network
        target_features, target_labels = self.extract_features(self.test_loader, network=network)

        if len(target_features) == 0:
            print("[Error] No features extracted!")
            return None, None

        num_samples = len(target_features)
        num_classes = len(np.unique(target_labels))
        seed_tag = self._get_seed_tag()
        print(f'Samples: {num_samples}, Classes: {num_classes}, '
              f'Dim: {target_features.shape[1]}, Seed: {self.args.get("seed", "N/A")}')

        perplexity = min(30, num_samples // 4)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000, verbose=1)
        features_2d = tsne.fit_transform(target_features)

        # CSV 文件名包含 seed
        csv_path = os.path.join(save_dir, f'tsne_target_task_{self._cur_task}{seed_tag}.csv')
        pd.DataFrame({
            'tsne_x': features_2d[:, 0], 'tsne_y': features_2d[:, 1],
            'label': target_labels.astype(int),
            'seed': self.args.get('seed', None)
        }).to_csv(csv_path, index=False)

        plt.rcParams['font.family'] = 'Times New Roman'
        fig, ax = plt.subplots(figsize=(10, 8))
        color_map = {i: c for i, c in enumerate(
            ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
             '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
        )}
        for cls in np.unique(target_labels):
            mask = target_labels == cls
            ax.scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=color_map.get(int(cls) % 10, '#000'), s=50, alpha=0.7,
                       edgecolors='k', linewidth=0.3, label=f'Class {int(cls)}')
        ax.set_xlabel('t-SNE Dim 1', fontsize=14)
        ax.set_ylabel('t-SNE Dim 2', fontsize=14)
        seed_display = self.args.get('seed', 'N/A')
        ax.set_title(f't-SNE Target Domain | Task {self._cur_task} | Seed {seed_display}', fontsize=16)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'tsne_target_task_{self._cur_task}{seed_tag}.png'), dpi=300)
        plt.savefig(os.path.join(save_dir, f'tsne_target_task_{self._cur_task}{seed_tag}.svg'), format='svg')
        plt.close()
        plt.rcParams['font.family'] = 'sans-serif'

        return features_2d, target_labels

    def generate_confusion_matrix(self, save_dir=None):
        seed_tag = self._get_seed_tag()
        print(f'\nGenerating Confusion Matrix for Task {self._cur_task} (seed={self.args.get("seed", "N/A")})...')
        if save_dir is None:
            save_dir = os.path.join(self.vis_save_dir, 'confusion_matrix')
        os.makedirs(save_dir, exist_ok=True)

        if self.test_loader is None:
            return None

        network = self._snet if self._snet is not None else self._network
        network.eval()
        all_preds, all_labels = [], []

        with torch.no_grad():
            for _, inputs, targets in self.test_loader:
                inputs = inputs.to(self._device)
                logits = network(inputs)["logits"]
                _, preds = torch.max(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(targets.numpy())

        all_preds, all_labels = np.array(all_preds), np.array(all_labels)
        cm = confusion_matrix(all_labels, all_preds)
        num_classes = len(np.unique(all_labels))

        report = classification_report(all_labels, all_preds, digits=4)
        report_path = os.path.join(save_dir, f'classification_report_task_{self._cur_task}{seed_tag}.txt')
        with open(report_path, 'w') as f:
            f.write(f'Seed: {self.args.get("seed", "N/A")}\n\n')
            f.write(report)

        cm_csv_path = os.path.join(save_dir, f'confusion_matrix_task_{self._cur_task}{seed_tag}.csv')
        pd.DataFrame(cm).to_csv(cm_csv_path)

        fig_size = max(8, num_classes * 0.6)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        seed_display = self.args.get('seed', 'N/A')
        ax.set_title(f'Confusion Matrix Task {self._cur_task} | '
                     f'Acc: {(all_preds == all_labels).mean() * 100:.2f}% | Seed {seed_display}')
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'confusion_matrix_task_{self._cur_task}{seed_tag}.png'), dpi=300)
        plt.close()

        return cm, all_preds, all_labels
