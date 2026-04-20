"""
main.py — 6任务×5次重复实验 自动化运行 + CSV结果汇总

流程：
  外层循环: 5 次重复实验 (不同 seed)
    内层循环: 6 个任务顺序执行 (不同源域/目标域组合)
      每个任务: 完整的增量学习训练

最终输出:
  ./diagnostics/results_summary.csv  — 每行一次实验 + Mean/Std
"""

import json
import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
from trainer import train
from IEEE_path import setup_task, TASK_CONFIGS
# from Tsinghua_path import setup_task, TASK_CONFIGS

def print_memory_usage():
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        cached = torch.cuda.memory_reserved() / (1024 ** 2)
        print(f"Allocated GPU memory: {allocated:.2f} MB")
        print(f"Cached GPU memory: {cached:.2f} MB")
    else:
        print("GPU not available.")


def main():
    num_runs = 5
    seeds = [21, 3, 1998, 1980, 42]
    # seeds = [1998]
    num_domain_tasks = len(TASK_CONFIGS)  # 6 个域任务

    all_results = []

    for run_idx in range(num_runs):
        seed = seeds[run_idx]
        print(f"\n{'#' * 70}")
        print(f"# Experiment {run_idx + 1}/{num_runs}  (seed={seed})")
        print(f"# {num_domain_tasks} domain tasks to run sequentially")
        print(f"{'#' * 70}")

        run_start = time.time()
        print_memory_usage()

        # 加载基础配置
        cli_args = setup_parser().parse_args()
        param = load_json(cli_args.config)
        base_args = vars(cli_args)
        base_args.update(param)
        base_args['seed'] = seed

        # 顺序运行 6 个域任务，收集每个任务的结果
        task_results = []
        for task_id in range(1, num_domain_tasks + 1):
            print(f"\n{'=' * 60}")
            print(f"  Experiment {run_idx + 1}, Domain Task {task_id}/{num_domain_tasks}")
            config = TASK_CONFIGS[task_id]
            print(f"  Train: {config['train']}  →  Test: {config['test']}")
            print(f"{'=' * 60}")

            # 复制配置，注入当前域任务信息
            task_args = base_args.copy()
            task_args['domain_task_id'] = task_id

            # 设置路径（通过 args 传递，让 data_manager 能读到）
            T3, T0 = setup_task(task_id)
            task_args['train_paths'] = T3
            task_args['test_paths'] = T0
            task_args['train_sample'] = config['train']
            task_args['test_sample'] = config['test']

            task_start = time.time()

            # 运行训练
            result = train(task_args)
            result['domain_task_id'] = task_id
            result['train_sample'] = config['train']
            result['test_sample'] = config['test']
            task_results.append(result)

            task_elapsed = time.time() - task_start
            print(f"  Domain Task {task_id} completed in {task_elapsed:.1f}s")

        # 汇总本次实验的结果
        run_result = _aggregate_run(run_idx + 1, seed, task_results)
        all_results.append(run_result)

        print_memory_usage()
        run_elapsed = time.time() - run_start
        print(f"\nExperiment {run_idx + 1} completed in {run_elapsed:.1f}s\n")

    # ============ 全部实验完成，保存 CSV ============
    save_dir = base_args.get('diagnostics_dir', './diagnostics')
    save_results_to_csv(all_results, save_dir, num_domain_tasks)


def _aggregate_run(run_id, seed, task_results):
    """
    汇总一次完整实验（6个域任务）的结果
    """
    result = {
        'run_id': run_id,
        'seed': seed,
    }

    # 每个域任务的精度和遗忘率
    for tr in task_results:
        tid = tr['domain_task_id']
        prefix = f'DomainTask{tid}'

        # 最终精度（最后一个增量任务的精度）
        if tr['cnn_curve']:
            result[f'{prefix}_CNN_final_acc'] = tr['cnn_curve'][-1]
            result[f'{prefix}_CNN_avg_acc'] = tr['cnn_avg_acc']
            result[f'{prefix}_CNN_avg_fgt'] = tr['cnn_avg_forgetting']

        # 精度曲线和遗忘矩阵（字符串存储，方便查看）
        result[f'{prefix}_CNN_curve'] = str(tr['cnn_curve'])
        result[f'{prefix}_CNN_fgt_matrix'] = str(tr['cnn_forgetting_matrix'])
        result[f'{prefix}_domain'] = f"{tr['train_sample']}->{tr['test_sample']}"

    # 跨域任务的平均指标
    cnn_finals = [tr['cnn_curve'][-1] for tr in task_results if tr['cnn_curve']]
    cnn_avgs = [tr['cnn_avg_acc'] for tr in task_results if tr['cnn_curve']]
    cnn_fgts = [tr['cnn_avg_forgetting'] for tr in task_results if tr['cnn_curve']]

    result['Overall_CNN_mean_final_acc'] = round(np.mean(cnn_finals), 2) if cnn_finals else 0
    result['Overall_CNN_mean_avg_acc'] = round(np.mean(cnn_avgs), 2) if cnn_avgs else 0
    result['Overall_CNN_mean_avg_fgt'] = round(np.mean(cnn_fgts), 2) if cnn_fgts else 0

    return result


def save_results_to_csv(all_results, save_dir, num_domain_tasks):
    """保存全部实验结果为 CSV"""
    if not all_results:
        print("No results to save.")
        return

    os.makedirs(save_dir, exist_ok=True)

    df = pd.DataFrame(all_results)

    # 计算 Mean 和 Std（只对数值列）
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [c for c in df.columns if c not in numeric_cols]

    mean_row = {c: 'Mean' if c in ['run_id', 'seed'] else '' for c in non_numeric_cols}
    std_row = {c: 'Std' if c in ['run_id', 'seed'] else '' for c in non_numeric_cols}
    for col in numeric_cols:
        vals = df[col].values.astype(float)
        mean_row[col] = round(np.mean(vals), 2)
        std_row[col] = round(np.std(vals), 2)

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # 保存主汇总表
    csv_path = os.path.join(save_dir, 'results_summary_IEEE_0.3.csv')
    df.to_csv(csv_path, index=False)

    # ============ 打印 ============
    print('\n' + '=' * 70)
    print('ALL EXPERIMENTS COMPLETED')
    print('=' * 70)
    print(f'\n--- Key Metrics ({len(all_results)} runs × {num_domain_tasks} domain tasks) ---')
    print(df_key.to_string(index=False))
    print(f'\n--- Mean ± Std ---')
    for k, v in summary.items():
        print(f'  {k}: {v}')
    print(f'\nFull results:  {csv_path}')
    print(f'Key metrics:   {key_csv_path}')
    print('=' * 70)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorithms.')
    parser.add_argument('--config', type=str, default='./configs/foster_uda.json',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()
