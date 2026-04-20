import json
import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
from trainer import train


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

    all_results = []

    for i in range(num_runs):
        seed = seeds[i]
        print(f"\nRunning iteration {i + 1}/{num_runs} (seed={seed})...")
        start_time = time.time()
        print_memory_usage()

        args = setup_parser().parse_args()
        param = load_json(args.config)
        args = vars(args)
        args.update(param)
        args['seed'] = seed

        # 训练并收集结果
        result = train(args)
        all_results.append(result)

        print_memory_usage()
        elapsed_time = time.time() - start_time
        print(f"Iteration {i + 1} completed in {elapsed_time:.2f} seconds.\n")

    # ============ 全部实验完成，保存 CSV ============
    save_results_to_csv(all_results, args)


def save_results_to_csv(all_results, args):
    """
    将 5 次实验的结果汇总保存为 CSV

    生成文件：
      results_summary.csv — 每行一次实验 + 最后两行是 Mean 和 Std
    """
    if not all_results:
        print("No results to save.")
        return

    # 确定任务数
    nb_tasks = len(all_results[0]['cnn_curve'])

    # ============ 构造 DataFrame ============
    rows = []
    for r in all_results:
        row = {'seed': r['seed']}

        # CNN 每任务精度: Task0_acc, Task1_acc, ...
        for t, acc in enumerate(r['cnn_curve']):
            row[f'CNN_Task{t}_acc'] = acc

        row['CNN_Avg_Acc'] = r['cnn_avg_acc']

        # CNN 遗忘矩阵: Task0_fgt, Task1_fgt, ...
        for t, fgt in enumerate(r['cnn_forgetting_matrix']):
            row[f'CNN_Task{t}_fgt'] = fgt

        row['CNN_Avg_Forgetting'] = r['cnn_avg_forgetting']

        # NME 每任务精度
        for t, acc in enumerate(r['nme_curve']):
            row[f'NME_Task{t}_acc'] = acc

        row['NME_Avg_Acc'] = r['nme_avg_acc']

        # NME 遗忘矩阵
        for t, fgt in enumerate(r['nme_forgetting_matrix']):
            row[f'NME_Task{t}_fgt'] = fgt

        row['NME_Avg_Forgetting'] = r['nme_avg_forgetting']

        rows.append(row)

    df = pd.DataFrame(rows)

    # ============ 计算 Mean 和 Std ============
    numeric_cols = [c for c in df.columns if c != 'seed']

    mean_row = {'seed': 'Mean'}
    std_row = {'seed': 'Std'}
    for col in numeric_cols:
        vals = df[col].values.astype(float)
        mean_row[col] = round(np.mean(vals), 2)
        std_row[col] = round(np.std(vals), 2)

    df = pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)

    # ============ 保存 ============
    save_dir = args.get('diagnostics_dir', './diagnostics')
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'results_summary.csv')
    df.to_csv(csv_path, index=False)

    # ============ 打印 ============
    print('\n' + '=' * 70)
    print('ALL EXPERIMENTS COMPLETED — SUMMARY')
    print('=' * 70)
    print(df.to_string(index=False))
    print('=' * 70)
    print(f'Results saved to: {csv_path}')

    # 单独打印关键指标的 Mean ± Std
    print('\n--- Key Metrics (Mean ± Std) ---')
    for metric in ['CNN_Avg_Acc', 'CNN_Avg_Forgetting', 'NME_Avg_Acc', 'NME_Avg_Forgetting']:
        if metric in df.columns:
            vals = df[metric].iloc[:-2].values.astype(float)  # 排除 Mean/Std 行
            print(f'  {metric}: {np.mean(vals):.2f} ± {np.std(vals):.2f}')

    print('=' * 70)


def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)
    return param


def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple continual learning algorthms.')
    parser.add_argument('--config', type=str, default='./configs/foster_uda.json',
                        help='Json file of settings.')
    return parser


if __name__ == '__main__':
    main()
