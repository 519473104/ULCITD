import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as tfs

# ============ 数据根目录 ============
DATA_ROOT = r"F:\IEEE\IEEE_32_WT_200_V2"

# ============ 6 个任务的域配置 ============
# 每个任务: (train_sample, test_sample)
# train_sample → T3 (源域/训练), test_sample → T0 (目标域/测试)
TASK_CONFIGS = {
    1: {"train": "Sample_2", "test": "Sample_3"},
    2: {"train": "Sample_3", "test": "Sample_2"},
    3: {"train": "Sample_9", "test": "Sample_2"},
    4: {"train": "Sample_2", "test": "Sample_9"},
    5: {"train": "Sample_3", "test": "Sample_8"},
    6: {"train": "Sample_8", "test": "Sample_3"},
}
# TASK_CONFIGS = {
#     1: {"train": "Sample_2", "test": "Sample_3"},
#     2: {"train": "Sample_8", "test": "Sample_3"},
# 
# }

# 每个样本内的子文件夹（10个故障类别）
SUBFOLDER_NAMES = [
    'M0_G0', 'M0_G1', 'M0_G2', 'M0_G3', 'M0_G4',
    'M1_G0', 'M2_G0', 'M3_G0', 'M0_G0_LA1', 'M0_G0_LA2'
]


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)
    return file_list[0]


def setup_task(task_id, data_root=None):
    """
    根据任务编号设置路径，返回 (T3, T0)
    T3 = 训练域路径列表 (10个类别)
    T0 = 测试域路径列表 (10个类别)

    Args:
        task_id: 1~6
        data_root: 数据根目录，默认使用 DATA_ROOT
    Returns:
        T3: list of 10 paths (训练)
        T0: list of 10 paths (测试)
    """
    if data_root is None:
        data_root = DATA_ROOT

    assert task_id in TASK_CONFIGS, f"Invalid task_id={task_id}, must be 1~6"

    config = TASK_CONFIGS[task_id]
    train_dir = os.path.join(data_root, config["train"])
    test_dir = os.path.join(data_root, config["test"])

    T3 = [os.path.join(train_dir, sub) for sub in SUBFOLDER_NAMES]  # 训练
    T0 = [os.path.join(test_dir, sub) for sub in SUBFOLDER_NAMES]   # 测试

    return T3, T0


# ============ 默认加载 Task 1（兼容旧代码直接 import T3, T0）============
_default_train_dir = os.path.join(DATA_ROOT, TASK_CONFIGS[1]["train"])
_default_test_dir = os.path.join(DATA_ROOT, TASK_CONFIGS[1]["test"])

T3 = [os.path.join(_default_train_dir, sub) for sub in SUBFOLDER_NAMES]
T0 = [os.path.join(_default_test_dir, sub) for sub in SUBFOLDER_NAMES]


if __name__ == "__main__":
    # 测试所有任务的路径
    for tid in range(1, 7):
        t3, t0 = setup_task(tid)
        print(f"\nTask {tid}:")
        print(f"  Train ({TASK_CONFIGS[tid]['train']}): {t3[0]}")
        print(f"  Test  ({TASK_CONFIGS[tid]['test']}):  {t0[0]}")
