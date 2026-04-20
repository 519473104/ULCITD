import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as tfs


# ============ 数据根目录 ============
DATA_ROOT = r"F:\Tsinghua\DAIC\Tsinghua_32_ic_WT_x1_200"

# ============ 6 个任务的域配置 ============
TASK_CONFIGS = {
    1: {"train": "1200rpm_0dot4A", "test": "1200rpm_0A"},
    2: {"train": "900rpm_0dot4A", "test": "900rpm_0A"},
    3: {"train": "1200rpm_0A", "test": "1500rpm_0A"},
    4: {"train": "1500rpm_0A", "test": "1200rpm_0A"},
    5: {"train": "1500rpm_0A", "test": "900rpm_0A"},
    6: {"train": "900rpm_0A", "test": "1500rpm_0A"},
}
# TASK_CONFIGS = {
#     1: {"train": "1200rpm_0dot4A", "test": "1200rpm_0A"},
#     # 2: {"train": "900rpm_0dot4A", "test": "900rpm_0A"},
#     # 1: {"train": "1200rpm_0A", "test": "1500rpm_0A"},
#     # 4: {"train": "1500rpm_0A", "test": "1200rpm_0A"},
#     # 5: {"train": "1500rpm_0A", "test": "900rpm_0A"},
#     2: {"train": "900rpm_0A", "test": "1500rpm_0A"},
# }
# 10个故障类别子文件夹
SUBFOLDER_NAMES = ['H', 'SG_TM', 'IB_IRC','IB_ORC',
                   'PB_IRC','PB_ORF','PG_TM', 'PG_CT', 'SG_CT','PB_REF']


def get_file(root_path):
    file_list = os.listdir(path=root_path)
    file_list = [os.path.join(root_path, f) for f in file_list]
    assert len(file_list) == 1, 'There are {} files in [{}]'.format(len(file_list), root_path)
    return file_list[0]


def setup_task(task_id, data_root=None):
    """
    根据任务编号设置路径
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

    T3 = [os.path.join(train_dir, sub) for sub in SUBFOLDER_NAMES]
    T0 = [os.path.join(test_dir, sub) for sub in SUBFOLDER_NAMES]

    return T3, T0


# ============ 默认加载 Task 1（兼容旧代码直接 import T3, T0）============
_default_train_dir = os.path.join(DATA_ROOT, TASK_CONFIGS[1]["train"])
_default_test_dir = os.path.join(DATA_ROOT, TASK_CONFIGS[1]["test"])

T3 = [os.path.join(_default_train_dir, sub) for sub in SUBFOLDER_NAMES]
T0 = [os.path.join(_default_test_dir, sub) for sub in SUBFOLDER_NAMES]


if __name__ == "__main__":
    for tid in range(1, 7):
        t3, t0 = setup_task(tid)
        cfg = TASK_CONFIGS[tid]
        print(f"Task {tid}: train={cfg['train']} -> test={cfg['test']}")
        print(f"  T3[0]: {t3[0]}")
        print(f"  T0[0]: {t0[0]}")

