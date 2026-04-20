import os
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms as tfs
# from utils.Tsinghua_path import T3, T0
import random
# from utils.IEEE_path import T3, T0
import torch
from sklearn.model_selection import train_test_split


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class tsinghua(iData):
    use_path = False

    def read_directory(self, directory_name, height, width):
        """读取目录中的所有图像"""
        file_list = os.listdir(directory_name)
        img = []
        for each_file in file_list:
            img0 = Image.open(os.path.join(directory_name, each_file))
            gray = img0.resize((height, width))
            img.append(np.array(gray).astype(np.float32))
        data = np.array(img) / 255.0  # 归一化
        data = data.reshape(-1, 3, height, width)
        return data

    def safe_sample(self, data, k=100):
        """兼容 dict / list / ndarray 的随机采样"""
        if isinstance(data, dict):
            data = list(data.values())
        elif isinstance(data, np.ndarray):
            data = data.tolist()
        return random.sample(data, min(k, len(data)))

    def get_data(self, data_file, height=32, width=32, num_classes=10):
        """
        读取训练或测试数据
        Args:
            data_file: 类别路径列表 (长度应为num_classes)
            height: 图像高度
            width: 图像宽度
            num_classes: 类别数
        Returns:
            data: numpy数组 (N, 3, H, W)
            label: 标签列表
        """
        classes_data = []

        for i in range(num_classes):
            class_data = self.read_directory(data_file[i], height, width)
            # 随机选取100个样本（如果不足100个则取全部）
            class_sample = random.sample(list(class_data), min(100, len(class_data)))
            classes_data.append(np.array(class_sample))

        # 拼接所有类别数据
        data = np.concatenate(classes_data, axis=0)

        # 生成标签
        label = []
        for i in range(num_classes):
            label.extend([i] * (data.shape[0] // num_classes))

        return data, label

    def get_data_T(self, data_file, height=32, width=32, num_classes=10):
        """
        读取测试数据（同get_data）
        """
        classes_data = []

        for i in range(num_classes):
            class_data = self.read_directory(data_file[i], height, width)
            # 随机选取100个样本
            class_sample = random.sample(list(class_data), min(100, len(class_data)))
            classes_data.append(np.array(class_sample))

        data = np.concatenate(classes_data, axis=0)

        label = []
        for i in range(num_classes):
            label.extend([i] * (data.shape[0] // num_classes))

        return data, label

    def Tsing_hua_data_split(self, train_paths=None, test_paths=None):
        """
        清华数据集分割
        Args:
            train_paths: 训练集路径列表 (10个故障类别)，如果为None则使用全局T3
            test_paths: 测试集路径列表 (10个故障类别)，如果为None则使用全局T0
        """
        self.num_classes = 10

        # ✅ 关键：如果没有传递路径，使用默认的全局T3和T0
        if train_paths is None:
            train_paths = T3
        if test_paths is None:
            test_paths = T0

        # 加载数据
        self.train_data, self.train_targets = self.get_data(
            train_paths, height=32, width=32, num_classes=10
        )
        self.test_data, self.test_targets = self.get_data_T(
            test_paths, height=32, width=32, num_classes=10
        )

# class iData(object):
#     train_trsf = []
#     test_trsf = []
#     common_trsf = []
#     class_order = None
#
# class tsinghua(iData):
#     use_path = False
#
#     def read_directory(self,directory_name, height, width):
#         # height=64
#         # width=64
#         # normal=1
#         file_list = os.listdir(directory_name)
#         img = []
#         for each_file in file_list:
#             img0 = Image.open(directory_name + '/' + each_file)
#             gray = img0.resize((height, width))
#             img.append(np.array(gray).astype(np.float32))
#             data = np.array(img) / 255.0  # 归一化
#             data = data.reshape(-1, 3, height, width)
#         return data
#
#     def safe_sample(data, k=100):
#         """兼容 dict / list / ndarray 的随机采样"""
#         if isinstance(data, dict):
#             data = list(data.values())  # 如果是字典，取 values
#         elif isinstance(data, np.ndarray):
#             data = data.tolist()  # 如果是 ndarray 转 list
#         # 默认 data 是 list
#         return random.sample(data, min(k, len(data)))
#
#     def get_data(self, data_file, height=32, width=32,num_classes=10):
#         # data
#         class1 = self.read_directory(data_file[0], height, width)
#         class2 = self.read_directory(data_file[1], height, width)
#         class3 = self.read_directory(data_file[2], height, width)
#         class4 = self.read_directory(data_file[3], height, width)
#         class5 = self.read_directory(data_file[4], height, width)
#         class6 = self.read_directory(data_file[5], height, width)
#         class7 = self.read_directory(data_file[6], height, width)
#         class8 = self.read_directory(data_file[7], height, width)
#         class9 = self.read_directory(data_file[8], height, width)
#         class10 = self.read_directory(data_file[9], height, width)
#
#
#
#         # class_1 = class1[:100]
#         # class_2 = class2[:100]
#         # class_3 = class3[:100]
#         # class_4 = class4[:100]
#         # class_5 = class5[:100]
#         # class_6 = class6[:100]
#         # class_7 = class7[:100]
#         # class_8 = class8[:100]
#         # class_9 = class9[:100]
#         # class_10 = class10[:100]
#
#         # 随机选取 100 个样本（不放回采样）
#         class_1 = random.sample(list(class1), min(100, len(class1)))
#         class_2 = random.sample(list(class2), min(100, len(class2)))
#         class_3 = random.sample(list(class3), min(100, len(class3)))
#         class_4 = random.sample(list(class4), min(100, len(class4)))
#         class_5 = random.sample(list(class5), min(100, len(class5)))
#         class_6 = random.sample(list(class6), min(100, len(class6)))
#         class_7 = random.sample(list(class7), min(100, len(class7)))
#         class_8 = random.sample(list(class8), min(100, len(class8)))
#         class_9 = random.sample(list(class9), min(100, len(class9)))
#         class_10 = random.sample(list(class10), min(100, len(class10)))
#
#         data = np.concatenate((class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10), axis=0)
#
#         label = []
#         for i in range(num_classes):
#             j = 0
#             while j < (data.shape[0] // num_classes):
#                 label.append(i)
#                 j += 1
#
#         return data, label
#
#     def get_data_T(self, data_file, height=32, width=32,num_classes=10):
#         # data
#         class1 = self.read_directory(data_file[0], height, width)
#         class2 = self.read_directory(data_file[1], height, width)
#         class3 = self.read_directory(data_file[2], height, width)
#         class4 = self.read_directory(data_file[3], height, width)
#         class5 = self.read_directory(data_file[4], height, width)
#         class6 = self.read_directory(data_file[5], height, width)
#         class7 = self.read_directory(data_file[6], height, width)
#         class8 = self.read_directory(data_file[7], height, width)
#         class9 = self.read_directory(data_file[8], height, width)
#         class10 = self.read_directory(data_file[9], height, width)
#
# # tsinghua data
#
#         # class_1 = class1[100:200]
#         # class_2 = class2[100:200]
#         # class_3 = class3[100:200]
#         # class_4 = class4[100:200]
#         # class_5 = class5[100:200]
#         # class_6 = class6[100:200]
#         # class_7 = class7[100:200]
#         # class_8 = class8[100:200]
#         # class_9 = class9[100:200]
#         # class_10 = class10[100:200]
#
#         # class_1 = class1[:100]
#         # class_2 = class2[:100]
#         # class_3 = class3[:100]
#         # class_4 = class4[:100]
#         # class_5 = class5[:100]
#         # class_6 = class6[:100]
#         # class_7 = class7[:100]
#         # class_8 = class8[:100]
#         # class_9 = class9[:100]
#         # class_10 = class10[:100]
#
#         # 随机选取 100 个样本（不放回采样）
#         class_1 = random.sample(list(class1), min(100, len(class1)))
#         class_2 = random.sample(list(class2), min(100, len(class2)))
#         class_3 = random.sample(list(class3), min(100, len(class3)))
#         class_4 = random.sample(list(class4), min(100, len(class4)))
#         class_5 = random.sample(list(class5), min(100, len(class5)))
#         class_6 = random.sample(list(class6), min(100, len(class6)))
#         class_7 = random.sample(list(class7), min(100, len(class7)))
#         class_8 = random.sample(list(class8), min(100, len(class8)))
#         class_9 = random.sample(list(class9), min(100, len(class9)))
#         class_10 = random.sample(list(class10), min(100, len(class10)))
#
#
#         data = np.concatenate((class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8, class_9, class_10), axis=0)
#         # data = np.concatenate(
#         #     (class_1, class_2, class_3, class_4, class_5, class_6, class_7, class_8), axis=0)
#
#         label = []
#         for i in range(num_classes):
#             j = 0
#             while j < (data.shape[0] // num_classes):
#                 label.append(i)
#                 j += 1
#
#         return data, label
#
#     def Tsing_hua_data_split(self):
#         # get source train and val
#         self.num_classes = 10
#         self.train_data, self.train_targets = self.get_data(T3, height=32, width=32, num_classes=10)
#         self.test_data, self.test_targets = self.get_data_T(T0, height=32, width=32, num_classes=10)

