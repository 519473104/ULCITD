import sys
import os
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import numpy as np


def train(args):
    """训练入口，返回本次实验的结果字典"""
    return _train(args)


def _train(args):
    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"], args["dataset"], init_cls, args['increment'])
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = 'logs/{}/{}/{}/{}/{}_{}_{}_{}_{}'.format(
        args["model_name"], args["dataset"], init_cls, args['increment'],
        args['prefix'], args['seed'], args['convnet_type'],
        args['beta1'], args["beta2"]
    )

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(filename)s] => %(message)s',
        handlers=[
            logging.FileHandler(filename=logfilename + '.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    _set_random()
    _set_device(args)
    print_args(args)

    # ============ 关键改动：传递train_paths和test_paths ============
    data_manager = DataManager(
        args['dataset'],
        args['shuffle'],
        args['seed'],
        args['init_cls'],
        args['increment'],
        train_paths=args.get('train_paths', None),  # 新增
        test_paths=args.get('test_paths', None)  # 新增
    )

    model = factory.get_model(args['model_name'], args)

    nb_tasks = data_manager.nb_tasks
    cnn_curve, nme_curve = {'top1': [], 'top5': []}, {'top1': [], 'top5': []}

    for task in range(nb_tasks):
        logging.info('All params: {}'.format(count_parameters(model._network)))
        logging.info('Trainable params: {}'.format(count_parameters(model._network, True)))
        model.incremental_train(data_manager)

        cnn_accy, nme_accy = model.eval_task()
        model.after_task()

        if nme_accy is not None and cnn_accy is not None:
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            logging.info('NME: {}'.format(nme_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        elif nme_accy is None:
            logging.info('No NME accuracy.')
            logging.info('CNN: {}'.format(cnn_accy['grouped']))
            cnn_curve['top1'].append(cnn_accy['top1'])
            logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        else:
            logging.info('No CNN accuracy.')
            logging.info('NME: {}'.format(nme_accy['grouped']))
            nme_curve['top1'].append(nme_accy['top1'])
            logging.info('NME top1 curve: {}'.format(nme_curve['top1']))

        # 每个任务结束后计算遗忘率
        if len(cnn_curve['top1']) >= 2:
            _compute_forgetting(cnn_curve['top1'], 'CNN', task)
        if len(nme_curve['top1']) >= 2:
            _compute_forgetting(nme_curve['top1'], 'NME', task)

    # ============ 训练循环结束，输出汇总 ============
    logging.info('\n' + '=' * 60)
    logging.info('All {} tasks completed.'.format(nb_tasks))
    logging.info('=' * 60)

    # CNN 汇总
    cnn_avg = 0.0
    cnn_forgetting_matrix = []
    cnn_avg_forgetting = 0.0
    if len(cnn_curve['top1']) > 0:
        cnn_avg = np.mean(cnn_curve['top1'])
        cnn_forgetting_matrix = _get_forgetting_matrix(cnn_curve['top1'])
        cnn_avg_forgetting = np.mean(cnn_forgetting_matrix[1:]) if len(cnn_forgetting_matrix) > 1 else 0.0
        logging.info('CNN top1 curve: {}'.format(cnn_curve['top1']))
        logging.info('CNN Average Incremental Accuracy: {:.2f}%'.format(cnn_avg))
        logging.info('CNN forgetting matrix: {}'.format([round(f, 2) for f in cnn_forgetting_matrix]))
        logging.info('CNN Average Forgetting: {:.2f}%'.format(cnn_avg_forgetting))

    # NME 汇总
    nme_avg = 0.0
    nme_forgetting_matrix = []
    nme_avg_forgetting = 0.0
    if len(nme_curve['top1']) > 0:
        nme_avg = np.mean(nme_curve['top1'])
        nme_forgetting_matrix = _get_forgetting_matrix(nme_curve['top1'])
        nme_avg_forgetting = np.mean(nme_forgetting_matrix[1:]) if len(nme_forgetting_matrix) > 1 else 0.0
        logging.info('NME top1 curve: {}'.format(nme_curve['top1']))
        logging.info('NME Average Incremental Accuracy: {:.2f}%'.format(nme_avg))
        logging.info('NME forgetting matrix: {}'.format([round(f, 2) for f in nme_forgetting_matrix]))
        logging.info('NME Average Forgetting: {:.2f}%'.format(nme_avg_forgetting))

    logging.info('=' * 60)

    # ============ 生成组合可视化 ============
    if hasattr(model, 'enable_visualization') and model.enable_visualization:
        if hasattr(model, 'task_features_cache') and len(model.task_features_cache) > 0:
            logging.info("\n" + "=" * 60)
            logging.info("All tasks completed. Generating combined visualization...")
            logging.info("=" * 60)
            model.visualize_all_tasks_combined()
            logging.info("Combined visualization saved successfully.")
            logging.info("=" * 60 + "\n")

    # ============ 返回结果字典 ============
    return {
        'seed': args['seed'],
        'cnn_curve': cnn_curve['top1'],
        'nme_curve': nme_curve['top1'],
        'cnn_avg_acc': round(cnn_avg, 2),
        'nme_avg_acc': round(nme_avg, 2),
        'cnn_forgetting_matrix': [round(f, 2) for f in cnn_forgetting_matrix],
        'nme_forgetting_matrix': [round(f, 2) for f in nme_forgetting_matrix],
        'cnn_avg_forgetting': round(cnn_avg_forgetting, 2),
        'nme_avg_forgetting': round(nme_avg_forgetting, 2),
    }


def _get_forgetting_matrix(acc_list):
    """构造 1×T 遗忘矩阵"""
    a = np.array(acc_list)
    T = len(a)
    matrix = np.zeros(T)
    for i in range(1, T):
        matrix[i] = np.max(a[:i + 1]) - a[i]
    return matrix.tolist()


def _compute_forgetting(acc_list, name, current_task, final=False):
    """打印当前任务的遗忘率"""
    prefix = 'FINAL ' if final else ''
    a = np.array(acc_list)
    T = len(a)

    forgetting_matrix = np.zeros(T)
    for i in range(1, T):
        best_so_far = np.max(a[:i + 1])
        forgetting_matrix[i] = best_so_far - a[i]

    avg_forgetting = np.mean(forgetting_matrix[1:]) if T > 1 else 0.0

    logging.info(f'[{prefix}Forgetting] {name} after Task {current_task}:')
    logging.info(f'[{prefix}Forgetting] {name} top1 curve:       {list(a)}')
    logging.info(f'[{prefix}Forgetting] {name} forgetting matrix: '
                 f'{[round(f, 2) for f in forgetting_matrix]}')
    logging.info(f'[{prefix}Forgetting] {name} best: {a.max():.2f}%, '
                 f'current: {a[-1]:.2f}%, drop: {a.max() - a[-1]:.2f}%')
    logging.info(f'[{prefix}Forgetting] {name} Average Forgetting: {avg_forgetting:.2f}%')


def _set_device(args):
    device_type = args['device']
    gpus = []
    for device in device_type:
        if device_type == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:{}'.format(device))
        gpus.append(device)
    args['device'] = gpus


def _set_random():
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info('{}: {}'.format(key, value))
