import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['CompactResNet', 'compact_resnet8', 'compact_resnet14', 'compact_resnet20', 
           'compact_resnet32', 'wide_compact_resnet', 'se_compact_resnet']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompactBasicBlock(nn.Module):
    """针对小图像优化的BasicBlock"""
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_se=False):
        super(CompactBasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('CompactBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in CompactBasicBlock")

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride
        
        # 可选的SE注意力机制
        self.se = SEBlock(planes) if use_se else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        
        # SE注意力
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CompactBottleneck(nn.Module):
    """针对小图像优化的Bottleneck"""
    expansion = 2  # 减小expansion比例

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, use_se=False):
        super(CompactBottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        
        self.se = SEBlock(planes * self.expansion) if use_se else None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.se is not None:
            out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class CompactResNet(nn.Module):
    """针对32x32小图像优化的ResNet架构"""

    def __init__(self, block, layers, num_classes=10, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, args=None, use_se=False, dropout_rate=0.0):
        super(CompactResNet, self).__init__()
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 16  # 减小初始通道数
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple")
        
        self.groups = groups
        self.base_width = width_per_group
        self.use_se = use_se
        self.dropout_rate = dropout_rate
        self.args = args if args is not None else {}

        # 针对32x32图像的轻量化输入层
        # 避免过度下采样，保留更多空间信息
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )
        # 输出: 32x32

        # 渐进式特征提取，避免过度下采样
        self.layer1 = self._make_layer(block, 16, layers[0])                    # 32x32
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2,          # 16x16
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2,          # 8x8
                                       dilate=replace_stride_with_dilation[1])
        
        # 可选的第四层（对于更深的网络）
        if len(layers) > 3:
            self.layer4 = self._make_layer(block, 128, layers[3], stride=2,     # 4x4
                                           dilate=replace_stride_with_dilation[2])
            self.out_dim = 128 * block.expansion
        else:
            self.layer4 = None
            self.out_dim = 64 * block.expansion

        # 全局平均池化
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        # 可选的dropout
        self.dropout = nn.Dropout(dropout_rate) if dropout_rate > 0 else None

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, CompactBottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, CompactBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        
        if dilate:
            self.dilation *= stride
            stride = 1
            
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, self.use_se))
        self.inplanes = planes * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, use_se=self.use_se))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        """前向传播实现"""
        # 输入层: 32x32x3 -> 32x32x16
        x = self.conv1(x)

        # 特征提取层
        x_1 = self.layer1(x)      # 32x32x16
        x_2 = self.layer2(x_1)    # 16x16x32  
        x_3 = self.layer3(x_2)    # 8x8x64

        feature_maps = [x_1, x_2, x_3]
        
        if self.layer4 is not None:
            x_4 = self.layer4(x_3) # 4x4x128
            feature_maps.append(x_4)
            final_features = x_4
        else:
            final_features = x_3

        # 全局池化和特征提取
        pooled = self.avgpool(final_features)  # 1x1x(64 or 128)
        features = torch.flatten(pooled, 1)    # (64 or 128)
        
        # 可选dropout
        if self.dropout is not None:
            features = self.dropout(features)

        return {
            'fmaps': feature_maps,
            'features': features
        }

    def forward(self, x):
        return self._forward_impl(x)

    @property
    def feature_dim(self):
        """返回特征维度"""
        return self.out_dim

    @property
    def last_conv(self):
        """返回最后一个卷积层"""
        if self.layer4 is not None:
            if hasattr(self.layer4[-1], 'conv3'):
                return self.layer4[-1].conv3
            else:
                return self.layer4[-1].conv2
        else:
            if hasattr(self.layer3[-1], 'conv3'):
                return self.layer3[-1].conv3
            else:
                return self.layer3[-1].conv2


# ============ 工厂函数 ============

def _compact_resnet(arch, block, layers, **kwargs):
    """通用构造函数"""
    model = CompactResNet(block, layers, **kwargs)
    return model


def compact_resnet8(**kwargs):
    """8层紧凑ResNet - 适合小数据集"""
    return _compact_resnet('compact_resnet8', CompactBasicBlock, [1, 1, 1], **kwargs)


def compact_resnet14(**kwargs):
    """14层紧凑ResNet"""
    return _compact_resnet('compact_resnet14', CompactBasicBlock, [2, 2, 2], **kwargs)


def compact_resnet20(**kwargs):
    """20层紧凑ResNet - CIFAR标准配置"""
    return _compact_resnet('compact_resnet20', CompactBasicBlock, [3, 3, 3], **kwargs)


def compact_resnet32(**kwargs):
    """32层紧凑ResNet - 更深的网络"""
    return _compact_resnet('compact_resnet32', CompactBasicBlock, [5, 5, 5], **kwargs)


def compact_resnet26_bottleneck(**kwargs):
    """26层Bottleneck版本"""
    return _compact_resnet('compact_resnet26', CompactBottleneck, [2, 2, 2, 2], **kwargs)


def wide_compact_resnet(**kwargs):
    """宽度增强的紧凑ResNet"""
    kwargs['width_per_group'] = 128  # 增加宽度
    return _compact_resnet('wide_compact_resnet', CompactBasicBlock, [2, 2, 2], **kwargs)


def se_compact_resnet(**kwargs):
    """带SE注意力的紧凑ResNet"""
    kwargs['use_se'] = True
    return _compact_resnet('se_compact_resnet', CompactBasicBlock, [3, 3, 3], **kwargs)


# ============ 多分支网络 ============

class MultiPathCompactResNet(CompactResNet):
    """多路径紧凑ResNet，增强特征表示"""
    
    def __init__(self, block, layers, **kwargs):
        super(MultiPathCompactResNet, self).__init__(block, layers, **kwargs)
        
        # 多尺度输入处理
        self.scale_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=1, padding=0),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(inplace=True)
            )
        ])
        
        # 特征融合
        self.feature_fusion = nn.Sequential(
            nn.Conv2d(16, self.inplanes, kernel_size=1),
            nn.BatchNorm2d(self.inplanes),
            nn.ReLU(inplace=True)
        )

    def _forward_impl(self, x):
        # 多尺度特征提取
        scale_features = []
        for scale_conv in self.scale_convs:
            scale_features.append(scale_conv(x))
        
        # 特征融合
        multi_scale = torch.cat(scale_features, dim=1)
        fused = self.feature_fusion(multi_scale)
        
        # 继续标准ResNet流程
        x_1 = self.layer1(fused)
        x_2 = self.layer2(x_1)
        x_3 = self.layer3(x_2)
        
        feature_maps = [x_1, x_2, x_3]
        
        if self.layer4 is not None:
            x_4 = self.layer4(x_3)
            feature_maps.append(x_4)
            final_features = x_4
        else:
            final_features = x_3

        pooled = self.avgpool(final_features)
        features = torch.flatten(pooled, 1)
        
        if self.dropout is not None:
            features = self.dropout(features)

        return {
            'fmaps': feature_maps,
            'features': features,
            'multi_scale_features': multi_scale
        }


def multipath_compact_resnet(**kwargs):
    """多路径紧凑ResNet"""
    return MultiPathCompactResNet(CompactBasicBlock, [3, 3, 3], **kwargs)


# ============ 使用示例和测试 ============
if __name__ == "__main__":
    # 测试不同配置
    models = {
        'compact_resnet8': compact_resnet8(),
        'compact_resnet20': compact_resnet20(), 
        'se_compact_resnet': se_compact_resnet(),
        'wide_compact_resnet': wide_compact_resnet(),
        'multipath_compact_resnet': multipath_compact_resnet()
    }
    
    # 测试输入
    batch_size = 4
    test_input = torch.randn(batch_size, 3, 32, 32)
    
    print("=" * 60)
    print("Testing Compact ResNet variants for 32x32 images")
    print("=" * 60)
    
    for name, model in models.items():
        with torch.no_grad():
            outputs = model(test_input)
            
            # 计算参数量
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            print(f"\n{name}:")
            print(f"  Total parameters: {total_params:,}")
            print(f"  Trainable parameters: {trainable_params:,}")
            print(f"  Feature dimension: {model.feature_dim}")
            print(f"  Output features shape: {outputs['features'].shape}")
            print("  Feature map shapes:")
            for i, fmap in enumerate(outputs['fmaps']):
                print(f"    Layer {i+1}: {fmap.shape}")
            
            # 计算FLOPs（简单估算）
            input_size = test_input.numel()
            feature_size = sum(fmap.numel() for fmap in outputs['fmaps'])
            print(f"  Approx. memory usage: {(input_size + feature_size) * 4 / 1024**2:.2f} MB")
    
    print("\n" + "=" * 60)
    print("All tests completed successfully!")
    print("Recommended for CIFAR-10/CIFAR-100: compact_resnet20")
    print("Recommended for small datasets: compact_resnet8 or compact_resnet14")
    print("Recommended for enhanced performance: se_compact_resnet")
    print("=" * 60)