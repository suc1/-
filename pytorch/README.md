# torch
1. deepshare.net
2. [60AI经典论文](https://deepshare.feishu.cn/docs/doccnewbeOX1q1t4Pk5npjv0p5d)
3. [深度学习路线图](https://www.zhihu.com/question/437199981/answer/3310028730)

## env
1. `conda create --name pytorchTest opencv matplotlib -y`
2. `conda activate pytorchTest`
3. `conda install -c conda-forge pytorch-cpu torchvision tensorboard future`
4. `conda install psutil`
5. `tensorboard --logdir=./runs`
6. `pip install torchsummary`
7. Install `libjpeg` or `libpng`:
	1. `conda install Pillow`
	2. Linux: `sudo apt-get install libjpeg-dev libpng-dev`
	3. `conda uninstall torchvision`
	4. `conda install torchvision`



## version
```
torch.__version__
torch.cuda.is_available()
torch.version.cuda
torch.cuda.get_device_name(0)
```

## Tensor
```
torch.tensor(data, dtype=None, device=None, requires_grad=False, pin_memory=False)
torch.from_numpy(ndarray)		# tensor.data 共享内存

torch.zeros (*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.zeros_like(input, dtype=None, layout=None, device=None, requires_grad=False)	# 只是形状一致

w.grad.zero_()		#Reset tensor.grad
x.data.numpy()

torch.ones()
torch.ones_like()

torch.full (size, fill_value, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.full_like()

torch.arange( start=0, end, step=1, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) # 等差数列
torch.linspace(start, end, steps=100, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) # 均分
torch.logspace(start, end, steps=100, base=10.0, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False)
torch.eye(n, m=None, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False )

torch.normal (mean, std, out=None)																	# 正态分布
torch.normal (mean, std, size, out=None)
torch.randn (*size, out=None, dtype=None, layout=torch.strided, device=None, requires_grad=False) 	#标准正态分布
torch.randn_like ()

torch.rand()
torch.rand_like()		# 均匀分布
torch.randint()
torch.randint_like()

torch.randperm()		# 均匀一维张量
torch.bernoulli(input, *, generator=None, out=None)
```

## 张量操作
```
torch.cat (tensors, dim=0, out=None)		# 在指定维度 dim 上进行扩充
torch.stack(tensors, dim=0, out=None)		# 在新创建的维度 dim 上进行拼接

torch.chunk( input, chunks, dim=0)
torch.split(tensor, split_size_or_sections, dim=0)

torch.index_select( input, dim, index, out=None)
t.le(5)		#ge(), gt(), lt()
torch.masked_select( input, mask, out=None)		# 结果是一维

torch.reshape( input, shape)		# 共享数据内存
torch.transpose( input, dim0, dim1)
torch.t( input )

torch.squeeze( input, dim=None, out=None)
torch.usqueeze( input, dim, out=None)

torch.add()			# out = input + alpha × other
torch.addcdiv()		# out = input + value * tensor1 / tensor2
torch.addcmul()		# out = input + value * tensor1 * tensor2
torch.sub()
torch.div()
torch.mul()

torch.mm()          # 矩阵相乘，不传播
torch.matmul()      # 矩阵相乘，传播

torch.log( input , out=None)
torch.log10( input , out=None)
torch.log2( input , out=None)
torch.exp( input , out=None)
torch.pow()

torch.abs( input , out=None)
torch.acos( input , out=None)
torch.cosh( input , out=None)			#
torch.cos( input , out=None)
torch.asin( input , out=None)
torch.atan( input , out=None)
torch.atan2( input , other, out=None)	#
```

## 线性回归 Linear Regression
1. 线性关系: y = wx + b
2. 求解步骤: 确定模型，选择损失函数，求解梯度并更新: w = w - LR * w.grad, b = b - LR * w.grad, 

## 计算图与动态图机制
1. 结点表示数据, 边表示运算
2. 梯度求导: 每条路径之间分段是相乘关系，所有路径之间是相加关系（求导法则）
3. `torch.Tensor`: data, dtype, shape, device, requires_grad, grad, grad_fn, is_leaf
4. 动态图, 静态图

## autograd 自动求导系统
1. `torch.autograd.backward( tensors, grad_tensors=None, retain_graph=None, create_graph=False)`
2. `torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False)` 求取梯度
    1. `outputs`进行求导运算
	2. `inputs`求导以后带入值计算最后值
	3. 梯度不自动清零
    4. 依赖于叶子结点的结点， requires_grad 默认为 True
    5. 叶子结点不可执行 in place
3. 逻辑回归 Logistic Regression
    1. 线性 的 二分类
    2. Sigmoid 函数，也称为Logistic函数, (0, 1)
    3. 线性回归 `y = Wx + b`
    4. 对数回归 `lny = Wx + b`
    5. 对数几率回归 `ln(y/(1-y)) = Wx + b`
4. 迭代训练步骤: 数据， 模型， 损失函数， 优化器

## DataLoader与Dataset
1. 数据: 收集, 划分(Train, Test), 读取(DataLoader, DataSet), 预处理(transforms)
2. `DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)`
3. `epoch` = `iteration` * `batchSize`
4. `Dataset.__getitem__()`
5. `DataLoader`:
	1. 读哪些数据: Sampler输出的 Index
	2. 从哪读数据: Dataset中的 data_dir
	3. 怎么读数据: Dataset中的 __getitem__
6. DataLoader -> DataLoaderIter -> Sampler (-> Index) -> DatasetFetcher -> Dataset -> getitem -> (Img, Label) -> collate_fn -> (Batch Data)
7. `BASE_DIR = os.path.dirname(os.path.abspath(__file__))`
8. `os.path`: `abspath()`, `exists()`
9. 分成三组: train, valid, test. 按比例划分，各个集不相交
```
	train_pct = 0.8
    valid_pct = 0.1
    test_pct = 0.1

    for root, dirs, files in os.walk(dataset_dir):	# 只处理当前目录，不进入下面的子目录
        for sub_dir in dirs:

            imgs = os.listdir(os.path.join(root, sub_dir))
            imgs = list(filter(lambda x: x.endswith('.jpg'), imgs))
            random.shuffle(imgs)
            img_count = len(imgs)

            train_point = int(img_count * train_pct)
            valid_point = int(img_count * (train_pct + valid_pct))

            if img_count == 0:
                print("{}目录下，无图片，请检查".format(os.path.join(root, sub_dir)))
                import sys
                sys.exit(0)
            for i in range(img_count):
                if i < train_point:
                    out_dir = os.path.join(train_dir, sub_dir)
                elif i < valid_point:
                    out_dir = os.path.join(valid_dir, sub_dir)
                else:
                    out_dir = os.path.join(test_dir, sub_dir)

                makedir(out_dir)

                target_path = os.path.join(out_dir, imgs[i])
                src_path = os.path.join(dataset_dir, sub_dir, imgs[i])

                shutil.copy(src_path, target_path)
```
10. 增加python搜索路径`sys.path.append(hello_pytorch_DIR)`

## 图像预处理 transforms
1. `torchvision.transforms`, `torchvision.datasets`, `torchvision.model`
2. 预处理方法: 数据中心化, 数据标准化, 缩放, 裁剪, 旋转, 翻转, 填充, 噪声添加, 灰度变换, 线性变换, 仿射变换, 亮度、饱和度及对比度变换
3. 可以放到数据清洗阶段: ??
4. 在`getitem()`之后的`transforms`是一变多??
5. 标准化: `transforms.Normalize()`是所有通道的`mean`,`std`. 怎么选择: 取决于数据集的特性

## 图像增强
1. 裁剪: `transforms.CenterCrop()`, `transforms.RandomCrop()`, `RandomResizedCrop()`, `FiveCrop()`, `TenCrop()`
2. 翻转: `RandomHorizontalFlip()`, `RandomVerticalFlip()`
3. 旋转: `RandomRotation()`
4. 图像变换: `Pad()`
4. 调整亮度、对比度、饱和度和色相: `ColorJitter()`. 标准化以后，不用这个???
5. 依概率将图片转换为灰度图: `Grayscale()`, `RandomGrayscale()`
6. `RandomAffine()`, `transforms.LinearTransformation`, `RandomErasing()`, `transforms.Lambda()`
7. `transforms.Resize`, `transforms.Totensor`, `transforms.Normalize`
8. `transforms.RandomChoice()`, `transforms.RandomApply`, `transforms.RandomOrder()`
9. 自定义transforms: `__init__`传入参数，`__call__`只有一个图片参数. 椒盐噪声
10. 应用(原则：让训练集与测试集/现实更接近): 空间位置 ：平移; 色彩 ：灰度图，色彩抖动; 形状 ：仿射变换; 上下文场景 ：遮挡，填充
11. `if isinstance(m, nn.Conv2d)`
12. 训练模式`net.train()`, 评价模式`net.eval()`, 使用模式
13. 刚体变换（Rigid Transformation）: 平移（Translation）和旋转（Rotation）. 物体的大小和形状不变
14. 仿射变换（Affine Transformation）：刚体变换 + 缩放/翻转/剪切（Shear）. 直线仍为直线, 平行线仍为平行线
15. 投影变换（Projective Transformation）: 仿射变换 + 透视变换(3D -> 2D). 
16. 在模型训练中，通常同一张图片会被输入模型多次，而不是仅输入一次。这一技术通常被称为数据增强（Data Augmentation）: 包括随机旋转、水平翻转、垂直翻转、缩放、平移等. 增加训练数据的多样, 提升模型的泛化能力
17. 在不同的`epoch`中，即使没有图片的随机变换，同一个图片还是被输入多次
18. `with torch.no_grad():` 用于指定在其内部的代码块中不计算梯度。用于验证/推断（inference）,可以提高代码的执行效率，并减少内存使用

## 卷积核的数量或卷积核的深度
1. `self.conv1 = nn.Conv2d(3, 6, 5)`表示输入通道数为 3，输出通道数为 6，卷积核大小为 5x5
2. 输出通道数6，也叫做卷积核的数量或卷积核的深度
3. 每个卷积核都在所有输入通道上逐通道进行卷积操作, 并将3通道结果相加，最终得到一个输出通道的结果
4. 因此，对于每个卷积核，不论输入是几个通道, 最终都会产生一个输出通道
5. 在卷积神经网络中，卷积核的大小通常指的是在一个通道上的大小，而不是考虑所有通道。因此，在 `nn.Conv2d(3, 6, 5)` 中，卷积核的大小是指在每个输入通道上的大小为 `5x5`
6. 有 6 个这样的卷积核，每个卷积核都是大小为 5x5 的矩阵。这 6 个卷积核分别与输入数据进行卷积，产生 6 个输出通道
7. 查看这些卷积核的具体数值: `print(self.conv1.weight)`
8. `self.fc1 = nn.Linear(16 * 5 * 5, 120)`中的`16 * 5 * 5`来自于前两个卷积层的`输出通道 * 输出形状`
9. `torch.nn.Linear(in_features, out_features, bias=True)`

## lesson-06/2_train_lenet.py
1. 输入数据的形状为 `(batch_size, 3, height, width)`, 即`(bs, 3, 32, 32)`
2. 经过`class LeNet(nn.Module).forward()`的各个步骤:
3. `nn.Conv2d(3, 6, 5)`: `(bs, 6, 28, 28)`, 其中`28=32-5+1`
4. `F.max_pool2d(out, 2)`: `(bs, 6, 14, 14)`
5. `nn.Conv2d(6, 16, 5)`: `(bs, 16, 10, 10)`
6. `F.max_pool2d(out, 2)`: `(bs, 16, 5, 5)`
7. `nn.Linear(16*5*5, 120)`: `(bs, 120)`

## 有用函数
```
# tools/common_tools.py
def set_seed(seed=1)
def get_memory_info()
def transform_invert(img_, transform_train)	

# model/lenet.py
def forward(self, x)			# 经过多层处理
def initialize_weights(self)	# nn.Conv2d, nn.BatchNorm2d, nn.Linear
class LeNet2(nn.Module)			# features, classifier

l = torch.tensor([0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 0])
len(l)							# 16
l.size()						# torch.Size([16])
l.size(0)						# 16
a = l.size()
a[0]
l.shape							# torch.Size([16])
l.shape[0]						# 16
l.shape(0)						# TypeError: 'torch.Size' object is not callable

a = torch.tensor([False,  True, False, False, False,  True,  True, False, False, False,
         True,  True, False, False,  True, False])		# False=0; True=1
a.sum()							# tensor(6)
a.sum().numpy()					# array(6, dtype=int64)

p = [0, 1]
rmb = 1 if p[0] == 0 else 100	# C++ # rmb = (p[0] == 0) ? 1 : 100
```

## 模型创建与 nn.Module
1. 模型创建: 构建网络层(卷积层，池化层，激活函数层等), 拼接网络层(LeNet, AlexNet, ResNet)
2. 权值初始化: Xavier, Kaiming, 均匀分布，正态分布等
3. 模型构建两要素: `__init__()`, `forward()`=拼接子模块
4. `torch.nn`: nn.Parameter, nn.Module, nn.functional()=(卷积，池化，激活函数等), nn.init()=(参数初始化方法)
5. nn.Module(8 个字典管理它的属性): parameters, modules, buffers, ***_hooks(共5个钩子)

## 模型容器与 AlexNet 构建
1. 容器: nn.Sequetial, nn.ModuleList=(可以增加模型，自定义每个层之间的连接逻辑), nn.ModuleDict
2. 两个阶段: features, classifier
3. nn.Sequential 顺序性 ，各网络层之间严格按顺序执行，常用于 block 构建
4. nn.ModuleList 迭代性 ，常用于大量重复网构建，通过 for 循环实现重复构建
5. nn.ModuleDict 索引性 ，常用于可选择的网络层
6. AlexNet特点
	1. 采用 ReLU ：替换饱和激活函数，减轻梯度消失
	2. 采用 LRN(Local Response Normalization)：对数据归一化，减轻梯度消失
	3. Dropout ：提高全连接层的鲁棒性，防止过拟合, 增加网络的泛化能力
	4. Data Augmentation TenCrop ，色彩修改
	5. 5个卷积层, 3个池化层, 3个全连接层
	6. 卷积后的维度: `height = (input - kernel) / stride + 1`
	7. 卷积后的维度: `height = (input - dilation * (kernel - 1) + 2 * padding -1 ) / stride + 1`
	8. 池化层后的维度: `height = (input - pool) / stride + 1`
	9. 卷积和池化都是滑动. 卷积核：又称为滤波器，过滤器。
	10. (224*224*3) ->卷积(11*11, s=4)-> (54*54*96) ->ReLu-> ->MaxPooling(3*3, s=2)-> (26*26*96)
	11. ->卷积(5*5, p=2)-> (26*26*256) ->

## Lesson 7: Logistic-Regression-norm.py
1. 在 matplotlib 中，一个图形（figure）可以包含多个子图（subplot），而每个子图又包含坐标轴和绘图元素。
2. `plt.clf()` 的作用是清除当前图形的内容，即删除所有子图和坐标轴，使图形回到初始状态
3. `mask = y_pred.ge(0.5).float().squeeze()`
4. 介绍`lr_net.features`和`卷积核`的关系???
5. 提取权重和偏置参数, 对输入进行计算
``` 
# 划分是一条直线: `y = wx + b`, 和`权重` `偏置`对应关系？

w0, w1 = lr_net.features.weight[0]
w0, w1 = float(w0.item()), float(w1.item())

plot_b = float(lr_net.features.bias[0].item())

plot_x = np.arange(-6, 6, 0.1)
plot_y = (-w0 * plot_x - plot_b) / w1
```

## Lesson 8: transforms_methods_1.py
1. `from torch.utils.data import DataLoader`
2. `import torchvision.transforms as transforms`
3. `sys.path.append(hello_pytorch_DIR)`
4. 把图象变成张量: `transforms.ToTensor()`
5. 把张量变成图象: `def transform_invert(img_, transform_train) -> Image`
6. `img_tensor = inputs[0, ...]`去掉第0维的数据
7. 
```
plt.imshow(img)
plt.show()
plt.pause(0.5)
plt.close()
```

## Lesson 9: my_transforms.py, RMB_data_augmentation.py
1. `Dataset.__getitem__()`使用`transform`: `train_data = RMBDataset(data_dir=train_dir, transform=train_transform)`
2. `torch.max(input, dim, keepdim=False, out=None)`: 返回一个元组 (values, indices)，其中 values 是沿着指定维度的最大值张量，indices 是相应的索引张量
3. 
```
valid_transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])
```

## Lesson 11: module_containers.py
1. `x.view()` 是 PyTorch 中用于改变张量形状的方法
2. `x.view( x.size()[0], -1 )`: 第一个维度保持不变（即 x.size()[0]），而第二个维度被设置为 -1。当某个维度的大小被设置为 -1 时，PyTorch 会根据张量的总元素数量自动计算该维度的大小。这种做法常用于将多维张量展平为二维，以便输入全连接层等操作
3. `nn.ReLU(inplace=True)`
4. features维度变化: (4, 3, 32, 32) --Conv2d-> (6, 28, 28) --MaxPool2d-> (6, 14, 14) --Conv2d-> (16, 10, 10) --MaxPool2d-> (16, 5, 5)
5. classifier维度变化: (16, 5, 5) -> 120 -> 84 -> 2
6. `nn.Sequential()`
7. 
```
nn.Sequential(OrderedDict({
            'conv1': nn.Conv2d(3, 6, 5),
            'relu1': nn.ReLU(inplace=True),
            'pool1': nn.MaxPool2d(kernel_size=2, stride=2),
        }))
```
8. `ModuleDict`没有`OrderedDict`, 主要用于外部选择处理
```
nn.ModuleDict({
            'conv': nn.Conv2d(10, 10, 3),
            'pool': nn.MaxPool2d(3)
        })
```
9. `torchvision.models.AlexNet()`
 
## 卷积层: 03-03-ppt-nn网络层-卷积层.pdf; Code: lesson 12; Video:【第三周】nn网络层-卷积层
1. 卷积核学习: 边缘，条纹，色彩
2. `nn.Conv2d( in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')`
3. 在同一次卷积计算中，不同通道的卷积是独立进行的，每个通道都有自己的卷积核（权重）和对应的偏置。允许网络学习并捕捉不同特征层次的信息。
4. `weight`: shape=[2, 3, 3, 3] = [输出通道数, 输入通道数, 卷积核w, 卷积核h]; 
5. 为什么三维卷积核实现二维卷积？每维卷积核只在一个通道上滑动，三维卷积结果相加，再加上偏执，得到一个结果数; 17:21/27:24
6. 转置卷积(Transpose Convolution); 部分跨越卷积 (Fractionally-strided Convolution), 用于对图像进行上采样 (UpSample). 卷积核形状上是转置关系, 但是值不相同
7. 卷积计算转成矩阵运算: 都变成一维列向量，Out = kernel * Input, 18:33
8. `nn.ConvTranspose2d (in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros')`
9. 转置卷积尺寸计算：Out=(in−1) * stride + kernel; 公式是相逆的
10. 转置卷积完整版：H = (in-1) * stride - 2 * padding + dilation * (kernel-1) + padding + 1
11. 转置卷积stride>1, 实现特征图的放大; 棋盘效应
12. 使用场景: 卷积：用于特征提取，具有平移不变性，常见于卷积神经网络的卷积层; 转置卷积：用于上采样或反卷积，可以实现特征图的放大，常见于生成对抗网络（GANs）、图像分割等任务中

## 池化Pooling、线性Linear、激活函数层Activation
1. “收集”：多变少; “总结”：最大值/平均值
2. 卷积和池化都是滑动. 池化为了减维，stride>1
3. 上采样是一种用于增加图像或特征图的空间分辨率的技术; 转置卷积, 双线性插值, 最近邻插值; 语义分割, 生成对抗网络（GANs）, 目标检测, 图像超分辨率
4. 下/降采样是指降低信号或图像的采样率，从而减小数据的规模; stride>1卷积和池化; 降低计算成本, 减少内存占用, 抽取更抽象的特征, 防止过拟合
5. `nn.MaxPool2d( kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`
6. `nn.AvgPool2d( kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None`
7. `nn.MaxUnpool2d( kernel_size, stride=None, padding=0)`, `forward(self, input , indices, output_size=None)`, 11:59/30:28
8. 线性层又称全连接层，out = input * W; 和卷积矩阵运算相反
9. 线性层的shape: 入度，神经元的个数???
10. `nn.Linear( in_features , out_features , bias=True)`, `y = xW^T + bias`
11. n层线性变换相当于1层，所以必须引入激活函数(非线性变换)，赋予深度的意义
12. `nn.Sigmoid`: 符合概率; 导数范围是 (0, 0.25),易导致梯度消失; 非 0均值，破坏数据分布
13. `nn.tanh`: 符合 0 均值; 导数范围是 (0, 1), 易导致梯度消失
14. `nn.ReLU`: 输出值均为正数，负半轴导致死神经元; 导数是 1, 缓解梯度消失，不易引发梯度爆炸
15. `nn.LeakyReLU`: 负半轴很小斜率
16. `nn.PReLU`: 负半轴可学习斜率
17. `nn.RReLU`: 负半轴随机均匀分布
14. gradient梯度, derivative衍生物

## 14. 权值初始化: 04-01-ppt-权值初始化.pdf; 【第四周】权值初始化
1. 下一层的梯度的乘法因子是上一层的输出
2. 经过一层网络，输出方差扩大n倍(n个神经元)，标准差扩大sqrt(n)
3. 解决办法: D(w) = 1/n, 只能权值初始化(考虑前／后向，激活函数)
4. Xavier: (1) 方差1, (2) 激活函数: sigmoid, tanh
5. w = [-sqrt(6)/sqrt(n[i] + n[i+1]), sqrt(6)/sqrt(n[i] + n[i+1])]
6. std(w) = 
7. 十种初始化方法: Xavier均匀/正态分布; Kaiming均匀/正态分布; 均匀/正态/常数分布; 正交／单位／稀疏矩阵初始化
8. `nn.init.calculate_gain( nonlinearity, param= None )` 计算激活函数的方差变化尺度

## 15. 损失函数1
1. 损失函数=单个样本; 代价函数=全部样本的平均; 计算两个概率分布的差值
2. 目标函数 Obj = Cost + Regularization; (Regularization 防止过拟合)
3. 交叉熵计算`nn.CrossEntropyLoss ( weight=None, size_average=None, ignore_index=-100, reduce=None, reduction='mean')`
4. 交叉熵 = 信息熵 + 相对熵
5. 实现负对数似然函数中的 负号功能`nn.NLLLoss()`
6. 二分类交叉熵`nn.BCELoss()`
7. 结合Sigmoid 与 二分类交叉熵`nn.BCEWithLogitsLoss()`, 模型中不加`Sigmoid`

## 16. 损失函数2, 04-03-ppt-损失函数(二).pdf, 【第四周】损失函数(二)
1. `nn.CrossEntropyLoss()`
2. `nn.NLLLoss()`
3. `nn.BCELoss()`
4. `nn.BCEWithLogitsLoss`
5. 差的绝对值`nn.L1Loss()`
6. 差的平方`nn.MSELoss()`
7. `nn.SmoothL1Loss()`
8. 泊松分布的负对数似然损失函数`nn.PoissonNLLLoss()`
9. KL 散度，相对熵`nn.KLDivLoss()`
10. 向量之间的相似度`nn.MarginRankingLoss()`
11. `nn.MultiLabelMarginLoss()`
12. `nn.SoftMarginLoss()`
13. SoftMarginLoss 多标签版本`nn.MultiLabelSoftMarginLoss()`
14. 计算多分类的折页损失`nn.MultiMarginLoss()`
15. 三元组损失，人脸验证中常用`nn.TripletMarginLoss()`
16. 两个输入的相似性`nn.HingeEmbeddingLoss`
17. 采用余弦相似度计算两个输入的相似性`nn.CosineEmbeddingLoss()`
18. `nn.CTCLoss()`
19. 函数参数传递可以分两次
```
loss_f_mse = nn.MSELoss(reduction='none')
loss_mse = loss_f_mse(inputs, target)
```

## 17. 优化器 Optimizer, 04-04-ppt-优化器（一）.pdf
1. `tensor.detach()` 创建一个新的张量，与原始张量共享相同的数据，但不再追踪梯度信息的方法
2. `Optimizer.state`：自己的参数
3. `Optimizer.params_groups`：管理的参数组
4. `zero_grad()`: 清空所管理参数的梯度, 张量梯度不自动清零
5. `step()` 执行一步更新
6. `add_param_group()` 添加参数组
7. `state_dict()`：获取优化器当前状态信息字典
8. `torch.save(optimizer.state_dict(), "optimizer_state_dict.pkl")`
8. `load_state_dict()`：加载状态信息字典
```
state_dict = torch.load("optimizer_state_dict.pkl")
optimizer.load_state_dict(state_dict)
```

## 18. 优化器 Optimizer2, 【第四周】torch.optim.SGD
1. 梯度下降, 学习率控制更新的步伐. w[i+1] = w[i] - LR * g(w[i])
2. 学习率太大，容易发散。太小，收敛速度太慢。实践选用小的值
3. Momentum（动量，冲量）：结合当前梯度与上一次更新信息，用于当前更新
4. 指数加权平均：`V[t] = B * V[t-1] + (1-B) * O[t]`;  `SUM{(1-B) * B^i * O[n-i]}`
5. B意义: 记忆因子。越大记忆时间越长; 越小记忆时间越短。 
6. B通常取0.9， 1/(1-0.9)=10，记忆10天的信息
7. pytorch中更新公式：`V[i] = M * V[i-1] + g(W[i); W[i+1] = W[i] - LR * V[i]`
8. pytorch中更新公式展开：SUM{g(W[i) * M^(n-i)}
9. 随机梯度下降法: `optim.SGD(params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False)`
10. 10种优化器

## 19. 05-01-ppt-学习率调整策略.pdf; 【第五周】学习率调整策略
1. `class _LRScheduler`
2. `step()` 更新下一个epoch的学习率
3. `get_lr()` 虚函数
4. 等间隔调整`lr = lr * gamma`:  `lr_scheduler.StepLR(optimizer, step_size, gamma=0.1, last_epoch=-1)`
5. 给定间隔调整`lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)`, `milestones= [50, 125, 160]`
6. 按指数衰减调整`lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)`
7. 余弦周期调整`lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min =0, last_epoch=-1)`, 最后没有衰减，意义？
8. 当指标不再变化则调整`lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=False, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-8)`
9. 自定义调整策略`lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch=-1)`. 例子有两个`lr_lambda`，学习率怎么使用?
10. 有序调整：Step、MultiStep、Exponential 和 CosineAnnealing
11. 自适应调整：ReduceLROnPleateau
12. 自定义调整：Lambda
13. 初始化：(1)设置较小数 (2) 搜索最大学习率
14. 每个`epoch`只调整一次，在`iteration`不调整

## 20. 【第五周】TensorBoard简介与安装; 05-02-ppt-TensorBoard简介与安装.pdf
1. `pip install tensorboard`. If No module named 'past', `pip install future`
2. 子图: main_tag, tag_scalar_dict
```
writer = SummaryWriter(comment='test_tensorboard')
writer.add_scalar('y=2x', x * 2, x)			# tag, Y, X
writer.add_scalars()
writer.close()

tensorboard --logdir=./runs
http://localhost:6006/
```

## 21. 【第五周】TensorBoard使用（一）; 05-03-ppt-TensorBoard使用（一）.pdf
1. `class SummaryWriter`
2. `add_scalar(tag, scalar_value, global_step =None, walltime=None)`, global_step=X
3. `add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None)`, `tag_scalar_dict: key是变量的tag, value是变量的值`
4. `add_histogram(tag, values, global_step=None, bins='tensorflow', walltime=None)`

## 22. 【第五周】TensorBoard使用（二）; 05-04-ppt-TensorBoard使用（二）.pdf
1. `import torchvision.utils as vutils`, `from torch.utils.tensorboard import SummaryWriter`
2. 图片重叠 `add_image(tag, img_tensor, global_step =None, walltime=None, dataformats ='CHW')`
3. 图片网格 `make_grid(tensor, nrow =8, padding=2, normalize=False, range=None, scale_each =False, pad_value=0)`, `add_image()`
4. 模型计算图 `add_graph( model, input_to_model=None, verbose=False)`
5. 查看模型信息 `summary(model, input_size, batch_size=-1, device="cuda")`
6. `if isinstance(sub_module, nn.Conv2d)`
7. 卷积核的参数包括权重(Weights), 偏置(Bias). 卷积核的权重是共享的
8. `weights.shape`形状通常表示为 `[out_channels, in_channels, kernel_height, kernel_width]`, out_channels（输出通道数）：这是卷积层中卷积核的数量，也是该层输出的特征图的通道数。每个卷积核对输入三个通道计算结果相加会生成一个输出通道。
9. 全连接层中, 每个连接都有一个权重参数
10. `sub_module.weight`
11. `sub_module.bias`, 每个输出通道都有一个对应的偏置
12. `img_pil = PIL.Image.open(path_img)`, `print(img_pil.mode)`, RGB
13. 参数计算: 卷积核`[6, 3, 5, 5]`(一个卷积核在每个通道都有不同参数), 乘积加上6个bias = 456

## 23. 【第五周】hook函数与CAM可视化; 05-05-ppt-hook函数与CAM可视化.pdf
1. 目的: 不改变主体，提取特征图/改变梯度
2. tensor反向传播 `torch.Tensor.register_hook (hook)`
3. 模块钩子 `torch.nn.Module.register_forward_hook()`
4. `torch.nn.Module.register_forward_pre_hook()`
5. `torch.nn.Module.register_backward_hook()`
6. `handle.remove()`
7. `feature map`卷积层的输出，或输出通道的集合. 每个特征图对应一个卷积核. 形状[BCHW], C卷积核的数量
8. 卷积层的权重（weight）是指卷积核的参数, 形状[out_channels, in_channels, kernel_height, kernel_width]
9. out_channels 是卷积层的输出通道数，表示卷积核的数量。每个卷积核生成一个输出通道，计算结果是卷积层的一个特征图
10. 输出通道数 : 卷积核的数量 : 特征图 = 1 : 1 : 1
11. 输入通道数 : 每个卷积核的参数数量 = 1 : 1
12. CAM：类激活图 class activation map. 把最后输出的特征图和权值相乘取和，看算法根据哪些部分得到分类结果
13. 全局池化（Global Pooling）对于每个通道取值(是一个标量)，最后把结果串联，从而将整个特征图转化为一个一维数组
14. 特征图的通道数 : 全局池化一维数组的长度 = 1 : 1
15. 普通的池化，特征图的一个通道是二维，池化后还是二维，只是降维
16. CAM：缺点需要改变网络结构，重新训练. (只能训练完，看看训练的关注点对否？)
17. Grad-CAM: 改进版，利用梯度作为特征图权重. 通过钩子函数，不用修改模型，不用重训练
18. [PyTorch的hook及其在Grad-CAM中的应用](https://zhuanlan.zhihu.com/p/75894080)
19. ```
fmap_dict = dict()
fmap_dict.setdefault(key_name, list())

fmap_dict[key_name].append(o)

alexnet._modules[n1]._modules[n2].register_forward_hook(hook_func)
```

## 24. 【第六周】正则化之weight_decay;	06-01-ppt-正则化之weight_decay.pdf
1. Regularization：减小方差
2. 误差 = 偏差 + 方差 + 噪声
3. 偏差度量了学习算法的期望预测与真实结果的偏离程度，即刻画了学习算法本身的拟合能力
4. 方差度量了同样大小的训练集的变动所导致的学习性能的变化，即刻画了数据扰动所造成的影响
5. 噪声则表达了在当前任务上任何学习算法所能达到的期望泛化误差的下界
6. 过拟合就是方差大
7. `L1`参数的绝对值之和, 直线 -> 菱形
8. 加上`L1`的解在坐标轴上，另一个参数为0，稀疏解
9. `L2`参数平方和，圆形, weight decay( 权值衰减 )
10. `nn.Module.named_modules()` 方法返回模块及其子模块的迭代器。每个模块都有一个`data`属性和一个 `parameters()`方法，它们分别用于获取模块的非参数数据和可学习参数
11. `for name, layer in nn.Modulenamed.named_parameters():`: `nn.Module.named_parameters()` 返回模块中所有可学习参数的迭代器. 值有:`linears.0.weight`, `linears.0.bias`, 每个再分两个:
12. `layer.data` 参数的实际值
13. `layer.grad` 参数的梯度

## 25. 【第六周】正则化之Dropout; 06-02-ppt-正则化-Dropout.pdf
1. Dropout：神经元随机失活, 跟它相关的weight = 0, 防止对某一个神经元过度依赖，防止过拟合
2. 由于训练时失活，数据大小缩小，所以测试时也要缩小, w *= (1-p)。`torch`为了避免缩小测试数据，在训练时放大训练数据
3. `torch.nn.Dropout(p=0.5, inplace =False)`
4. 放在需要dropout层的前面，但是在最后输出层一般不加
5. 类似`L2`收缩数据尺度，减少权重的方差
6. 开始训练模式`net.train()`; 开始测试模式`net.eval()`

## 26. 【第六周】Batch Normalization; 06-03-Batch Normalization.pdf
1. Batch Normalization ：批标准化, 0 均值， 1 方差
2. 优点：
	1.可以用更大学习率，加速模型收敛
	2.可以不用精心设计权值初始化
	3.可以不用dropout或较小的dropout
	4.可以不用L2或者较小的 weight decay
	5.可以不用LRN(local response normalization)
3. `Internal Covariate shift (ICS)` 数据尺度分布变化
	1. 求均值
	2. 求标准差
	3. 归一化
	4. 可学习的逆归一化: 仿射变换=放缩+平移, (r, B) = (weight, bias)
4. 最后层的数据尺度 = 每一层数据尺度 相乘
5. 必须在`relu`前使用BN, 因为`relu`会改变数据分布
6. ```
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):				# bias都是清0
                nn.init.xavier_normal_(m.weight.data)	# 卷积层初始化: 卷积核
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):			# BN层初始化: 参数置1
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):				# 线性连接层: 参数标准正态分布
                nn.init.normal_(m.weight.data, 0, 1)
                m.bias.data.zero_()
```
7. 基类 `_BatchNorm__(num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True)`
8. 派生类 `nn.BatchNorm1d`, `nn.BatchNorm2d`, `nn.BatchNorm3d`
9. `num_features`：一个样本特征数量(最重要). `affine`是否学习3.4(r, B), `track_running_stats`：是训练状态，还是测试状态
10. running_mean：经过动量计算后均值; running_var：动量后方差; weight: affine transform 中的 gamma; bias affine transform 中的 beta
11. 卷积后的特征数量 = 通道数 = 卷积核的数量 
12. input= B * 特征数 * 1/2/3d特征
13. 4个`shape`相同(=卷积核的数量): running_mean, running_var, weight, bias; 所以BN层可以随便插入或移出
14. PPT中input.shape: (1)=3*5*1; (2)=3*3*2*2; (3)=3*4*2*2*3
15. 求均值和方差是在同一个特征上，对batch size个数据求出一个标量

## 27. 【第六周】Normalizaiton_layers; 06-04-ppt-Normalizaiton_layers.pdf
1. Internal Covariate Shift (ICS)：数据尺度/分布异常，导致训练困难
2. `𝐃(H𝟏)=𝒏∗𝑫(𝑿)∗𝑫(𝑾)`; 参数纬度是n, x是输入，w是参数; 梯度的消失／爆炸
3. 概括: 减均值， 除标准差，乘r，加B
4. 其它的正则化, 差异: 均值和方差求取方式
	1. Batch Normalization: 求均值和方差在 一行
	2. Layer Normalization: 求均值和方差在 一列
	3. Instance Normalization: 一列中每个通道单独计算
	4. Group Normalization: 一列中多个通道成组计算
5. LN: BN不适用于变长的网络，逐层计算均值和方差, 不再有running_mean和running_var, gamma 和 beta 为逐元素的
6. `nn.LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True)`
7. ln.weight.shape
	1. elementwise_affine=True, 和输入的shape一样
	2. elementwise_affine=False, 标量
	3. 可以从输入的shape后面向前取一部分
8. IN: BN 在图像生成（ Image Generation ）(风格迁移)中不适用; 逐 Instance channel 计算均值和方差
9. `nn.InstanceNorm2d(num_features, eps=1e-05, momentum=0.1, affine=False, track_running_stats=False)`
10. GN: 应用场景：大模型（特征图多, 小batch size ）; BN 估计的值不准; 通道来凑; 不再有 running_mean 和 running_var; gamma 和 beta 为逐通道（ channel ）
11. `nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True)`, 通道能被组数整除
12. gn.weight.shape = 输入通道数

## 28. 【第七周】模型保存与加载; 07-01-ppt-模型保存与加载.pdf
1. `torch.save()`; 保存整个 Module, 保存模型参数; `torch.save(net, path_model)`, `torch.save(net.state_dict(), path_state_dict)`
2. `torch.load()`; `net_load = torch.load(path_model)`,  `net_new.load_state_dict(torch.load(path_state_dict))`
3. 断点续训练: ```
checkpoint = {
	"model_state_dict ": net.state_dict
	"optimizer_state_dict": optimizer.state_dict
	"epoch": epoch
}

torch.save(checkpoint, path_checkpoint)

checkpoint = torch.load(path_checkpoint)
net.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
start_epoch = checkpoint['epoch']
scheduler.last_epoch = start_epoch

for epoch in range(start_epoch + 1, MAX_EPOCH):
```

## 29. 【第七周】模型finetune; 07-02-ppt-模型finetune.pdf
1. TransferLearning ：迁移
2. Model Finetune: 模型微调; 参数=知识; features extractor不变，classifier改变
3. 重新训练耗时长; 而且新的数据集一般比较小，效果不好
4. 模型微调步骤：
	1. 获取预训练模型参数
	2. 加载模型: load_state_dict(), 全链接层参数是否也加载？
	3. 修改输出层
5. 模型微调训练方法：
	1. 固定预训练的参数: requires_grad =False 或 lr =0
	2. Features Extractor 较小学习率 ( params_group )
6. Resnet-18: 数据 https://download.pytorch.org/tutorial/hymenoptera_data.zip
7. 模型 https://download.pytorch.org/models/resnet18-5c106cde.pth
8. 加载参数
9. 替换fc层, 先得出输入通道: `num_ftrs = resnet18_ft.fc.in_features      resnet18_ft.fc = nn.Linear(num_ftrs, classes)`
10. 冻结卷积层因为新数据量很小，为了补足FC参数(新增加)，需要训练多次，容易导致过拟合`for param in resnet18_ft.parameters(): param.requires_grad = False`
11. conv 小学习率: ```
fc_params_id = list(map(id, resnet18_ft.fc.parameters()))     # 返回的是fc parameters的 内存地址
base_params = filter(lambda p: id(p) not in fc_params_id, resnet18_ft.parameters())
optimizer = optim.SGD([
    {'params': base_params, 'lr': LR*0},   				# 怎么知道两个参数组，一个为卷积，一个为FC
    {'params': resnet18_ft.fc.parameters(), 'lr': LR}], momentum=0.9)
```
12. GPU ```
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet18_ft.to(device)									# 模型送到GPU上
inputs, labels = inputs.to(device), labels.to(device)	# 数据送到GPU上
```

## 30 【第七周】GPU的使用; 07-03-ppt-GPU的使用.pdf
1. CPU 控制单元多，GPU 运算单元多
2. 转换数据类型 /设备: `data.to("cuda")`, `model.to(torch.device("cpu"))`; 张量不执行inplace，模型执行inplace
3. 模型转到`GPU`上，模型执行inplace, 但是地址为什么不变? 因为 inplace 操作修改了 GPU 上的存储，而模型本身的结构仍然存储在 CPU 内存中
4. 逻辑gpu(py可见) <= 物理gpu
5. 常用方法 ```
torch.cuda.device_count()
torch.cuda.get_device_name()
torch.cuda.manual_seed_all()
os.environ.setdefault ("CUDA_VISIBLE_DEVICES", "2,3")
```
6. 主GPU(0): 分发 -> 并行运算 -> 结果回收. 在一个batch中分发
7. `torch.nn.DataParallel(module, device_ids=None, output_device=None, dim=0)`
8. GPU内存排序 ```
def get_gpu_memory():
	import platform
    if 'Windows' != platform.system():
		import os
		os.system('nvidia-smi -q -d Memory | grep -A4 GPU | grep Free > tmp.txt')
		memory_gpu = [int(x.split()[2]) for x in open('tmp.txt', 'r').readlines()]
		os.system('rm tmp.txt')
		return memory_gpu
	else:
        print("显存计算功能暂不支持windows操作系统")
		return False
			
gpu_memory = get_gpu_memory()
if gpu_memory:
	gpu_list = np.argsort(gpu_memory)[::-1]
	gpu_list_str = ','.join( map(str, gpu_list))
	os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpu_list_str)

	print("\ngpu free memory: {}".format( gpu_memory ))
	print("CUDA_VISIBLE_DEVICES :{}".format(os.environ ["CUDA_VISIBLE_DEVICES"]))
```
9. 模型参数存是否在gpu上, `torch.load( path_state_dict , map_location= "cpu")`
10. 多GPU训练，参数前面多了'module.' ```
from collections import OrderedDict
new_state_dict = OrderedDict
for k, v in state_dict_load.items():
	namekey= k[7:] if k.startswith ('module.') else k
	new_state_dict[namekey] = v
```

## 31 【第七周】PyTorch常见报错; 07-04-ppt-PyTorch常见报错.pdf
1. [常见报错](https://shimo.im/docs/PvgHytYygPVGJ8Hv/)
2. `dataparallel`，所有 module 都增加一个属性 `module`
3. 加载保存的网络模型前, 必须定义网络模型
4. 标签从 0 开始, 而不是从 1 开始
5. 查看数据在哪里`data.device`

## 32 【第八周】图像分类一瞥; 08-01-ppt-图像分类一瞥.pdf
1. Inference 推理. (1)`resnet18.eval()`, (2)`with torch.no_grad():`, (3)数据预处理需保持一致，
2. 模型库`anaconda3\envs\pytorchTest\Lib\site-packages\torchvision\models`
3. 经典模型: alexnet, densenet, googlenet, inception, resnet, vgg, mnasnet
4. 轻量模型: mobilenet, shufflenetv2, squeezenet
5. ResNet: 原始X前向传播，减轻梯度消失，增加网络层次, BasicBlock
6. 一开始输入图象: 224 * 224; 处理流程`forward(self, x: Tensor)`
7. 3 * 3, 64: 卷积核是3*3，共64个核

## 33 【第八周】图像分割一瞥; 08-02-ppt-图像分割一瞥.pdf
1. 图像分割(Image Segmentation)：将图像每一个像素分类. 图像分类：将图像给一个分类
2. 图像分割分类：超像素分割(预处理), 语义分割(默认), 实例分割(没有背景), 全景分割(语义分割 + 实例分割)
3. (C, W, H) -> (分类的类别, W, H)
4. `model = torch.hub.load( github , model, *args , **kwargs)`
5. `torch.hub.list( github , force_reload =False)`
6. `torch.hub.help( github , model, force_reload =False)`
7. 得到预训练的模型`https://pytorch.org/hub/`
8. 像素间不是独立的，考虑了关键部位
9. FCN: 全卷积, 去掉了全链接层，输入图象大小可变
10. UNet: 1*572*572 -> 2*388*388, 层间拷贝
11. DeepLab V1: (1)孔洞卷积，增大感受野 (2) 采用 CRF条件随机场(Conditional Random Field) 进行 mask 后处理
12. DeepLab V2: 卷积核大小不变，但strike变化。卷积空洞空间金字塔池ASPP(Atrous spatial pyramid pooling ）：解决多尺度问题
13. DeepLab V3: (1) 孔洞卷积的串行 (2) ASPP 的并行
	1. Image Pyramid: 多个感受野/strike的结果，最后融合
	2. Encoder-Decoder: UNet
	3. Deeper w. Atrous Convolution: 经过空洞卷积后，图象分辨率缩小变慢
	4. Spatial Pyramid Pooling:
14. DeepLab V3+: deeplabv3 基础上加上 Encoder Decoder 思想
15. 综述: 《Deep Semantic Segmentation of Natural and Medical Images: A Review 》 2019
16. 实现人像抠图 (Portrait Matting)

## 34 【第八周】图像目标检测一瞥（上）（下）; 08-03-ppt-图像目标检测一瞥.pdf
1. 背景是p0
2. 两要素/输出: (1) 分类; (2)回归边界框
3. 识别出来物体数量的确定:
	1. 传统方法: 滑动窗; 缺点：(1) 重复计算量大, (2) 窗口大小难确定
	2. 利用卷积减少重复计算: 特征图一个像素 对应 原图 一块区域
4. 两个阶段多出一个推荐框Proposal Rect; 一阶段直接把特征图分成N*N网格
5. 一个阶段算法: YOHO, SSD, Retina-Net, 
6. 两个阶段算法: RCNN, Fast RCNN, Pyramid Network. 区域建议 + 对象检测
7. Faster RCNN 数据流分四个阶段:
	1. Feature map
	2. RPN区域提议网络（Region Proposal Network）: (1) 2 Softmax (背景和前景); (2) Regressors调整候选框 (3) NMS OUT: 非极大值抑制(Non-Maximum Suppression)-从多个重叠的边界框（bounding boxes）中选择最佳的边界框
	3. ROI Layer (Region of Interest Layer): 提取这些候选区域的特征
	4. (1)FC1 FC2; (2) c+1 Softmax; (3) Regressors:
8. `fasterrcnn_resnet50_fpn`输入是图象列表，代替batch
9. `backbone`使用`resnet`, 5个变量; 1 shape=B*256*168*336; 2 shape=B*256*84*168; 减半
10. anchor 是一系列预定义的矩形框（或称为候选框），它们用于在图像中定位和预测潜在的目标对象
11. `RPNHead`: feature对进行分类logits（前景／背景）和bbox回归
12. shape变化: Page 13
13. `NMS`: `filter_proposals()` , `rpn.py`
14. `roi_heads`: `MultiScaleRoIAlign:box_roi_pool()`把不同尺度的特征图统一7*7, `box_head()`, `box_predictor()`
15. Faster RCNN 主要组件(同7)
	1. backbone
	2. RPN
	3. NMS(filter_proposals)
	4. roi_heads
16. 行人检测-finetune, [目标检测推荐github](https://github.com/amusi/awesome-object-detection)

## 35 【第九周】生成对抗网络一瞥; 09-01-ppt-生成对抗网络一瞥.pdf
1. GAN(Generative Adversarial Nets): 生成对抗网络-一种可以生成 `特定分布数据` 的模型
2. `Generator()`生成器(输入噪音)，, `Discriminator()`判别器(二分类)
3. 训练目的: (1)对于D：对真样本输出高概率, (2)对于G：输出使D会给出高概率的数据
4. 算法: 损失函数，对D是最大梯度上升，对G是最小梯度下降
5. 监督学习训练模式: 训练数据 -> 模型   -> 输出值 -> 损失函数(标签)  -> 差异值
6. GAN训练模式:      随机数   -> G模型 -> 输出值  -> D模型(训练数据) -> 差异值
7. 都是卷积模型, G是一个放大过程(wh放大, C缩小到3)，D是一个缩小过程，
8. G:(100*1*1) -> (3*64*64); D: (3*64*64) ->(1) 
9. 人脸五个关键点: 眼睛2，鼻子1，嘴巴2
10. 提高质量措施: (1)标签平滑， (2)特征数量， (3)训练数据
11. DCGAN
12. [GAN的应用](https://jonathan-hui.medium.com/gan-some-cool-applications-of-gans-4c9ecca35900)
13. [GAN推荐 github](https://github.com/nightrome/really-awesome-gan)
14. `gan_demo.py`: 真实数据(标签通常设为 1), 假数据（标签通常设为 0）; 引入标签平滑（label smoothing）
15. 判别器分别处理真数据和假数据，两次反向传播
16. 生成器输入的是随机数，损失函数是和真实标签(1)相比
17. 生成器的目标是让判别器认为假数据是真实的，因此使用与真实数据相同的标签. 而不是判别器的输出（为了简单?）
18. 为什么生成器的标签不是和判别器的输出相关? 例如(1-判别器的输出)

## 36 【第九周】循环神经网络一瞥; 09-02-ppt-循环神经网络一瞥.pdf
1. RNN **循环**神经网络(Recurrent Neural Networks), 不定长输入
2. 常用于 NLP 及时间序列任务（输入数据具有 前后关系)
3. RNN网络结构: 神经元的输出S[t] = F(U * X[t] + W * S[t-1]);   最终的输出O[t] = SoftMax(V * S[t])
4. 实现: 循环，最后一个循环作为模型的输出
```
    for i in range(line_tensor.size()[0]):		# 单词中的每个字符
        output, hidden = rnn(line_tensor[i], hidden)		# y, h = model( [0, 0, …, 1, …, 0], h)
```
5. 字符总量: 57 = 26大写 + 26小写 + " .,;'"
6. 输入: 字符变成向量, 正交基，不是ASCII值
7. 输出: 18分类
8. 如何实现不定长字符串 到 分类向量 的映射？ 循环 + 全链接层 + 激活函数
9. `rnn_demo.py`: 训练数据随机: 文件名随机，里面行随机
10. 每个字符转成字符向量(一维长度为57 tensor). 一个词转成三维向量: [名字第几个字符][长度为1的1维][字符向量]
11. 图象是4维向量: BCHW. 而这里是3维
12. `loss = criterion(output, category_tensor)`, output.shape=[1, 18], category_tensor.shape=1

## 作业
### 【第一周】[作业讲解1](https://www.jianshu.com/p/5ae644748f21)
1. Scalar 标量，0维; Vector 一维数组; Matrix 二维矩阵; Tensor 张量
2. `grad_fn`是指向函数的指针，自动求导的关键
3. tensor, ndarray共享同一个数据内存
4. `torch.normal()`, broadcast机制

### 【第一周】[作业讲解2](https://www.jianshu.com/p/cbce2dd60120)
1. 线性回归: 使用线性函数逼近目标，`Y=K[0] + K[1]X[1] + ... + K[n]X[n] + e`
2. 斜率K不扰动(如果扰动就不能称为线性？)，只是截距X扰动
3. 计算图: 节点是数据，边是计算
4. 动态图，静态图

### 【第一周】作业讲解3
1. 逻辑回归: 二分类问题（binary classification), 实际上是一种广义的线性回归模型(提取特征K，然后使用Sigmoid函数，[0,1])
2. Sigmoid/logistic函数: 1/(1+e^(-z))
3. 线性回归(标量)，逻辑回归(概率)
4. Sigmoid函数当数值很大或很小，梯度消失，无法训练
5. n分类问题常用算法:
	1. 逻辑回归（Logistic Regression）: 一对多
	2. 神经网络（Neural Networks）
	3. 决策树（Decision Trees
	4. 随机森林（Random Forests）
	5. 支持向量机（Support Vector Machines, SVM）: 超平面
	6. K最近邻（K-Nearest Neighbors, KNN）
	7. 梯度提升决策树（Gradient Boosting Decision Trees, GBDT）
	8. 多层感知机（Multilayer Perceptrons, MLP）

### 【第二周】作业讲解
1. Dataset, DataLoader
```
for i, data in enumerate(train_loader):
class RandomSampler(Sampler[int])
Dataset.__getitem__(index)
```
2. [猫狗二分类](https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data)
3. 特征图大小减半 伴随 卷积核增倍
4. 卷积, BatchNorm, Relu (CBR)一起
5. 卷积最后，全链接层前使用`AdaptiveAvgPool`, `AdaptiveMaxPool`.
6. `AdaptiveMaxPool`不需要指定窗口大小或步幅，而是直接指定输出的维度, 处理不同尺寸输入. 对特征图的一个平面(w,h)求Max, 最后把通道Cat起来. 
	1. 好处 (1)大大减少参数个数(平面->2点), (2)正则化(抽象:平均，最大), (3)方便调节网络输入图片的大小
	2. 卷积和全链接的过度，只关心核的个数，跟特征图大小无关
7. 优化: 数据优化（裁剪，映射）; 模型
8. 先写模型类，再写训练类
9. [作业代码](https://github.com/greebear/pytorch-learning)

### 【第三周】作业讲解
1. `Sequential`: 简单的线性流水线. 自动执行子模块的前向传播; 限制是不允许多输入、多输出或需要分支的模型。
2. `ModuleList`: 明确的顺序，通过索引迭代. 需要在父模块的forward方法中手动调用每个子模块, 仅列表容器
3. `ModuleDict`: 通过名称访问，没有明确的顺序
4. `class Module`: 8 个字典管理它的属性: parameters, modules, buffers, ***_hooks(共5个钩子)
5. `class _ConvNd(Module)`: stride, in_channels, out_channels, kernel_size...
6. `class Conv2d(_ConvNd)`: 怎么进行梯度传播`forward()`
7. `class Linear(Module)`: self.weight, self.bias
8. python的联合: typing.Union[int, typing.Tuple[int, int]]
9. `torchvision.models.AlexNet()._modules['features']._modules.keys()`
	1. `AlexNet()._modules`是`module`的字典变量
	2. 'features'是`AlexNet()`的特征提取层: features, avgpool, classifier
	3. 最后的`_modules`是特征提取层的每一个基本步骤
10. `ModuleDict()`加上了名字, 但是失去了顺序`forward()`，所以使用`Sequential(OrderedDict({}))`代替`ModuleDict()`
11. 当修改层时，使用名字把索引方便
12. 转置卷积(Transpose Convolution): 反卷积（Deconvolution）:  将数据的空间维度从较小尺寸映射到较大尺寸, 实现特征图尺寸的扩大(上采样（upsampling）)
13. 转置卷积实现: 插入零填充（Zero Insertion）, 输出填充（Output Padding）
14. 转置卷积应用: 图像分割, 生成对抗网络
15. 反池化(MaxUnpool): 在进行最大池化时，记录最大值的位置, 恢复时其他位置则填充为零
16. 激活函数: ReLu及变体, Sigmoid, Tanh
17. `nn.Conv2d(in_channels, out_channels...)`: out_channels共有几个卷积进行运算，in_channels是每个卷积有几个分片, `conv_layer1.weight.shape`
18. 卷积是对应位置相乘然后相加得到一个元素, 和矩阵乘法第一步一样
	1. 一个卷积核对应输入每个通道都有一个分片, 输出一个特征图, 作为后面输入的一个通道
	2. 通道0和卷积核0分片计算，形成特征图分片0-0
	3. 通道1和卷积核1分片计算，形成特征图分片0-1
	4. 通道2和卷积核2分片计算，形成特征图分片0-2
	4. 输出特征图 = 特征图分片0-0 + 特征图分片0-1 + 特征图分片0-2 + bias
19. 2D卷积用于图片B*C*H*W，3D卷积用于视频B*C*D*H*W

### 【第四周】作业讲解
1. 梯度消失／爆炸: 梯度剪切，权重正则化，激活函数改进，batchnorm, ResNet
2. 梯度剪切: 超过了阈值，缩放到该阈值以下
3. ResNet: 每层学习的是目标映射与输入之间的残差 (差异); 残差块趋向于零; 机制(梯度直接传递, 恒等映射, 简化学习目标)
4. 其它网络学习的是输入 -> 目标，存在梯度消失／爆炸问题
5. 损失函数，优化算法

### 【第五周】作业讲解: 学习率
1. Tensorboard
2. [anaconda安装包](https://anaconda.cloud/)

### 【第六周】作业讲解: 正则化
1. [CS231n: CNN for Visual Recognition.](https://cs231n.github.io/)


### 【第七周】作业讲解: 模型的保存与加载
1. `torch.save()`, `torch.load()`
2. [API DOC](https://pytorch.org/docs/stable/torch.html)
3. 模型微调（Finetune）: 模型微调是迁移学习的一部分; 共享同一个目标，即利用已有的知识和数据来提高新任务的学习效率和性能
4. [Transfer Learning](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html)
5. [Finetuning Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)
6. GPU: `torch.nn.DataParallel`, `torch.cuda`
7. [Distributed and Parallel Training Tutorials](https://pytorch.org/tutorials/distributed/home.html)
8. ssh connect Remote Notebook


### 数据源
1. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
2. [ImageNet](https://image-net.org/)
3. [ImageFolder](https://www.tugraz.at/institute/icg/home)
4. [LSUN Classification](https://www.cs.princeton.edu/)
5. [COCO (Captioning and Detection)](http://mscoco.org/)
6. [kaggle](www.kaggle.com)

## 为什么要使用`1*1`卷积?
1. 降维度和升维度 - 卷积特点
2. 非线性变换, 因为后面应用（如ReLU）
3. 参数共享: 1x1卷积核的使用减少了参数的数量，因为它只在通道之间共享权重，而不是在整个输入图像上。这有助于减小模型的尺寸，降低过拟合的风险，并提高模型的计算效率
4. 计算效率

## Read Image, convert to Tensor
```
from PIL import Image

img = Image.open(path_img).convert('RGB')`

img_transform = transforms.Compose([transforms.ToTensor()])
img_tensor = img_transform(img)		# C*H*W
img_tensor.unsqueeze_(dim=0)    	# C*H*W to B*C*H*W, [1 * 3 * 512 * 512]

# 经过 3*3 卷积
conv_layer = nn.Conv2d(3, 1, 3)     # input:(i, o, size) weights:(o, i , h, w)
nn.init.xavier_normal_(conv_layer.weight.data)
img_conv = conv_layer(img_tensor)   # B*C*H*W [1, 1, 510, 510]

# 转回图象
img_raw  = transform_invert(img_tensor.squeeze(),  img_transform)   # img_tensor.shape=[1, 3, 512, 512]
img_conv = transform_invert(img_conv[0, 0:1, ...], img_transform)   # img_conv.shape  =[1, 1, 510, 510]
```

## slice
```
import numpy as np
img_conv = np.random.random((3, 4, 5))
img_conv.shape		# (3, 4, 5)

result = img_conv[0, ...]
result.shape		# (4, 5)

result = img_conv[0:2, ...]
result.shape		# (2, 4, 5)

result = img_conv[0, 0:1, ...]
result.shape		# (1, 5)

result = img_conv[0, 0:1, :]
result.shape		# (1, 5)
```

## plt
```
plt.cla()   # 防止社区版可视化时模型重叠
plt.plot(x, y, 'r-', lw=5)	# `r-'表示红色（r）的实线（-）	# lw=5: 表示线的宽度5
plt.text(2, 20, 'Loss=%.4f' % loss.data.numpy(), fontdict={'size': 20, 'color': 'red'})
plt.xlim(1.5, 10)
plt.title("title")

plt.show(block=True)
plt.pause(0.5)
```

## `torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)`
1. `out_channels ：输出通道数，等价于卷积核个数`????
2. 卷积核运算次数W方向: `处理次数=(处理前w - 核的w) / stride + 1`
3. 因为每次运算产生一个单元，所以处理后W维数等于卷积核运算次数
4. 卷积维度：一般情况下 ，卷积核在几个维度上滑动，就是几维卷积
5. 1维卷积核是一维, 2维卷积核是二维比如(5*5), 3维卷积核是三维比如(3*3*3)
6. 如果要保持前后维度不变，需要填充`核的w-1`个格子; 四个方向（前后）都填充(因为核是奇数，所以前后填充相等)
7. 孔洞卷积可以看成卷积核扩大，(3*3) -> (5*5)

## 函数参数
```
kernel_size = 3
kernel_size_ = _pair(kernel_size)		# (3, 3)

padding: Union[str, _size_2_t] = 0
padding_ = padding if isinstance(padding, str) else _pair(padding)

padding_mode: str = 'zeros'
```