# torch
deepshare.net
[60AI经典论文](https://deepshare.feishu.cn/docs/doccnewbeOX1q1t4Pk5npjv0p5d)

## env
1. `conda create --name pytorchTest opencv matplotlib -y`
2. `conda activate pytorchTest`
3. `conda install -c conda-forge pytorch-cpu torchvision tensorboard future`
4. `conda install psutil`
5. `tensorboard --logdir=./runs`
6. Install `libjpeg` or `libpng`:
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
3. `nn.MaxPool2d( kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)`


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