# HandWriting Eraser 2.0

这是一个把试卷像素分成 `背景(0) / 手写(1) / 印刷(2)` 三类，并将手写区域擦除为白色的语义分割项目。本次升级的默认模型面向 CPU 推理，同时保留旧 DeepLabV3+ 模型入口，便于对照实验。

## 为什么升级

### 1. 原模型结构的问题

- 训练参数虽然提供 `--model`，代码却始终实例化 `deeplabv3plus_resnet101`，实验配置与实际模型不一致。
- ResNet-101 + 五分支 ASPP 对这个三类文档任务偏重。高分辨率试卷必须切片，CPU 上的延迟和内存占用都较高。
- 输出步长较大，而手写笔画通常只有几个像素；仅依赖深层特征容易漏掉细笔画或模糊手写/印刷边界。
- 语义分割只能判断“当前像素是什么”，无法恢复被手写墨迹完全遮住的印刷内容；旧后处理也只是涂白，并不是真正的内容重建。
- 训练固定把页面拉伸到 `1024×1024`，破坏试卷和字形比例；训练与推理归一化还分别使用了 `[0.5]` 和 ImageNet 参数。
- 训练使用 3 类，两个旧推理脚本却构建 2 类模型；checkpoint 名称、实际 backbone 和类别数可能互相矛盾。

升级后的 `lite_eraser` 使用 MobileNetV2 编码器和轻量上下文门控解码器。stride-4 浅层特征用于保留细笔画，高层全局门控提供页面上下文，深度可分离卷积完成融合。相比 ResNet-101 + ASPP，它更适合 CPU，且训练、验证、普通推理和 TorchScript 导出共用同一模型配置。

| 模型 | 参数量 | FP32 权重 | 本机 CPU 256×256 延迟（4 线程） |
| --- | ---: | ---: | ---: |
| 旧 DeepLabV3+ ResNet-101 | 58.75 M | 224.1 MiB | 约 498 ms |
| 新 `lite_eraser` | 1.92 M | 7.3 MiB | 约 98 ms |

延迟是在当前 x86_64 机器、PyTorch 2.2.2 上预热后测得，只用于量级对比。模型精度仍需在完整数据集上重新训练并以独立验证集评估；仓库内没有附带原始训练图片和标签，因此本次升级不虚构精度结果。

本次仍沿用三类分割，是因为现有数据只有分割标签。它能可靠地“擦除可见手写像素”，但不会凭空补回被遮挡的题目文字。如果业务目标要求恢复遮挡内容，需要另行准备“带手写试卷 / 对应干净试卷”配对数据，增加图像复原分支，或利用同版空白试卷做配准后重建；仅更换分割网络无法解决这个信息缺失问题。

### 2. 原损失函数的问题

- 普通交叉熵会被大量背景像素主导，整体准确率很高也可能漏掉手写区域。
- 原 focal loss 的标量 `alpha=0.25` 统一乘到所有类别，只改变损失尺度，没有改变类别相对权重，因此不能真正解决类别不平衡。
- 只优化逐像素分类，没有直接约束细小前景区域的重叠质量。

默认损失升级为：

```text
loss = class-weighted cross entropy + 0.5 × foreground soft Dice
```

默认类别权重为 `1,4,2`，强调手写类；应根据自己的训练集统计通过 `--class-weights` 调整。focal loss 也已修复，可接收逐类别权重。最佳模型不再按容易被背景抬高的 overall accuracy 或 mean IoU 保存，而按 `Handwriting IoU` 保存。

### 3. 原训练方法的问题

- 图片和标签是两个目录分别排序后按下标配对，文件缺失或扩展名不同可能造成静默错配。
- 直接按文件排序切分数据，容易让相邻生成样本聚集在同一个 split；验证集还开启了 shuffle。
- README 提到重叠裁剪增强，但训练代码实际上只有固定 resize；随机裁剪也可能抽到几乎全背景区域。
- SGD 初始学习率较高、没有 warmup、没有梯度裁剪；GPU 与 CPU checkpoint 依赖 `DataParallel` 包装。

现在按文件 stem 严格配对并检查尺寸，用固定随机种子切分；训练采用比例不变的随机缩放、手写类感知随机裁剪和温和颜色增强；优化器使用 AdamW、backbone 较小学习率、warmup-poly 调度与梯度裁剪。AMP 只在 CUDA 上开启。checkpoint 始终保存未包装模型的权重，可直接映射到 CPU。

## 环境

建议 Python 3.9+：

```bash
python -m pip install -r requirements.txt
```

快速自检：

```bash
python -m unittest tests.test_smoke
```

数据目录结构：

```text
datasets/data/
├── Images/
│   ├── 0001.jpg
│   └── 0002.jpg
└── Labels/
    ├── 0001.png
    └── 0002.png
```

图像与标签扩展名可以不同，但文件 stem 必须一一对应。标签必须是单通道类别 ID：背景 0、手写 1、印刷 2。

### 转换新增数据集

两个转换器都生成相同的标准结构：`Images/`、`Labels/`、`splits/` 和 `dataset.json`。默认输出不会修改原始数据，也不会静默覆盖已有结果。

```bash
python tools/convert_scut_ensexam.py
python tools/convert_signatr6k.py
```

转换被中断时可追加 `--resume`，已经完成且标签合法的样本会直接复用；需要从头重建时使用 `--overwrite`，两者不能同时使用。

默认输出目录：

- `/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format`
- `/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format`

SCUT-EnsExam 的原图和擦除图是对齐图像，转换器在手写四边形内进行成对差分，学生答案和教师批改都映射为手写类；印刷类从擦除图的局部对比度和暗像素提取。相关阈值都可通过 `python tools/convert_scut_ensexam.py --help` 调整。官方 430 张训练页会固定拆出 15% 作为验证集，115 张官方测试页不会进入训练或验证。

SignaTR6K 的颜色映射为：蓝→背景，绿→手写，红→印刷。黄色表示手写与印刷重叠；为了与 Baidu 的互斥三类标签兼容并保证擦除召回率，默认映射为手写。官方 train/validation/test split 原样保留。

## 训练

三个数据集一起训练：

```bash
python main.py \
  --data-root /Users/peng/Documents/data/HandWritingData/baidu \
  --data-root /Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format \
  --data-root /Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format \
  --model lite_eraser \
  --dataset-sampling balanced \
  --batch-size 8 \
  --crop-size 512 \
  --total-itrs 30000
```

`--data-root` 可以重复任意次数。默认 `balanced` 让每个数据集获得相同的抽样概率，避免样本最多的 SignaTR6K 主导训练；`--dataset-sampling proportional` 恢复按样本数混合，`--dataset-weights 2,1,1` 可自定义三个数据源的相对概率。验证时会分别打印每个数据集及总集合的指标，最佳模型按各数据集 `Handwriting IoU` 的宏平均保存，避免高分辨率 SCUT 页面仅凭像素数主导模型选择。

小尺寸 SignaTR6K 图像会先保持比例放大到训练 crop 的最小尺寸，不会在 768/512 crop 周围填充大面积空白。

如果同一原始试卷生成了多个增强样本，建议把同源样本放在同一个 split，避免验证泄漏。可在文本文件中逐行写验证集文件 stem，并传入 `--val-list validation.txt`。

显存不足时先降低 `--batch-size`，再降低 `--crop-size`。恢复完整训练状态：

```bash
python main.py --data-root datasets/data \
  --ckpt checkpoints/latest.pth --continue-training
```

仅验证：

```bash
python main.py --data-root datasets/data \
  --ckpt checkpoints/best.pth --test-only
```

## CPU 推理

新推理程序使用重叠 tile 的 logits 加权融合，不会像旧版覆盖式拼接那样在 tile 边缘产生明显接缝。默认设备就是 CPU：

```bash
python predict.py \
  --input samples/input.jpg \
  --output results/output.png \
  --checkpoint checkpoints/best.pth \
  --device cpu \
  --tile-size 768 \
  --overlap 128
```

目录可作为输入和输出，目录层级会保留。`--save-mask` 可同时保存手写 mask，`--threads` 控制 CPU 线程数。

### 导出 CPU TorchScript

TorchScript 不依赖训练脚本中的 Python 模型定义，适合部署：

```bash
python export_cpu.py \
  --checkpoint checkpoints/best.pth \
  --output handwriting_eraser_cpu.pt

python predict.py \
  --input samples/input.jpg \
  --output results/output.png \
  --checkpoint handwriting_eraser_cpu.pt \
  --device cpu
```

导出时会在 CPU 上比较 eager 与 TorchScript 输出，不一致时拒绝生成模型。

## 旧模型兼容说明

新训练生成的 checkpoint 会自动记录模型配置。旧 checkpoint 没有这些元数据，推理时必须显式指定其真实配置，例如：

```bash
python predict.py --input input.jpg --output output.png \
  --checkpoint old.pth \
  --model deeplabv3plus_resnet50 --num-classes 3 --output-stride 16
```

旧仓库曾出现“文件名写 ResNet-50、训练代码实际建 ResNet-101”和 2/3 类不一致的情况；如果严格加载失败，应以 checkpoint 的权重 shape 为准核对结构，不能用 `strict=False` 掩盖类别头不匹配。
