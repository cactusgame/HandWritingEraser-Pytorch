# HandWriting Eraser 4.0

这是一个面向 4–8 核 CPU 服务端的试卷去手写项目。4.0 默认使用单个 `joint_eraser` 模型，同时完成 `背景/手写/印刷` 分割和被遮挡印刷内容的 RGB 修复；输出使用模型内部 soft mask 融合，未检测为手写的区域严格保留输入像素。旧分割模型仍然保留，便于兼容已有 checkpoint。

## 4.0：一个模型完成检测、擦除和补齐

`joint_eraser` 共享 `server_eraser` 的 ResNet-18 编码器和多尺度解码特征，内部包含两个联合训练的输出头：分割头产生手写概率，修复头预测干净 RGB residual。它们属于同一个 checkpoint、一次前向和一个 ONNX/TorchScript 文件，不是串行调用两个模型。

```text
RGB -> shared encoder/decoder -> segmentation logits
                         └-----> clean RGB candidate
input * (1-soft mask) + candidate * soft mask -> restored RGB
```

修复目标直接使用配对干净图的 RGB，因此可以学习纸张、印刷文字、表格线原本的黑色、灰色或彩色，而不是统一填白。完全被遮挡的信息仍只能根据上下文推断；如果业务能够取得同版空白试卷，模板配准仍然比生成式修复更可靠。

## 为什么升级

### 1. 原模型结构的问题

- 训练参数虽然提供 `--model`，代码却始终实例化 `deeplabv3plus_resnet101`，实验配置与实际模型不一致。
- ResNet-101 + 五分支 ASPP 对这个三类文档任务偏重。高分辨率试卷必须切片，CPU 上的延迟和内存占用都较高。
- 输出步长较大，而手写笔画通常只有几个像素；仅依赖深层特征容易漏掉细笔画或模糊手写/印刷边界。
- 语义分割只能判断“当前像素是什么”，无法恢复被手写墨迹完全遮住的印刷内容；旧后处理也只是涂白，并不是真正的内容重建。
- 训练固定把页面拉伸到 `1024×1024`，破坏试卷和字形比例；训练与推理归一化还分别使用了 `[0.5]` 和 ImageNet 参数。
- 训练使用 3 类，两个旧推理脚本却构建 2 类模型；checkpoint 名称、实际 backbone 和类别数可能互相矛盾。

当前提供三档 CPU 模型：

- `server_eraser`（默认）：ImageNet 预训练 ResNet-18 编码器，使用 stride 32 上下文和 stride 16/8/4/2 逐级解码。它用更多语义容量区分形态相近的手写、印刷批注和表格线，适合 4–8 核服务端。
- `quality_eraser`：MobileNetV2 编码器，同样具有多尺度上下文和 stride 8/4/2 解码，速度和效果居中。
- `lite_eraser`：原 2.0 轻量模型，只有 stride-4 跳连，速度最快，适合资源很紧的场景。

三个模型都只包含标准卷积、深度可分离卷积、池化和双线性插值，可在普通 PyTorch CPU、TorchScript 或 ONNX Runtime CPU 上运行。训练、验证、推理和导出共用 checkpoint 中的模型配置。

| 模型 | 参数量 | FP32 权重 | 本机 CPU 256×256 延迟（4 线程） |
| --- | ---: | ---: | ---: |
| 旧 DeepLabV3+ ResNet-101 | 58.75 M | 224.1 MiB | 约 498 ms |
| 新 `lite_eraser` | 1.92 M | 7.3 MiB | 约 98 ms |
| 新 `quality_eraser` | 1.96 M | 7.5 MiB | 约 206 ms |
| 新 `server_eraser` | 11.52 M | 44.0 MiB | 约 331 ms |
| 新 `joint_eraser` | 11.53 M | 44.0 MiB | 约 419 ms |

延迟是在当前 x86_64 机器、PyTorch 2.2.2 上预热后测得，只用于量级对比。模型精度仍需在完整数据集上重新训练并以独立验证集评估；仓库内没有附带原始训练图片和标签，因此本次升级不虚构精度结果。

本次仍沿用三类分割，是因为现有数据只有分割标签。它能可靠地“擦除可见手写像素”，但不会凭空补回被遮挡的题目文字。如果业务目标要求恢复遮挡内容，需要另行准备“带手写试卷 / 对应干净试卷”配对数据，增加图像复原分支，或利用同版空白试卷做配准后重建；仅更换分割网络无法解决这个信息缺失问题。

### 2. 原损失函数的问题

- 普通交叉熵会被大量背景像素主导，整体准确率很高也可能漏掉手写区域。
- 原 focal loss 的标量 `alpha=0.25` 统一乘到所有类别，只改变损失尺度，没有改变类别相对权重，因此不能真正解决类别不平衡。
- 只优化逐像素分类，没有直接约束细小前景区域的重叠质量。

2.0 的 `weighted CE + Dice` 仍有两个问题：Dice 对漏检和误检一视同仁，无法按业务代价强调“残留手写”；同时它对细笔画边界没有直接监督，擦除后容易留下彩色边缘。3.0 默认损失升级为：

```text
loss = class-weighted CE
     + 0.7 × handwriting Tversky(alpha=0.35, beta=0.65)
     + 0.2 × handwriting boundary BCE
```

`beta > alpha` 让漏擦比误擦受到更大惩罚；边界带 BCE 专门约束笔画内外两侧，减少断笔和边缘残留。默认类别权重为 `1,3,2`：抽样统计中 Baidu/SCUT 的手写像素约占 2.6%–3.1%，但前景裁剪已经显著提高其训练出现率，所以不再用过大的手写权重。最佳 checkpoint 按各数据源 `sqrt(Handwriting IoU × Print IoU)` 的宏平均保存，防止靠误擦印刷内容换取手写召回。

联合模型在上述分割 loss 之外增加 Mask 内 Charbonnier RGB、SSIM、Sobel 边缘和颜色差 loss，并在 Mask 外使用 identity loss。真实或可靠合成的配对样本参与全部修复项；无干净目标样本只参与分割和 Mask 外不变约束，不会把涂白伪结果当成真值。

### 3. 原训练方法的问题

- 图片和标签是两个目录分别排序后按下标配对，文件缺失或扩展名不同可能造成静默错配。
- 直接按文件排序切分数据，容易让相邻生成样本聚集在同一个 split；验证集还开启了 shuffle。
- README 提到重叠裁剪增强，但训练代码实际上只有固定 resize；随机裁剪也可能抽到几乎全背景区域。
- SGD 初始学习率较高、没有 warmup、没有梯度裁剪；GPU 与 CPU checkpoint 依赖 `DataParallel` 包装。

现在按文件 stem 严格配对并检查尺寸，用固定随机种子切分；训练采用比例不变的随机缩放、小角度旋转、颜色变化、模糊/噪声/JPEG 文档退化增强。75% crop 主动寻找手写区域，25% 保持普通随机裁剪，避免模型没见过足够的纯背景/印刷区域而误擦。优化器使用 AdamW、backbone 较小学习率、warmup-poly 调度、梯度裁剪和 EMA；验证与默认部署均使用 EMA 权重，恢复训练则使用同时保存的原始训练权重。AMP 只在 CUDA 上开启。

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

两个转换器都生成标准结构：`Images/`、`Labels/`、`CleanTargets/`、`splits/` 和 `dataset.json`。默认输出不会修改原始数据，也不会静默覆盖已有结果。

```bash
python tools/convert_scut_ensexam.py --resume
python tools/convert_signatr6k.py --resume
```

转换被中断时可追加 `--resume`，已经完成且标签合法的样本会直接复用；需要从头重建时使用 `--overwrite`，两者不能同时使用。

默认输出目录：

- `/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format`
- `/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format`

SCUT-EnsExam 的原图和擦除图是对齐图像，转换器会把擦除图复制到 `CleanTargets/`，作为真实 RGB 修复目标；同时在手写四边形内进行成对差分，生成三分类标签。相关阈值都可通过 `python tools/convert_scut_ensexam.py --help` 调整。完整数据的 430 张训练页会固定拆出 15% 作为验证集，115 张官方测试页不会进入训练或验证。

SignaTR6K 的颜色映射为：蓝→背景，绿→手写，红→印刷。黄色表示手写与印刷重叠；为了与 Baidu 的互斥三类标签兼容并保证擦除召回率，默认映射为手写。官方 train/validation/test split 原样保留。

Baidu 和 SignaTR6K 没有与输入严格对应的干净整图，不能把算法涂白结果当作修复真值。训练时它们始终参与分割监督；另外，数据集会在线寻找不含手写的真实印刷区域，再覆盖来自同一数据源的真实笔迹，构造“合成输入/原始干净区域”配对。这样底层印刷文字和颜色仍是真实像素。`--synthetic-restoration-probability` 控制该增强比例，默认 `0.5`。

当前这台开发机上的三份数据是小型样例副本：Baidu 5 张、SCUT train/test 各 5 张、SignaTR6K 每个 split 5 张。它们只适合验证代码；正式训练必须使用服务器上的完整数据。

## 训练

三个数据集一起训练：

```bash
python main.py \
  --data-root /Users/peng/Documents/data/HandWritingData/baidu \
  --data-root /Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format \
  --data-root /Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format \
  --model joint_eraser \
  --loss joint \
  --output-stride 32 \
  --init-segmentation-ckpt /Users/peng/Documents/models/hw_clear/cpu_v2/best.pth \
  --dataset-sampling balanced \
  --dataset-weights 1,2,1 \
  --batch-size 4 \
  --crop-size 640 \
  --total-itrs 60000 \
  --checkpoint-dir /Users/peng/Documents/models/hw_clear/cpu_v3_joint
```

`cpu_v2/best.pth` 的编码器、分割解码器和分类头会严格加载到联合模型。默认前 2000 iterations 冻结这些权重，只训练修复头；随后自动解冻并用 backbone 小学习率联合微调。`--dataset-weights 1,2,1` 提高真实 SCUT 配对数据的比例。

`--data-root` 可以重复任意次数。验证会记录分割指标以及真实配对样本上的 `Restoration Masked MAE/PSNR/Fidelity`。最佳模型按 `sqrt(macro Erase Quality × Restoration Fidelity)` 保存，同时约束手写检测、印刷内容保留和原色修复。

验证指标也会写入本地 TensorBoard event 文件，默认目录是 `<checkpoint-dir>/runs`。只会写本地文件，不会上传到网络。查看方式：

```bash
tensorboard --logdir checkpoints/runs
```

需要换目录可传 `--tensorboard-dir /path/to/runs`；需要完全关闭本地日志可传 `--no-tensorboard`。

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

推理程序会同时对分割 logits 和最终修复 RGB 做重叠 tile 加权融合，避免 tile 边缘出现颜色接缝。默认设备就是 CPU：

```bash
python predict.py \
  --input samples/input.jpg \
  --output results/output.png \
  --checkpoint checkpoints/best.pth \
  --device cpu \
  --threads 4 \
  --tile-size 512 \
  --overlap 96
```

目录可作为输入和输出，目录层级会保留。`--save-mask` 可同时保存手写 mask，`--threads` 控制 CPU 线程数。

推理默认启用不需要重新训练的 `balanced` 后处理：

- 仅在手写 mask 内，把与邻域纸张颜色明显冲突的纯白输出替换为局部背景；
- 从原图保留与印刷类别相连的深色低彩度像素；
- 沿水平、垂直和两个对角方向连接最长 7 像素的短印刷断点。

它不会修改 mask 外的像素，也不会对整页文字执行闭运算。可以保存中间结果检查每次修改：

```bash
python predict.py \
  --input samples/input.jpg \
  --output results/output.png \
  --checkpoint checkpoints/best.pth \
  --postprocess balanced \
  --postprocess-max-gap 7 \
  --save-mask \
  --save-postprocess-debug
```

调试文件包括 `_background.png`、`_background_repair.png`、
`_protected_print.png` 和 `_bridged_print.png`。如果原稿使用黑色手写且出现误保留，
可把 `--postprocess-max-gap` 降为 `3`，或使用
`--postprocess background` 只修纸张颜色；`--postprocess none` 完全恢复旧推理行为。
完全被手写覆盖且两侧没有印刷结构证据的内容无法通过这类确定性后处理可靠重建。

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

### 导出 ONNX

导出的 ONNX 输入是已完成 RGB、ImageNet 归一化的 `float32 NCHW` 张量。联合模型包含三个输出：`logits`、未遮罩的 `candidate` 和可以直接保存为图片的 `restored`；后两个输出为 `[0,1]` RGB。

```bash
python export_onnx.py \
  --checkpoint /Users/peng/Documents/models/hw_clear/cpu_v3_joint/best.pth \
  --output /Users/peng/Documents/models/hw_clear/cpu_v3_joint/best.onnx \
  --height 512 \
  --width 512
```

默认导出动态 batch/height/width，并用 ONNX Runtime CPU 对全部三个输出进行数值校验；需要固定输入尺寸时可加 `--fixed-shape`。

## 旧模型兼容说明

新训练生成的 checkpoint 会自动记录模型配置。旧 checkpoint 没有这些元数据，推理时必须显式指定其真实配置，例如：

```bash
python predict.py --input input.jpg --output output.png \
  --checkpoint old.pth \
  --model deeplabv3plus_resnet50 --num-classes 3 --output-stride 16
```

旧仓库曾出现“文件名写 ResNet-50、训练代码实际建 ResNet-101”和 2/3 类不一致的情况；如果严格加载失败，应以 checkpoint 的权重 shape 为准核对结构，不能用 `strict=False` 掩盖类别头不匹配。
