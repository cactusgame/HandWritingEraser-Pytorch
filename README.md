# HandWriting Eraser 5.0

5.0 默认使用 8 GB 显存可训练的单个 `layered_eraser` 模型，同时完成
`背景/手写/可见印刷` 分割、遮挡下印刷结构预测和干净 RGB 修复。部署不再强制
CPU；CUDA、普通 PyTorch、TorchScript 和 ONNX 均可使用。旧的 CPU 分割与
`joint_eraser` checkpoint 仍然兼容。

## 5.0：分层恢复，而不是把手写区域统一填白

`layered_eraser` 使用 ImageNet 预训练 ResNet-34、GroupNorm 多尺度解码器和
一个窄的全分辨率细笔画修复头。输入已经在上游完成高分辨率修复，因此模型不再
重复做超分辨率；这个小头只补偿 stride-2 解码后容易丢失的 1–3 像素印刷笔画。
四个输出属于同一个 checkpoint 和一次前向：

```text
RGB -> shared encoder/decoder -> segmentation logits
                         ├-----> amodal print logits
                         └-----> clean RGB candidate
input × (1 - soft hand mask) + candidate × soft hand mask -> restored RGB
```

`amodal print` 表示印刷内容即使被手写覆盖也要预测。训练把手写区域进一步拆成
“下面只有纸张”和“下面存在印刷”，分别计算颜色与笔画损失，避免大量白背景把
少量交叠像素淹没。彩色面板和彩色印刷会作为非手写负样本增强。

## 为什么升级

### 1. 原模型结构的问题

- 训练参数虽然提供 `--model`，代码却始终实例化 `deeplabv3plus_resnet101`，实验配置与实际模型不一致。
- ResNet-101 + 五分支 ASPP 对这个三类文档任务偏重。高分辨率试卷必须切片，CPU 上的延迟和内存占用都较高。
- 输出步长较大，而手写笔画通常只有几个像素；仅依赖深层特征容易漏掉细笔画或模糊手写/印刷边界。
- 语义分割只能判断“当前像素是什么”，无法恢复被手写墨迹完全遮住的印刷内容；旧后处理也只是涂白，并不是真正的内容重建。
- 训练固定把页面拉伸到 `1024×1024`，破坏试卷和字形比例；训练与推理归一化还分别使用了 `[0.5]` 和 ImageNet 参数。
- 训练使用 3 类，两个旧推理脚本却构建 2 类模型；checkpoint 名称、实际 backbone 和类别数可能互相矛盾。

当前提供三档 CPU 模型：

- `server_eraser`：ImageNet 预训练 ResNet-18 编码器，使用 stride 32 上下文和 stride 16/8/4/2 逐级解码。它用更多语义容量区分形态相近的手写、印刷批注和表格线，适合 4–8 核服务端。
- `quality_eraser`：MobileNetV2 编码器，同样具有多尺度上下文和 stride 8/4/2 解码，速度和效果居中。
- `lite_eraser`：原 2.0 轻量模型，只有 stride-4 跳连，速度最快，适合资源很紧的场景。

三档旧模型继续面向 CPU；`layered_eraser` 质量优先但仍只使用标准可导出算子。
训练、验证、推理和导出共用 checkpoint 中的模型配置。

| 模型 | 参数量 | FP32 权重 | 本机 CPU 256×256 延迟（4 线程） |
| --- | ---: | ---: | ---: |
| 旧 DeepLabV3+ ResNet-101 | 58.75 M | 224.1 MiB | 约 498 ms |
| 新 `lite_eraser` | 1.92 M | 7.3 MiB | 约 98 ms |
| 新 `quality_eraser` | 1.96 M | 7.5 MiB | 约 206 ms |
| 新 `server_eraser` | 11.52 M | 44.0 MiB | 约 331 ms |
| 新 `joint_eraser` | 11.53 M | 44.0 MiB | 约 419 ms |
| GPU `layered_eraser` | 26.76 M | 约 102 MiB | 质量优先，建议 CUDA |

延迟是在当前 x86_64 机器、PyTorch 2.2.2 上预热后测得，只用于量级对比。模型精度仍需在完整数据集上重新训练并以独立验证集评估；仓库内没有附带原始训练图片和标签，因此本次升级不虚构精度结果。

旧三档模型仍然只能擦除可见手写。5.0 通过真实 SCUT 配对和强制交叠合成数据训练
分层复原分支；完全不可见的内容仍属于上下文推断，若能取得同版空白试卷，模板
配准依然比任何生成模型更可靠。

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

5.0 的分层损失分别包含：三分类结构损失、遮挡下印刷 BCE+Dice、手写下背景
RGB/低频颜色场、交叠印刷 Charbonnier+SSIM+Sobel、最终合成损失和 Mask 外
identity。交叠区域单独归一化并默认使用 8 倍重建权重，不再与白纸像素一起平均。

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

分层训练数据目录结构：

```text
datasets/data/
├── Images/
├── Labels/
├── CleanTargets/       # 可选；去手写后的真实 RGB
├── CleanPrintMasks/    # 可选；含遮挡区域的完整印刷 mask
└── splits/
```

图像与标签扩展名可以不同，但文件 stem 必须一一对应。标签必须是单通道类别 ID：背景 0、手写 1、印刷 2。

### 转换新增数据集

两个转换器都生成标准结构。SCUT 还会生成 `CleanTargets/` 和
`CleanPrintMasks/`；后者保留被手写覆盖位置的印刷结构。默认输出不会修改原始
数据，也不会静默覆盖已有结果。

```bash
python tools/convert_scut_ensexam.py --resume
python tools/convert_signatr6k.py --resume
```

转换被中断时可追加 `--resume`，已经完成且标签合法的样本会直接复用；需要从头重建时使用 `--overwrite`，两者不能同时使用。

默认输出目录：

- `/Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format`
- `/Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format`

SCUT-EnsExam 的原图和擦除图是对齐图像，转换器会把擦除图复制到
`CleanTargets/`，并从干净图生成完整印刷 mask。已有转换目录执行一次
`--resume` 即可补齐新增 mask。

SignaTR6K 的颜色映射为：蓝→背景，绿→手写，红→印刷。黄色表示手写与印刷重叠；为了与 Baidu 的互斥三类标签兼容并保证擦除召回率，默认映射为手写。官方 train/validation/test split 原样保留。

Baidu 和 SignaTR6K 没有严格配对干净图，不能把涂白结果当作真值。训练会从
真实无手写印刷区域取底图，覆盖真实笔迹，并反复选择位置直到笔迹与印刷结构达到
最低交叠量。默认在线合成概率为 `0.85`。训练增强还会生成绿色/黄色/蓝色面板和
彩色印刷，它们的标签保持为背景或印刷，专门抑制颜色误检。

### 用干净文档生成强监督数据（推荐）

如果干净文档仍保存在 PDF 中，先批量导出为高清图片。默认递归查找目录中的所有
PDF，以 300 DPI 导出无损彩色 PNG，并保留目录结构：

```bash
python tools/export_pdfs_to_images.py \
  --input-dir /path/to/clean-pdfs \
  --output-dir /Users/peng/Documents/data/HandWritingData/CleanDocuments \
  --dpi 300 \
  --jobs 4
```

例如 `clean-pdfs/合同/A.pdf` 会生成
`CleanDocuments/合同/A/page_0001.png`。脚本会校验 PDF 页数、每张图片是否完整，
并在输出根目录生成 `export_manifest.json`。再次运行会跳过已经完整且不比源 PDF
旧的文件；需要强制重新导出时增加 `--overwrite`。

默认 PNG 最适合训练且不会引入 JPEG 压缩伪影。如果磁盘空间有限，可以使用
`--format jpeg --jpeg-quality 95`。彩色文档不要使用 `--grayscale`。脚本依赖
Poppler；macOS 可运行 `brew install poppler`，Ubuntu/Debian 可运行
`sudo apt-get install poppler-utils`。

准备没有手写的试卷、合同或其他文档图片后，使用下面的脚本。它会从原三份数据的
类别 1 区域提取真实笔迹形状和颜色，也会生成随机涂写、下划线和批注。默认样本
组成是：

- 25% 不添加手写的干净负样本，用于抑制彩色背景、彩色文字和图章误检；
- 45% 强制让手写覆盖印刷内容，用于学习补齐被遮挡的文字和线条；
- 30% 主要落在背景上，用于学习纸张、彩色面板和底纹的真实颜色。

```bash
python tools/synthesize_clean_documents.py \
  --clean-root /path/to/clean-exam-images \
  --clean-root /path/to/clean-contract-images \
  --output /Users/peng/Documents/data/HandWritingData/CleanDocumentSynthetic \
  --variants-per-document 5 \
  --clean-negative-ratio 0.25 \
  --overlap-ratio 0.45 \
  --random-scribble-probability 0.25 \
  --minimum-print-overlap 0.12 \
  --workers 4 \
  --seed 17
```

`--handwriting-root` 可以指向一个直接包含 `Images/`、`Labels/` 的转换后数据集，
也可以指向这些数据集的共同父目录。例如当前机器可使用总目录：

```bash
--handwriting-root /Users/peng/Documents/data/HandWritingData
```

脚本会自动发现它下一层的 `baidu`、`SCUT-EnsExam-baidu-format` 和
`SignaTR6K-baidu-format`。也可以重复传入三个具体子目录。不要传未经转换的
`SCUT-EnsExam` 或 `SignaTR6K` 原始目录，因为它们的 mask 还不是统一的
背景 0、手写 1、印刷 2 标签。

并行以“一个干净文档”为任务单位，同一页的印刷 mask 只计算一次，再连续生成该页
的多个变体。随机种子按 split 和文档编号独立派生，因此改变 worker 数不会改变
样本内容。每个 worker 会同时保存一张完整页面及其多个 mask；普通 300 DPI A4
建议 `--workers 4`，内存充足时可增加到 `8`。

如果只需要随机乱写乱画，可传
`--random-scribble-probability 1`，此时不需要旧手写数据。脚本先按原始干净文档
切分 train/validation/test，再为每页生成多个变体，所以同一页不会跨 split 泄漏。
输出与训练代码直接兼容：

```text
CleanDocumentSynthetic/
├── Images/              # 干净文档 + 合成手写
├── Labels/              # 0 背景、1 手写、2 可见印刷
├── CleanTargets/        # 原始干净 RGB，保留彩色背景和文字颜色
├── CleanPrintMasks/     # 包括手写遮挡位置在内的完整印刷结构
├── splits/
├── provenance.jsonl     # 每个样本的底图、笔迹来源和实际重叠率
├── dataset.json         # 数量、像素和重叠统计
└── preview.jpg          # 输入 / 干净目标 / 标签三列抽检图
```

若干净图片还是高分辨率修复之前的原图，可以通过
`--high-resolution-command` 接入与线上完全相同的本地处理程序。命令必须包含
`{input}` 和 `{output}`，脚本会分别处理“加手写后的输入”和干净目标：

```bash
--high-resolution-command \
  'python /path/to/enhance.py --input {input} --output {output}'
```

若提供的干净文档本身已经是该处理程序的输出，则不要再传这个参数，以免重复增强。
正式生成前建议先用 `--limit 20 --variants-per-document 2` 小批量运行并查看
`preview.jpg`；自动印刷 mask 只是训练监督，遇到特殊印章、复杂底纹时仍应抽检。

没有额外干净文档时，仍可从旧数据的无手写区域生成固定合成集：

```bash
python tools/build_layered_synthetic.py \
  --source-root /Users/peng/Documents/data/HandWritingData/baidu \
  --source-root /Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format \
  --output /Users/peng/Documents/data/HandWritingData/LayeredSynthetic \
  --train-count 20000 \
  --validation-count 1000 \
  --seed 13
```

训练和验证分别只从各自 source split 取底图与笔迹，防止验证泄漏；输出同时包含
输入、三分类标签、干净 RGB 和完整印刷 mask。

当前这台开发机上的三份数据是小型样例副本：Baidu 5 张、SCUT train/test 各 5 张、SignaTR6K 每个 split 5 张。它们只适合验证代码；正式训练必须使用服务器上的完整数据。

## 训练

8 GB CUDA 训练推荐配置：

```bash
python main.py \
  --data-root /Users/peng/Documents/data/HandWritingData/baidu \
  --data-root /Users/peng/Documents/data/HandWritingData/SCUT-EnsExam-baidu-format \
  --data-root /Users/peng/Documents/data/HandWritingData/SignaTR6K-baidu-format \
  --data-root /Users/peng/Documents/data/HandWritingData/CleanDocumentSynthetic \
  --model layered_eraser \
  --loss layered \
  --output-stride 32 \
  --device cuda \
  --dataset-sampling balanced \
  --dataset-weights 1,2,1,2 \
  --batch-size 2 \
  --crop-size 512 \
  --total-itrs 80000 \
  --checkpoint-dir /Users/peng/Documents/models/hw_clear/gpu_v1_layered
```

默认开启 CUDA AMP，ResNet-34 backbone 使用较小学习率且固定其 BatchNorm
统计，解码器使用 GroupNorm，适合 8 GB 下 batch 2。显存不足先改
`--batch-size 1`，再改 `--crop-size 448`；不要降低交叠损失来换显存。

验证除分割指标外，会分别记录 `Background MAE/Fidelity`、
`Overlap MAE/PSNR/Fidelity` 和 `Occluded Print F1/Recall`。最佳 checkpoint
使用四者的几何平均，任何一个维度退化都不能靠白背景平均分掩盖。

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

## 推理

推理程序会同时融合分割、遮挡下印刷 logits 和最终 RGB，避免 tile 接缝：

```bash
python predict.py \
  --input samples/input.jpg \
  --output results/output.png \
  --checkpoint checkpoints/best.pth \
  --device cuda \
  --tile-size 768 \
  --overlap 128 \
  --save-mask \
  --save-print-mask
```

目录可作为输入和输出，目录层级会保留。没有 CUDA 时仍可使用
`--device cpu --threads 4`，只是新模型延迟更高。

5.0 默认关闭旧的确定性后处理，因为分层模型应直接输出最终 RGB。需要与旧模型对比
时仍可显式使用 `--postprocess background` 或 `balanced`：

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

### 导出 TorchScript

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

导出的 ONNX 输入是归一化后的 `float32 NCHW` RGB。分层模型包含四个输出：
`logits`、`print_logits`、`candidate` 和 `restored`。

```bash
python export_onnx.py \
  --checkpoint /Users/peng/Documents/models/hw_clear/gpu_v1_layered/best.pth \
  --output /Users/peng/Documents/models/hw_clear/gpu_v1_layered/best.onnx \
  --height 512 \
  --width 512
```

默认导出动态 batch/height/width，并用 ONNX Runtime 校验全部四个输出；需要固定
输入尺寸时可加 `--fixed-shape`。

## 旧模型兼容说明

新训练生成的 checkpoint 会自动记录模型配置。旧 checkpoint 没有这些元数据，推理时必须显式指定其真实配置，例如：

```bash
python predict.py --input input.jpg --output output.png \
  --checkpoint old.pth \
  --model deeplabv3plus_resnet50 --num-classes 3 --output-stride 16
```

旧仓库曾出现“文件名写 ResNet-50、训练代码实际建 ResNet-101”和 2/3 类不一致的情况；如果严格加载失败，应以 checkpoint 的权重 shape 为准核对结构，不能用 `strict=False` 掩盖类别头不匹配。
