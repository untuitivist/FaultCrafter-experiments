# FaultCrafter 对比实验

独立公开实验仓库，用于运行 T1–T4 对比、源工况 CORAL、生成质量评价和结果导出。方法主仓库为 [FaultCrafter](https://github.com/untuitivist/FaultCrafter)，本文方法后续在主仓库发布。

## 环境

使用 Python 3.11 或更新版本、CUDA GPU，以及 Node.js 22 或更新版本（生成 Excel 时使用，不需要安装 npm 包）。先按服务器 CUDA 环境安装合适的 PyTorch，再安装其余依赖：

```bash
git clone https://github.com/untuitivist/FaultCrafter-experiments.git
cd FaultCrafter-experiments
python -m pip install -r requirements.txt
```

仓库公开，可直接克隆。所有命令在仓库根目录执行；无需安装方法主仓库。`faultcrafter/` 仅包含固定诊断 CNN 和必要的数据适配器。

## 下载实验数据

预处理数据存放在 [ModelScope 数据集](https://www.modelscope.cn/datasets/untuitivist/faultcrafter)，版本为 `prepared-v19-20261021`。共 21 个压缩包，约 9.64 GB，恢复后约 12.10 GB；包含八台设备、五个种子的原始信号缓存、窗口、划分与身份清单。

```bash
python -u tools/restore_dataset.py --download --snapshot ../faultcrafter-download --output ../faultcrafter-data
```

恢复工具逐包、逐文件核对 SHA256。若数据集需要认证，使用环境变量 `MODELSCOPE_API_TOKEN`，不要写进脚本或 Git。原始记录路径作为历史身份保留；运行时通过 `--data-run` 读取恢复后的缓存，不要求服务器存在原 Windows 路径。PU 已知异常记录及其掩码保持原样，详见数据集介绍。

## 查看计划与运行

先核对计划，不启动训练：

```bash
python -m research.benchmarks.comparison_train --data-run ../faultcrafter-data --output outputs/full --plan-only
```

当前精简全矩阵包含 1500 项生成器训练／适配、13295 项诊断 CNN，使用种子 42–46，包含 T4 DiffUCD 和 T1/T2 源工况 CORAL。计划从缓存附带的原始清单动态构建；不重写缓存清单。本文未定模型及其专属消融不计入这些数量。

Linux 服务器完整运行并保存日志：

```bash
set -o pipefail
mkdir -p outputs/full
python -u -m research.benchmarks.comparison_train --data-run ../faultcrafter-data --output outputs/full 2>&1 | tee -a outputs/full/console.log
```

只测量代表任务的完整训练耗时，使用独立目录并增加 `--representative`，默认种子 42；该子集不代表全矩阵完成。各方法的正式训练预算、辅助阶段和停止条件见 `research/benchmarks/recipes.py`，来源说明见 `research/protocols/baseline_training_recipes_20261020_zh.md`。

停止时在结果目录创建 `STOP_AFTER_CURRENT`，程序在当前任务结束后退出。恢复前重命名该文件，再执行相同命令；已有结果与检查点仅在代码、数据、配置身份匹配时复用。不要把不同预算的短测试结果混入正式结果目录。

## 评价和导出

```bash
python -m research.benchmarks.quality --data-run ../faultcrafter-data --output outputs/full --device cpu
python -m research.benchmarks.workbook --data-run ../faultcrafter-data --run outputs/full
python -m research.benchmarks.export --data-run ../faultcrafter-data --run outputs/full --output ../FaultCrafter_complete.zip
```

训练记录保存检查点、实际更新／轮数、源验证选模、预测与结果；Excel 保留每次运行的种子。完整交付导出要求所有登记任务完成并通过核验，包含结果、推理资产、绘图数据、代码与许可。五种子比较报告均值和样本标准差；已有条件扫描等单次分析仍采用种子 42，不改变原实验规则。

## 对比方法与源码边界

T2 另有[原设备生成效用评价入口](research/benchmarks/ADAPTATION_RETENTION.md)，在适配完成后检验原设备的 R+G 诊断表现；T1 为适配前参照，不另加适配前实验。该入口读取已完成的检查点，不重新训练生成器；本文 LoRA 模型尚未定型，其具体加载接口待模型确定后接入。

| 方法 | 当前实现来源 |
| --- | --- |
| DiffUCD | 按期刊论文重实现，非作者官方代码 |
| CcGAN、iCCDM | 作者源码及一维信号适配 |
| Time Weaver | ConTSG-Bench 第三方实现及连续条件适配 |
| 条件 FlowTS | 作者源码及连续条件适配 |
| CORAL | 仅对齐源工况，未见目标工况不参与训练 |

固定上游版本和适配说明保存在 `research/benchmarks/method_sources.json`。`third_party/` 只保留运行和来源校验所需的文件及原许可证；这些适配不代表原论文已经验证了本项目所有任务。

仓库不包含数据、权重、实验结果、论文、个人笔记、访问令牌或研究工作区 Git 历史。`source_manifest.json` 记录此次代码快照的逐文件哈希，Git 禁用自动换行转换以保持训练身份校验一致。更新实验代码需单独审阅、提交和推送，不随公开主仓库发布。
