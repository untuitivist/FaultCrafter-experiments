# T2 适配后的原设备诊断

在 T2 新设备适配完成后，使用原设备的实际模型调用路径生成故障样本，验证其 R+G 诊断表现。论文适配表新增 LoRA、全参数微调的原设备结果两行，并增加 HDU-BFDS、DIRG、HUSTbearing、CWRU 四个原数据集的 ACC 列，与四个新设备列并列；不只报告原设备总体均值。T1 作为适配前参照，不新增适配前行或重复的基座训练。从零训练没有继承原设备能力，仍只报告其新设备结果。

## 数据与评价

- 原设备为 HDU-BFDS、DIRG、HUSTbearing、CWRU，共 15 条通道。源数据、故障状态及 T1 基座留出条件沿用各检查点的划分；排除 T4 留出组合。
- 原设备各故障状态生成 63 窗，按源工况轮换取条件，生成的初始随机种子和批量顺序在适配策略间固定。与相同真实训练数据组合，从零训练统一 CNN；沿用源验证选模、最大 1000 轮及早停。
- 冻结 CNN 后，仅在原设备源工况的独立 `seen` 测试分区评价 ACC、Macro-F1；不把用于生成器训练的原始记录整段作为测试集。保存逐样本预测、测试清单、模型与数据身份。
- 五个种子为 42–46。每次先按原设备通道、原设备、新设备适配任务等权汇总，再计算五次均值与样本标准差；五种子必须覆盖相同适配任务。另按每个原数据集分别汇总，保存在 `by_device`，供四个原数据集列读取。适配前参照使用 T1 的对应原设备 `seen_conditions` 结果，不能与其未见工况指标混比。
- LoRA 可以使用保留的原设备路径；全参数微调必须使用更新后的模型，不能换回旧检查点。相同检查点、条件与种子的原设备结果只计算一次，再在各适配任务中引用。

## 完成适配后登记检查点

`cases.json` 为列表。以下是一种子示意；完整汇总需登记 42–46 的相同任务：

```json
[
  {
    "strategy": "lora",
    "split": 42,
    "base_generator": "results/generators/T1_BASE_ID",
    "adapted_generator": "results/generators/T2_LORA_ID",
    "original_generator": "results/generators/T1_BASE_ID"
  },
  {
    "strategy": "full_finetune",
    "split": 42,
    "base_generator": "results/generators/T1_BASE_ID",
    "adapted_generator": "results/generators/T2_FULL_ID",
    "original_generator": "results/generators/T2_FULL_ID"
  }
]
```

目录相对 `cases.json` 解析，需含 `best.pt`、`result.json`。`result.json` 沿用现有生成器格式，包括 `job.track/split/parent/method`、`best_checkpoint_sha256`、`parent_sha256`；检查点含 `model` 和 `schema`。读取时校验 T2 继承的确为所声明 T1 基座，所有策略在同种子下使用同一基座。

```bash
python -m research.benchmarks.adaptation_retention --data-run ../faultcrafter-data --cases cases.json --output outputs/retention --plan-only
set -o pipefail
python -u -m research.benchmarks.adaptation_retention --data-run ../faultcrafter-data --cases cases.json --output outputs/retention 2>&1 | tee -a outputs/retention/console.log
```

默认支持已经接入的外部生成器检查点。本文模型确定后，增加 `--model-factory package.module:load_generator`；函数接收 `(checkpoint, result, condition_dim)`，返回已加载权重的模型，并提供 `sample(conditions, steps, length)`。不能用外部模型冒充本文 LoRA 实验；当前尚未运行或填写本文模型的这些结果。

输出 `retention_summary.json`，缺少任何种子或原设备通道时，对应统计值留空。现有论文表导出命令增加 `--retention-run outputs/retention` 后读取本文模型的完整记录；数据和检查点不上传 GitHub。
