# 对比生成器训练设置核查（2026-10-20）

本记录区分原论文、作者实现和本任务迁移参数。执行配置在`research/benchmarks/recipes.py`，不再通过采样点“单位”换算训练次数。一次epoch指遍历一次该阶段的数据；一次update指一次优化器更新。CcGAN的一次主更新包含一次D更新和一次G更新，两者分别计入成本。

## 核查与采用设置

| 方法 | 核实来源 | 本次执行设置 | 停止与选模 |
| --- | --- | --- | --- |
| DiffUCD期刊版 | Measurement 242 (2025) 115951，第4节；本地Zotero WZKZMQKR全文 | Adam；1000轮；每20轮学习率乘0.1。论文未给出初始lr和batch，一维实现登记为1e-4、64 | 固定轮数，取结束时权重 |
| CcGAN | 作者固定revision的RC-49 `scripts/run_train.sh`、`main.py`、`Train_net_for_label_embed.py` | G/D均Adam 1e-4，betas=(0.5,0.999)，各batch256，30000次G与30000次D更新 | 固定更新次数；不额外加本文生成器的早停 |
| iCCDM | 原文附录F表S.1；作者RC64脚本、`label_embedding.py`及`trainer.py` | Adam 1e-4，betas=(0.9,0.999)，有效batch128，100000次更新，梯度范数上限1；辅助阶段单独训练 | 固定更新次数；作者默认未启用EMA |
| Time Weaver | 原文附录A.4未列完整训练时长／batch／早停；当前代码来自ConTSG-Bench | 使用该第三方ETTm1配置：AdamW 1e-3，wd=1e-4，batch256，最多700轮，余弦调度，clip=1 | 完整源验证损失选模，连续50轮不降早停 |
| 条件FlowTS | 原文4.1—4.2；作者`FMTS/run.sh`及模型训练时间采样代码 | 选原文效率实验2500—25000档中的25000次；作者脚本lr=3e-4、logit-normal时间采样；AdamW、batch64、wd=.01登记为一维迁移设置 | 固定更新次数；不是通用官方默认预算 |

CcGAN原文不同数据集使用5000、20000、30000或40000次更新，不能只抽取其中一个数值当作全部任务的官方上限。iCCDM也有50000、100000及更高设置；当前选定RC-49配置，未统一截成20000。上述图像方法的RC-49配置用于固定优化策略，一维网络与多物理参数接口仍是本任务迁移，不能称作原任务逐参数复现。

FlowTS公开solar YAML写`max_epochs=1000`、batch18、梯度累积2，但其导入的`engine/solver.py`未在已核实的发布树中提供；不能仅凭字段名把1000解释成训练轮数或更新数。`run.sh`中的800是推理步数。Time Weaver也没有把第三方700轮冒称为原作者训练设置。

## 必要辅助阶段

| 方法／阶段 | 数据 | 轮数 | batch | 优化器与初始lr | 调度 |
| --- | --- | --- | --- | --- | --- |
| CcGAN信号→条件回归 | 源训练窗口 | 200 | 256 | SGD .01，momentum .9，wd 1e-4 | 60/120/160轮后各除5 |
| CcGAN条件→嵌入 | 不重复的源条件向量 | 500 | 128 | SGD .01，momentum .9，wd 1e-4 | 100/200/400轮后各除5 |
| iCCDM条件回归 | 源训练窗口 | 10 | 256 | SGD 1e-4，momentum .9，wd 1e-4 | 80/140轮里程碑，本轮数范围内不触发 |
| iCCDM条件嵌入 | 不重复的源条件向量 | 500 | 128 | SGD .01，momentum .9，wd 1e-4 | 150/250/350轮后各乘.1 |
| iCCDM协方差回归 | 源训练窗口 | 10 | 256 | SGD 1e-4，momentum .9，wd 1e-4 | 同条件回归 |
| iCCDM协方差映射 | 不重复的源条件向量 | 500 | 128 | SGD .001，momentum .9，wd 1e-4 | 同条件嵌入 |

一维模型使用既有条件回归器和嵌入接口。按阶段独立创建优化器、保存权重和随机状态；T2从指定父模型继续适配。非GAN使用微批梯度累积保留表中的有效批量；CcGAN的BatchNorm依赖批量，保留完整256批次。当前一维实现保留FP32累积、TF32，iCCDM未照搬原图像脚本的fp16 Accelerator执行。

## 诊断CNN

参考`Z:/THU/增量生成/test_wlcnn.py`及`experiments/hit_sqm_ours_rep02_20260407_164048/effective_config.yaml`：Adam lr=.001、wd=.01、batch64，最多1000轮，每轮遍历训练池并验证，val_acc绝对改善门槛1e-4、patience=10，从第一轮选最佳。旧风格迁移代码的分类器外循环也遍历loader，不把它的epoch误计为单batch更新。

本项目保留六块CNN、多标签BCE、Sigmoid阈值.5、源验证和既定数据权限，只迁入上述训练策略。取消固定300轮、50轮以后才选模及有放回的五五抽样。不同训练池按各自大小遍历，共用最大轮数和早停规则，记录实际轮数与更新次数；验证改善门槛不限制保存更优的最佳模型。

## 工期重新判断

旧估计同时混用了采样点单位、batch8短流程和固定300轮CNN，不能沿用。先按每种方法的实际批量分别测主训练、辅助训练和源验证耗时，再按完整任务矩阵相加。1000轮不能直接换成1000次更新；正式CNN何时早停须观察源验证曲线。`research.benchmarks.estimate`只接受匹配新配置的计时，缺测时输出缺测项目，不给虚假的天数。

## 原始来源

- DiffUCD：[期刊论文](https://doi.org/10.1016/j.measurement.2024.115951)，全文来自本地Zotero WZKZMQKR。
- CcGAN：[原文](https://arxiv.org/html/2011.07466v6)、[固定作者代码](https://github.com/UBCDingXin/improved_CcGAN/tree/106ec275901de001b44beb238992debd89e74002/RC-49/RC-49_64x64/CcGAN-improved)。
- iCCDM：[附录F表S.1](https://arxiv.org/html/2602.02114v1)、[固定作者训练脚本](https://github.com/UBCDingXin/CCDM/blob/a63aa63010255b0b3107411c2d53e7869b8af9c3/iCCDM/config/RC64/run_train.sh)。
- Time Weaver：[原文](https://arxiv.org/html/2403.02682v2)、[第三方配置](https://github.com/seqml/ConTSG-Bench/blob/6c74409a6bc24825d8f08d878229485c68d41d67/configs/generators/timeweaver_ettm1.yaml)。
- FlowTS：[原文](https://arxiv.org/html/2411.07506v2)、[作者脚本](https://github.com/UNITES-Lab/FlowTS/blob/1ec35fb1d3d89d91a1607a9f949a515347d54c8c/FMTS/run.sh)。
