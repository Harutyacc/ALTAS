# Syn4–Syn6 条件掩码诊断

完整的实验链、关键结果、综合结论和下一阶段建议见 [EXPERIMENT_SUMMARY.md](EXPERIMENT_SUMMARY.md)。本文件保留各诊断脚本的具体用法与运行记录。

这个独立实验只训练现有的 `FeatureExtractor` 和 `TaskPredictor`，不会修改 ALTAS 主训练代码、生成器或已有运行结果。三种条件使用同一份合成数据、相同训练/测试划分、相同网络初始化与 batch size：

- `oracle`：使用数据生成器提供的逐样本真实掩码。
- `union`：所有样本都使用两条分支的特征并集。
- `full`：完整输入，作为参考；在有限样本下可能对噪声特征过拟合，并非测试表现上限。

默认只在掩码输入上训练预测器，以单独检验网络能否利用正确的条件特征。掩码位置采用项目现有的批内 shuffle 替换方法。测试报告按 `feature_10 < 0` 和 `feature_10 >= 0` 分组，报告交叉熵与准确率；shuffle 替换重复 3 次取平均。测试时只使用测试集，且各条件使用相同的评估随机种子。

在项目根目录运行：

```powershell
python -m diagnostics.oracle_mask --dataset Syn4 --output diagnostics/results/syn4.json
```

默认是 20,000 个样本、batch size 256、200 epoch、隐藏维度 64、学习率 `1e-4`。快速检查可用 `--epochs 2 --num-samples 1000`。若要更接近当前主训练中预测器的完整输入辅助损失，可加 `--auxiliary-full-weight 2.0`（此实验中掩码损失的系数固定为 1，故 2.0 对应主训练的完整输入:掩码输入权重 1:0.5）。

如何判断：如果 `oracle` 接近 `union` 或 `full`，但 ALTAS 的生成器始终给出并集，问题更可能在选择器的目标/联合训练；如果 `oracle` 明显更差，则先检查预测器对条件特征的学习能力。合成标签含随机性，准确率不应期待达到 100%。`diagnostics/results/` 被本目录的 `.gitignore` 忽略。

本机一次参考运行（Syn4，20,000 样本，seed 42，200 epoch，GPU）得到：

| 输入 | 测试交叉熵 | 测试准确率 | `feature_10 < 0` | `feature_10 >= 0` |
| --- | ---: | ---: | ---: | ---: |
| 真实条件 mask | 0.5639 | 69.11% | 58.89% | 79.09% |
| 固定并集 mask | 0.5703 | 69.05% | 58.83% | 79.04% |
| 完整输入 | 0.9937 | 58.30% | 52.50% | 63.96% |

这是单个随机种子的诊断，不是显著性结论。真实条件 mask 的表现与并集相当且更稀疏，支持先研究生成器目标和联合训练；完整输入组的训练损失下降但测试损失上升，提示有限样本下的噪声特征过拟合。

## 监督式生成器容量检查

`supervised_selector.py` 只使用现有的 `FeatureMaskGenerator` 网络，直接学习合成数据提供的真实逐样本 mask；不使用 ALTAS 的分类器、Critic 或对抗目标，不会更改主训练。这个实验是排错用的“有答案”上限，**不能**直接作为真实任务的训练方案。损失将训练集中曾相关的特征与其余噪声特征分别求平均，再相加；这样既不让 93 个噪声维度淹没条件特征，也不对正例赋予过高权重而人为推向并集。

```powershell
python -m diagnostics.supervised_selector --dataset Syn4 --output diagnostics/results/syn4_supervised.json
```

默认沿用 batch size 256、隐藏维度 64，以及主训练中生成器的 Adam 学习率 `1e-5` 和动量设置。评估时将保留概率以 0.5 阈值转为确定性 mask，避免 Gumbel 随机采样影响容量判断。报告包含测试集逐样本完全匹配率、TPR/FDR，以及 `feature_10` 正负两组中每个关键特征的选择频率。如果直接监督仍无法学会切换，应先检查网络表达与优化；如果能学会，就应重点排查 ALTAS 的间接损失和联合训练。

Syn4、20,000 样本、200 epoch 的两次独立随机种子结果：

| seed | 测试集 mask 完全匹配率 | TPR | FDR | 平均选择数 |
| ---: | ---: | ---: | ---: | ---: |
| 42 | 88.10% | 92.61% | 7.33% | 4.01 |
| 43 | 88.40% | 92.95% | 6.87% | 4.01 |

两次运行在 `feature_10 < 0` 时对 `0、1` 的选择频率约 90%、对 `2–5` 约 10%；在另一组则正好反过来。可见这个生成器结构能够表达并学出条件切换。但不能把这个有真实 mask 监督的结果直接等同于 ALTAS 原目标的表现。

## 同一训练模型上的反事实 mask 对照

`counterfactual_masks.py` 重新运行原 ALTAS 的 `train_step`，不改变损失或优化步骤；在指定 epoch 暂停更新，用**同一套已训练的**提取器、预测器和 Critic 对照真实条件 mask、固定并集 mask、生成器当前的确定性 mask。三种 mask 使用同一批样本与同一次 shuffle 替换。逐项记录预测交叉熵、生成器对抗项 `-critic`、稀疏项与加权总目标；测试与训练子集都分别按切换因子分组。反事实评估暂时将模型设为 `eval`，因此关闭预测器 dropout，不等同于带 dropout 的单次训练梯度。

```powershell
python -m diagnostics.counterfactual_masks --dataset Syn4 --epochs 200 --output diagnostics/results/syn4_counterfactual.json --model-output diagnostics/results/syn4_counterfactual.pt
```

默认 20,000 样本、batch size 256，评估每个划分前 4,000 个样本，每次重复 3 组 shuffle。`oracle_minus_union.total` 小于 0 表示在**当前固定模型**下真实 mask 的目标更好；大于 0 表示当前固定模型更偏好并集。最好同时看 `train` 与 `test`，以及预测项和对抗项分别贡献了多少。它只能解释当前目标的局部偏好，不能单独证明全局最优解或训练梯度一定会收敛。

本机 Syn4、seed 42、200 epoch 的测试集结果：

| epoch | 生成器确定性 mask 平均选择数 | 真实 mask − 并集的总目标 |
| ---: | ---: | ---: |
| 0 | 50.75 | −0.00879 |
| 50 | 0.15 | −0.00965 |
| 100 | 1.98 | −0.00846 |
| 200 | 4.00 | −0.00889 |

第 200 轮同一套模型的测试总目标：真实 mask `0.57773`、并集 `0.58661`、当前生成器的确定性 mask `0.61612`。当前生成器对切换因子正负两组都固定选择 `2–5`，逐样本完全匹配率为 0；其保留概率在两组中几乎相同。它在第 50 轮先接近全关，之后恢复了一个固定分支。这个对照支持“优化路径没有找到更好的条件选择”，但评估关闭了 dropout，并用概率 0.5 阈值而非训练时的随机 Gumbel mask，仍需避免把反事实数值当作实际训练梯度。

## 成对特征预训练与稀疏退火 A/B 实验

`pair_warmup.py` 检查 Syn4 中 `feature_0` 与 `feature_1` 的交互是否因为训练早期很少共同暴露而丢失。两组使用同一份数据、batch size、网络初始化、联合训练 batch 顺序和随机种子。基线组保持原始 ALTAS 联合训练；干预组先仅训练提取器和预测器 100 epoch，用保留率 0.9 的随机 mask（两项理论共同出现率 0.81），随后联合训练 200 epoch，并在前 100 epoch 将稀疏权重从 0 线性升到原本的 0.3。真实 mask **仅用于评估**，不参与任何训练。

```powershell
python -m diagnostics.pair_warmup --device cuda --output diagnostics/results/syn4_pair_warmup_seed42.json
```

报告记录预训练时两项实际共同出现率、负分支上预测器对“保留 0、1、10”相较“仅保留 10”的交叉熵收益、生成器两项共同采样概率，以及正负分支各自的准确 mask 比率。两项共同出现率高只证明预测器**见到了**这组特征，不能证明它**学会了**交互；因此需先查看预训练后 `pair_ce_gain` 是否为正且明显，再判断联合训练阶段的选择器结果。A/B 若有效，还需单独测试预训练和稀疏退火各自的贡献，并复验多个随机种子。

两次本机测试（Syn4，20,000 样本，batch size 256，100 epoch 预训练 + 200 epoch 联合训练，测试集 4,000 样本）第 200 轮结果：

| seed | 组别 | 负分支 `0、1` 共同采样概率 | 平均保留特征数 | 测试准确率 | 真实 mask 匹配率 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | 原始联合训练 | 0.0364 | 1.00 | 52.25% | 0% |
| 42 | 随机 mask 预训练 + 稀疏退火 | 0.8543 | 20.96 | 61.13% | 0% |
| 43 | 原始联合训练 | 0.000004 | 0.00 | 50.25% | 0% |
| 43 | 随机 mask 预训练 + 稀疏退火 | 0.8053 | 14.77 | 63.18% | 0% |

预训练实际共同出现率分别为 0.8099 / 0.8094，预训练后负分支 `pair_ce_gain` 分别为 0.0626 / 0.0440。干预确实让预测器利用该交互，也防止生成器丢掉 `0、1`；但第 200 轮它在**两个分支都固定选择 `0–5、10`**，且还保留许多噪声特征。这是从“丢失特征”转成“过度保留并集”，并非学会样本级条件选择。准确率提升不能单独证明掩码正确。此结论只覆盖两个种子和 200 轮联合训练，不能证明更长训练或其他配置一定无效。

## 冻结预测器后的故障定位

`frozen_oracle_selector.py` 先依照 `oracle_mask.py` 的方法，仅用真实 mask 训练预测器／提取器 200 epoch，然后将两者冻结。在同一个冻结模型上，用相同 shuffle 比较真实 mask、固定并集及逐特征删除的交叉熵与 `CE + 0.3 × 平均保留率`。最后只训练生成器，不训练 Critic。真实 mask **不参与生成器损失**；本实验使用真实 mask 训练诊断预测器、用训练集真值构造并集，属于带特权信息的故障定位，不是可直接使用的无监督方案。

```powershell
python -m diagnostics.frozen_oracle_selector --device cuda --selector-learning-rate 0.001 --union-keep-prob 0.7 --output diagnostics/results/syn4_frozen_oracle_seed42_mlp_lr1e3.json
python -m diagnostics.frozen_oracle_selector --device cuda --selector-kind switch_vectors --selector-learning-rate 0.001 --union-keep-prob 0.7 --output diagnostics/results/syn4_frozen_switch_seed42.json
```

`switch_vectors` 是定位用上界：它**预先知道** `feature_10` 是切换因子，但两组具体该选哪些特征仍由预测损失和稀疏损失学习，不使用真实 mask 监督生成器。下表是测试集第 200 轮结果；负／正表示对应分支的完整真实 mask 匹配率：

| seed | 生成器 | 学习率 | 负分支 | 正分支 | 测试目标 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 42 | 原 MLP，并集初始化 | `1e-5` | 0% | 0% | 0.58903 |
| 42 | 原 MLP，并集初始化 | `1e-4` | 0% | 0% | 0.58998 |
| 42 | 原 MLP，并集初始化 | `1e-3` | 5.7% | 0% | 0.60552 |
| 42 | 已知切换因子的双掩码向量 | `1e-3` | 100% | 100% | 0.57590 |
| 43 | 已知切换因子的双掩码向量 | `1e-3` | 100% | 100% | 0.57169 |

冻结模型上，真实 mask 相比并集的总体目标改善为 seed 42 的 `0.01312`、seed 43 的 `0.00479`。逐维反事实显示：seed 42 负分支删 `2` 有利 `0.00548`，但正分支删 `2` 有害 `0.14830`；正分支删 `0` 有利 `0.00590`，但负分支删 `0` 有害 `0.03736`。seed 43 同样存在收益／伤害不对称。因此“所有样本统一删一维”的平均更新会强烈保留并集；只有按切换因子分流后，较小的分支特有收益才能被利用。已知切换因子的上界能成功，说明合格预测器下该损失支持正确条件解；普通 MLP 没能从监督信号中稳定建立这道分流。不能仅凭此证明 Critic 的独立作用，也不能把带特权信息的上界当作最终方法。

## Critic 梯度消融

`critic_ablation.py` 在原始 ALTAS 联合训练中，只将**生成器目标**的对抗权重从 `1.0` 改为 `0`；Critic 仍按原步骤更新，以保持两组的更新顺序和随机数消耗相同。每个 seed 的两组使用同一数据、初始权重、batch 顺序、温度调度和评估随机种子。未改动 batch size 或主训练代码。不能跨组直接比较各自加权总损失，报告另给共同的“预测交叉熵 + 稀疏项”。

```powershell
python -m diagnostics.critic_ablation --device cuda --seeds 42,43 --output diagnostics/results/syn4_critic_ablation_seed42_43.json
python -m diagnostics.critic_ablation --device cuda --seeds 44 --output diagnostics/results/syn4_critic_ablation_seed44.json
```

Syn4，20,000 样本，batch size 256，200 epoch，测试集 4,000 样本、每次 shuffle 评估重复 3 次。下表是测试集结果；“负/正匹配”分别是两个分支的完整真实 mask 匹配率：

| seed | 生成器对抗权重 | 测试准确率 | 测试交叉熵 | 平均保留数 | 负/正匹配 | 负分支 `0、1` 共同采样概率 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 1.0 | 55.38% | 0.7403 | 3.68 | 0% / 0% | 0.3220 |
| 42 | 0 | 55.99% | 0.7369 | 4.06 | 0% / 0% | 0.7683 |
| 43 | 1.0 | 49.75% | 0.7906 | 0.00 | 0% / 0% | 0.000008 |
| 43 | 0 | 57.23% | 0.7166 | 3.88 | 0% / 0% | 0.0062 |
| 44 | 1.0 | 61.66% | 0.6721 | 7.00 | 0% / 0% | 0.4825 |
| 44 | 0 | 59.47% | 0.6932 | 5.00 | 0% / 99.5% | 0.1181 |

去掉对抗梯度的配对准确率变化依次为 `+0.61`、`+7.48`、`−2.19` 个百分点，三种子平均 `+1.97` 个百分点；方向不一致，样本量不足以声称稳定提升。第 50 轮两组在 seed 42、43 都几乎全不选，说明 Critic 不是早期坍缩的必要条件。第 200 轮所有组的负分支匹配率都是 0；seed 44 无 Critic 组的高正分支匹配率来自**两个分支都固定选 `2–5、10`**，并非样本级切换。消融量化出 Critic 会显著改变部分种子的后期轨迹，但拿掉它不能解决核心故障。后续优先研究无特权信息的条件路由／分支梯度机制，同时让掩码预测器充分学习负分支交互；Critic 可暂时关闭以减少变量，待条件选择成立后再单独加回。

## 不预先指定切换位置的路由器 + 两套掩码

`learned_router.py` 延续冻结预测器诊断：先用真实逐样本 mask 训练提取器／预测器 200 epoch，之后冻结。两套掩码以训练集真实 mask 的**并集**初始化，这只给出候选特征集合、不给出每个样本走哪个分支。路由器接收完整的 100 维输入；代码没有向其指定 `feature_10`，也没有把真实 mask 或 `feature_10` 符号放入选择器损失。选择器仅优化两个专家各自的预测交叉熵、平均保留率和轻微的路由使用率平衡。`feature_10` 符号和真实 mask 只在评估时用于分组及计分。

基本训练同时让所有样本按路由概率更新两位专家。为了检验误路由样本对专家梯度的污染，`--confidence-start-epoch 100 --expert-confidence-fraction 0.5` 使第 101 轮以后：路由器仍由**全部样本**更新，专家仅由当前路由器分配给自己、且置信度最高的一半样本更新。置信筛选只看模型概率，不看真实分支标签。两组使用同一 seed，在前 100 轮具有相同训练设置。

```powershell
python -m diagnostics.learned_router --device cuda --router-kind linear --selector-epochs 400 --checkpoints 0,100,200,300,400 --seed 42 --output diagnostics/results/syn4_learned_router_linear_seed42_400epochs.json
python -m diagnostics.learned_router --device cuda --router-kind linear --selector-epochs 400 --checkpoints 0,100,200,300,400 --seed 42 --confidence-start-epoch 100 --expert-confidence-fraction 0.5 --output diagnostics/results/syn4_learned_router_linear_confident_seed42_400epochs.json
```

Syn4，20,000 样本，batch size 256，预测器 200 epoch，选择器 400 epoch。以下为测试集 4,000 样本的结果；“完整匹配”要求全部 100 维都与真实逐样本 mask 相同：

| seed | 专家更新 | 负分支完整匹配 | 正分支完整匹配 | 总体完整匹配 | 测试准确率 | 测试目标 |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 42 | 普通概率加权 | 0% | 98.86% | 50.00% | 69.03% | 0.58130 |
| 42 | 高置信样本 | 93.27% | 99.36% | 96.35% | 68.80% | 0.57833 |
| 43 | 普通概率加权 | 0% | 96.49% | 49.43% | 69.76% | 0.57461 |
| 43 | 高置信样本 | 92.88% | 99.41% | 96.23% | 68.91% | 0.57169 |

两个线性路由器在第 400 轮都把 `feature_10` 排为绝对输入权重第 1 位，尽管训练时不知道它是切换因子。普通更新已能路由，但负分支专家仍选并集；高置信更新使两位专家分别学到完全正确的 `0、1、10` 和 `2–5、10`，剩余误差来自路由器把少量样本送错专家。这支持“误路由污染专家梯度”是可修复的关键环节，而不仅仅是提高学习率。测试准确率并不比普通更新高，因为更稀疏的正确掩码与最大化预测准确率并非同一指标；须同时看掩码、交叉熵和目标值。

**限制：**这是定位实验，不是已解决的原任务。冻结预测器使用真实 mask 训练，两位专家的并集初始化也来自真值；因此尚未证明完全不使用特权信息的 ALTAS 可以自举出同样结果。此处固定两个专家，并施加接近 50/50 的使用率平衡，只适合当前 Syn4 的受控测试，不能据此宣称可泛化到未知分支数、比例或其他数据集。下一步需移除 oracle 预测器和真值并集初始化，检查高置信训练能否仍然成立，然后才可考虑主训练集成与 Critic 回归。

### 分步移除特权信息

`learned_router.py` 现在可以分别设置预测器的 mask 来源 `--predictor-mask-source oracle|random`，以及专家初始化 `--expert-init union|uniform`。`random` 在每个样本独立抽取 `0.1–0.9` 的保留率，再随机采样 mask；`uniform` 则令所有 100 维从相同的保留概率（下面为 `0.5`）开始。两者同时使用时，预测器和选择器的训练 DataLoader 都不含真实 mask，专家初始化也不读取真值并集；真实 mask 仍仅用于测试集打分和显示 oracle/union 参照。这里没有把切换位置提供给线性路由器。

```powershell
python -m diagnostics.learned_router --device cuda --seed 42 --router-kind linear --expert-init uniform --union-keep-prob 0.5 --selector-epochs 400 --checkpoints 0,100,200,300,400 --confidence-start-epoch 100 --expert-confidence-fraction 0.5 --output diagnostics/results/syn4_router_oracle_predictor_uniform_seed42.json
python -m diagnostics.learned_router --device cuda --seed 42 --predictor-mask-source random --expert-init uniform --union-keep-prob 0.5 --router-kind linear --selector-epochs 400 --checkpoints 0,100,200,300,400 --confidence-start-epoch 100 --expert-confidence-fraction 0.5 --output diagnostics/results/syn4_router_random_predictor_uniform_seed42.json
```

同一 seed 42、同一数据，预测器训练 200 轮、选择器训练 400 轮的测试结果：

| 预测器训练 mask | 专家初始化 | 冻结预测器在真实 mask 下负分支 CE | 负分支完整匹配 | 正分支完整匹配 | 总体完整匹配 |
| --- | --- | ---: | ---: | ---: | ---: |
| 真实逐样本 | 真值并集 | 0.6785 | 93.27% | 99.36% | 96.35% |
| 真实逐样本 | 100 维均匀 | 0.6785 | 93.02% | 99.46% | 96.28% |
| 随机 mask | 真值并集 | 0.6968 | 0% | 95.65% | 48.38% |
| 随机 mask | 100 维均匀 | 0.6968 | 0% | 97.53% | 49.33% |

真值并集初始化不是主要依赖：在合格预测器下，从 100 维均匀初始化仍能学出两套正确专家掩码。真正失效的是不使用真实 mask 的预测器训练。随机 mask 训练 200 轮后，固定真实 mask 测试时负分支 CE 为 seed 42 的 `0.6968`、seed 43 的 `0.7016`；seed 42 延长到 400 轮仍为 `0.6957`。它没有稳定地学到 `0、1` 交互，于是负分支专家最终在两种初始化下都不选择 `0、1`。当前随机 mask 保留率均匀分布在 `0.1–0.9`，理论上 `0、1` 共同出现率约 30.3%，说明单纯增加成对暴露仍不够；需要改善无 oracle mask 的预测器学习和泛化，而不是继续单独调路由器。两种训练均在 seed 42 下完成了 400 轮，预测器弱的问题用第二个 seed 和额外训练轮数核查；这还不是对所有数据或训练方案的否定。

### 接回原式生成掩码反馈的预测器更新

`joint_router.py` 将预测器／提取器改为与原 `ALTASTrainer._update_predictor` 相同的更新形式：每个 batch 先用当前路由器／专家**采样、detach** 的 mask 更新任务模型，损失为 `CE(full) + 0.5 × CE(masked)`；随后用更新后的任务模型训练路由器和两位专家。高置信专家更新仍从第 101 轮开始。训练阶段不使用真实 mask、真值并集或预先给定的切换位置，专家从 100 维均匀保留概率 `0.5` 初始化。为了单独测试这条预测器反馈，**暂时不加入 Critic**；这并非对完整 ALTAS 的一键替换或对隐空间匹配价值的最终判断。

```powershell
python -m diagnostics.joint_router --device cuda --seed 42 --output diagnostics/results/syn4_joint_router_seed42_400epochs.json
python -m diagnostics.joint_router --device cuda --seed 43 --output diagnostics/results/syn4_joint_router_seed43_400epochs.json
```

Syn4，20,000 样本，batch size 256，400 epoch；测试结果：

| seed | 第 50 轮平均保留数 | 第 400 轮负分支 oracle-mask CE | 负/正完整 mask 匹配 | 生成 mask 测试准确率 | 第 400 轮任务训练 full/masked CE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 0.00 | 0.8094 | 0% / 0% | 60.14% | 0.393 / 0.725 |
| 43 | 0.00 | 0.8314 | 0% / 0% | 60.01% | 0.391 / 0.726 |

两个 seed 在第 50 轮均先接近全关。后期路由器虽能把 `feature_10` 学为线性权重第 1 位，掩码专家仍没有同时学对负／正分支。冻结随机 mask 预测器的同 seed 42 生成 mask 准确率约 59.1%，联合反馈提高到约 60.1%；但真实负分支 mask 下的预测器 CE 从冻结随机方案的 `0.6968` 恶化为 `0.8094`，seed 43 联合反馈更达到 `0.8314`。因此原式反馈**确实影响了预测器**，但当前更像是与错误掩码共同适应：训练 full CE 持续下降，masked CE 和对真实条件特征的泛化仍差。不能凭当前 mask 上的准确率小幅提升宣布样本级问题已解决。接下来需要在不使用真值 mask 的前提下，先让预测器在负分支获得可靠的交互信号，且防止选择器早期全关；再测联合反馈是否能稳定放大这个信号，最后才考虑把 Critic 加回。
