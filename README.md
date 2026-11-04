# 基于人工智能大模型多智能体协同的工业设备故障诊断与运维决策（代码与数据）

本仓库为论文《基于人工智能大模型多智能体协同的工业设备故障诊断与运维决策方法》的全部实验代码、提示模板、设备知识库和大语言模型原始输出，可复现论文中的全部表格与图中数据。

## 1. 环境

- Python ≥ 3.10，依赖见 `requirements.txt`：`pip install -r requirements.txt`
- 定量实验只使用 CPU，单次完整运行约 10 min。
- 大语言模型实验需要 Anthropic API 密钥（或兼容接口），见第4节。

## 2. 数据

`data/ai4i2020.csv` 为 AI4I 2020 预测性维护数据集（UCI Machine Learning Repository，DOI: 10.24432/C5HS5C，CC BY 4.0 许可），原始出处：Matzka S. Explainable artificial intelligence for predictive maintenance applications. AI4I 2020: 69-74。

## 3. 文件说明与复现步骤

| 文件 | 内容 | 对应论文内容 |
|---|---|---|
| `agents.py` | 数据感知、机理诊断（含阈值扰动 `perturb`、裕度证据 `evidence`）、数据诊断智能体 | 第二部分 |
| `run_lib.py` | 数据诊断智能体训练与预测、协同流程 `collab` | 第二部分 |
| `knowledge_base.md` | 设备知识库中的机理规则及来源 | 表1 |
| `prompts/coordinator.txt` | 协调智能体完整提示模板 | 表2 |
| `exp_main.py` | 主实验、知识误差稳健性、样本稀缺实验 | 表3、图3 |
| `baselines.py` | RF / SVM / GBDT / MLP 与3种输入组合的对比 | 表4 |
| `exp_cost.py` | 运维决策成本与停机成本比敏感性、协同诊断案例 | 表6、图4、表7 |
| `llm_prep.py` | 生成大语言模型实验的输入（样本、请求、案例消息） | 第四部分第4节 |
| `run_llm.py` | 通过 API 重新调用大语言模型 | 第四部分第4节 |
| `llm_eval_direct.py` / `llm_eval_plan.py` / `llm_eval_report.py` | 评价 LLM 直接诊断、自主规划和报告数值忠实度 | 表5 及正文 |
| `llm_runs/` | 论文所用的全部 LLM 输入与原始输出 | — |
| `results/` | 运行结果（JSON） | — |

复现顺序：

```bash
python exp_main.py        # 表3、图3
python baselines.py       # 表4
python exp_cost.py        # 表6、图4、表7
python llm_prep.py        # 生成LLM实验输入（已附带，可跳过）
python llm_eval_direct.py; python llm_eval_plan.py; python llm_eval_report.py   # 用已保存的LLM输出直接评价
```

全部随机划分使用固定种子（`StratifiedShuffleSplit(10, test_size=0.3, random_state=2026)`；第 i 次划分的阈值扰动种子为 100+i）。

## 4. 大语言模型实验

- 论文中的大语言模型为 Anthropic Claude Sonnet 5.5（`claude-sonnet-5-5`），通过智能体接口以对话方式调用，采样参数取默认值。原始输出保存在 `llm_runs/out_*.json`，可直接用评价脚本复核论文数字。
- 如需重新调用：`export ANTHROPIC_API_KEY=...`，然后 `python run_llm.py direct|plan|report`。大语言模型输出具有随机性，重新运行的结果会有小幅差异；token 用量写入 `results/usage_*.json`。
- 若使用其他模型（如国产大模型），修改 `run_llm.py` 中的 `call` 函数即可。

## 5. 主要超参数

- 数据诊断智能体：故障判别随机森林 300 棵树，各故障类型随机森林 200 棵树，`class_weight='balanced_subsample'`，其余为 scikit-learn 默认值（Gini 准则、`max_features='sqrt'`、不限深度）；概率阈值 0.5。
- 多标签处理：采用二元相关法，每类故障单独训练一个二分类器；仅当故障判别器判定故障时输出类型，可同时输出多类。

## 6. 许可

代码采用 MIT 许可；数据集遵循 CC BY 4.0。
