# 图解大模型（Hands-On Large Language Models）学习笔记

《图解大模型》一书的配套实践仓库：各章 notebook 运行笔记、图片整理与测试题。

> 原书：Jay Alammar & Maarten Grootendorst, _Hands-On Large Language Models_（图灵出品）

## 项目结构

```
├── 第一章/ ~ 第十二章/        # 各章 notebook（已完成本地化适配）
│   ├── picture/               # 章节配图（统一命名 N-M.jpg）
│   ├── 测试题.html            # 每章 44 题（单选15/多选6/判断10/填空8/简答5）
│   ├── data/                  # 数据集/词向量本地缓存（随仓库提供，第12章除外）
│   └── *.ipynb                # 运行成功的 notebook（含输出）
├── 图解DeepSeek-R1/           # 附录：DeepSeek-R1 原理精读（配图 A-1~A-18）
├── 大模型面试题200问.ipynb     # 面试题整理（216 个单元格）
├── models/                    # 本地模型文件（不入库，见下方"模型准备"）
├── .env.example               # API Key 配置模板（复制为 .env 使用）
└── requirements.txt
```

## 各章内容与环境适配

| 章节    | 主题                | 本地适配说明                                                                                          |
| ------- | ------------------- | ----------------------------------------------------------------------------------------------------- |
| 第1-3章 | 词元、嵌入、分类    | 全部本地运行                                                                                          |
| 第4章   | 文本分类微调        | BERT 微调（GPU）；ChatGPT 分类器改走百炼兼容接口                                                      |
| 第5章   | 主题建模            | BERTopic + 本地嵌入模型；主题标签 LLM 改走百炼                                                        |
| 第6章   | 提示工程            | Phi-3 q4 量化 GGUF 本地推理                                                                           |
| 第7章   | 链式架构/记忆/ReAct | **langchain 1.4.2 适配**：LLMChain→LCEL 管道，记忆改为手动历史管理，ReAct 需 OpenAI API（已标注跳过） |
| 第8章   | 语义搜索与 RAG      | rerank 走 Cohere（Key 在 .env，经 Cloudflare Worker 代理）                                            |
| 第9章   | 生成模型推理        | 全部本地运行                                                                                          |
| 第10章  | 嵌入微调            | 句向量微调（GPU）                                                                                     |
| 第11章  | 命名实体识别        | token 分类微调（GPU）                                                                                 |
| 第12章  | 微调生成模型        | QLoRA 指令微调 + DPO 偏好调优，**bf16 混合精度**（trl 1.14 对量化模型强制 LoRA 为 bf16）              |
| 附录    | DeepSeek-R1         | 纯理论图文，无代码                                                                                    |

## 环境配置

```bash
# 1. 创建虚拟环境（Python 3.10+）
python -m venv .venv
.\.venv\Scripts\activate

# 2. 安装 PyTorch（CUDA 13.0 版本，按本机 CUDA 调整）
pip install torch --index-url https://download.pytorch.org/whl/cu130

# 3. 安装 llama-cpp-python（CUDA 支持）
$env:CMAKE_ARGS="-DGGML_CUDA=on"
pip install llama-cpp-python==0.3.35

# 4. 安装其余依赖
pip install -r requirements.txt
```

## API 配置

第4/5章的 OpenAI 调用已改为**百炼兼容接口**（OpenAI SDK + 自定义 base_url），第8章 rerank 走 Cohere：

```bash
cp .env.example .env   # 填入真实 Key
```

| 变量             | 用途                                           |
| ---------------- | ---------------------------------------------- |
| `api_key`        | 百炼 API-KEY（第4/5章 LLM 调用）               |
| `ali_model`      | 百炼模型名（如 `qwen-plus`）                   |
| `base_url`       | 百炼 OpenAI 兼容端点（代码中有默认值，可覆盖） |
| `cohere_api_key` | Cohere Rerank（第8章）                         |
| `HF_TOKEN`       | HuggingFace（可选）                            |

## 数据集准备

各章 `data/` 目录为本地数据集缓存（**已随仓库提供**，仅第12章因体积约 3GB 不入库），notebook 均通过 `load_dataset(..., cache_dir="data")` 读取，克隆后即可离线运行：

| 章节   | 数据集                                                                           |
| ------ | -------------------------------------------------------------------------------- |
| 第2章  | glove-wiki-gigaword-50 词向量（gensim，`GENSIM_DATA_DIR` 指向 data/gensim-data） |
| 第4章  | rotten_tomatoes                                                                  |
| 第5章  | maartengr/arxiv_nlp                                                              |
| 第10章 | glue-mnli / glue-stsb；nltk punkt_tab（data/nltk）                               |
| 第11章 | conll2003、rotten_tomatoes                                                       |
| 第12章 | ultrachat_200k（test_sft）、distilabel-intel-orca-dpo-pairs — **需自行下载**     |

第12章数据下载（国内建议走 hf-mirror）：

```bash
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from datasets import load_dataset; load_dataset('HuggingFaceH4/ultrachat_200k', split='test_sft', cache_dir='data'); load_dataset('argilla/distilabel-intel-orca-dpo-pairs', cache_dir='data')"
```

## 模型准备

所有模型放 `models/` 目录（已被 .gitignore 排除），可从 HuggingFace 下载（国内可用 `HF_ENDPOINT=https://hf-mirror.com`）：

- `Phi-3-mini-4k-instruct-q4.gguf` — 第6、7章本地推理
- `TinyLlama-1.1B-intermediate-step-1431k-3T` — 第12章 SFT 基座
- `TinyLlama-1.1B-Chat-v1.0` — 第12章 DPO 基座
- `bert-base-cased` / `bert-base-uncased` — 第4、11章微调
- `all-mpnet-base-v2` / `all-MiniLM-L6-v2` / `gte-small` / `bge-small-en-v1.5` — 嵌入
- `flan-t5-small`、`blip2-opt-2.7b`、`clip-ViT-B-32` 等 — 其余章节按需

> 微调产物（`*_qlora/`、`checkpoints/`、`results/` 等）均已配置在 [.gitignore](.gitignore) 中，不会入库。

## 测试题

每章一份交互式 HTML 测试题（44 题），答案默认隐藏，支持单题/全部显示、选项标记作答，浏览器直接打开即可。

## 运行提示

- notebook 使用相对路径加载模型与图片，请在各章目录内启动 Jupyter
- 数据集与模型均已本地化：数据随仓库提供（第12章除外），模型需放入 `models/`
- 第4/5章 LLM 调用依赖 `.env` 中的百炼配置（见"API 配置"）
- 显存参考：第4/11章微调约需 8GB；第12章 QLoRA+SFT 约 11GB（RTX 4070 Ti 12GB 实测通过）
- 第7章已适配 langchain 1.4.2（LCEL 写法），运行时请勿混用 langchain 0.3.x 的记忆/Agent API
