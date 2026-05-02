# rag_HYMY

本项目把多份 JSON 语录资料处理成一个本地可检索知识库，并生成可直接复制到 Claude/GPT/Gemini 网页版的提问包。

默认流程不依赖外部服务：先清洗原始 JSON，再导出 AI 打标批次，导入打标结果，构建本地检索索引，整理思维 DNA 蒸馏语料，最后按问题生成 `clipboard.md`。

## 快速开始

```bash
python3 -m hymy_rag init
python3 -m hymy_rag ingest data/raw/example.json
python3 -m hymy_rag make-tag-batches
python3 -m hymy_rag import-tagged data/tagged/example_tagged.json
python3 -m hymy_rag build-index
python3 -m hymy_rag prepare-persona
python3 -m hymy_rag ask "我该怎么选择工作？"
```

最终输出会写入 `clipboard.md`，直接复制到网页大模型即可。

## 检索后端

当前支持两种检索 backend：

- `sparse`：旧版本地 JSON 稀疏检索，零额外模型依赖，适合快速对照和兜底。
- `vector`：新版 `BGE-M3 dense+sparse + Qdrant + reranker-v2-m3` 混合检索，默认 backend，也是推荐方案。

首次安装向量检索依赖：

```bash
pip install -e ".[rag]"
python3 -m hymy_rag build-index --backend vector
```

如果要强制使用旧版稀疏检索做对比：

```bash
python3 -m hymy_rag build-index --backend sparse
python3 -m hymy_rag ask "我该怎么选择工作？" --backend sparse
```

如果要使用新版向量检索：

```bash
python3 -m hymy_rag build-index --backend vector
python3 -m hymy_rag ask "我该怎么选择工作？" --backend vector
```

Mac 上当前也可以直接运行 `vector` backend。程序会自动按 `cuda > mps > cpu` 选择设备；如果是 Apple Silicon，默认会优先使用 `mps`，并把默认编码 batch 调低到更稳的 `8`，避免一上来就把显存打满。需要更激进或更保守时，可以手动设置 `HYMY_ENCODE_BATCH_SIZE`。

两者差异可以粗略理解成：

- `sparse` 更依赖关键词命中，适合问题和回答措辞接近的语料。
- `vector` 能处理“原回答没有复述问题关键词”的情况，尤其适合 HYMY 这种答法跳跃、靠 trigger 才能理解上下文的语料。

## 目录

```text
data/raw/              放你的原始 JSON 文件
data/processed/        清洗后的中间文件
data/tagging_batches/  给 AI 批量打标的输入和提示词
data/tagged/           放 AI 返回的打标 JSON
data/index/            本地检索索引
data/distill/          自动整理好的领域蒸馏语料
prompts/               打标和复制粘贴包模板
persona/               最终的人设 Markdown，网页提问时自动拼进去
clipboard.md           每次提问生成的复制粘贴包
```

## 输入格式

支持你给的这种列表 JSON：

```json
[
  {
    "id": 1,
    "time": "2025-11-30 16:49",
    "content": "用户问题",
    "answer": "回答内容"
  }
]
```

`answer` 为空的记录会默认跳过，因为它还不是可沉淀的知识；需要保留时可用 `--keep-empty-answer`。

`ingest` 和 `import-tagged` 都支持直接传目录，会自动递归读取里面的全部 `.json` 文件。

## 推荐工作流

先把所有原始资料扔进 `data/raw/`，然后跑：

```bash
python3 -m hymy_rag status
python3 -m hymy_rag ingest data/raw
python3 -m hymy_rag make-tag-batches
```

把 `data/tagging_batches/` 里的批次发给模型打标，返回结果放进 `data/tagged/`，再跑：

```bash
python3 -m hymy_rag import-tagged data/tagged
python3 -m hymy_rag build-index
python3 -m hymy_rag prepare-persona
python3 -m hymy_rag status
```

这时你会得到两类关键产物。

`data/distill/*.md` 是按领域整理好的原话档案，既可以按领域回看原话，也可以拿去做人工蒸馏。

`data/distill/_master.md` 是跨领域总语料，用来手动蒸馏唯一一份元思维 system prompt。

`prompts/distill_domain_prompt.md` 和 `prompts/distill_master_prompt.md` 是你手动蒸馏时参考的提示词，不会自动替你生成 persona。

## 两条路径

路径 A（RAG）：`data/distill/*.md` 是按领域整理的原话档案，提问时可以按领域过滤检索相关原话。

路径 B（Persona）：`persona/meta_thinking.md` 是唯一 system prompt，必须是你手动蒸馏出的跨领域元思维，不要拆成“投资_dna / 决策_dna”这种分领域 persona。

## 如何手动蒸馏 meta_thinking.md

先看 `data/distill/_master.md`，它是所有“思维方式 / 方法论 / 价值观”语录的总档案。

再把 `data/distill/_master.md` 和 `prompts/distill_master_prompt.md` 一起丢给你要用的大模型，手动产出一份跨领域元思维 Markdown。

最终把这份文件保存到 `persona/meta_thinking.md`。`ask` 默认只加载这一份；如果文件不存在，命令会直接提示你先手动蒸馏。

## ask 的增强用法

```bash
python3 -m hymy_rag ask "我该怎么选择工作？" --domains 职业,教育,决策 --context "当前时间：2026年4月；我的现实约束：服务期6年，家庭经济紧张。"
```

这样生成的 `clipboard.md` 会自动把你的当下处境拼进去，符合文档里“你负责时效，大模型负责推理”的工作方式。

如果你用了自定义 persona 文件，也可以显式指定：

```bash
python3 -m hymy_rag ask "我现在该不该继续加仓？" --domains 投资,搞钱,决策 --persona meta_thinking.md --context "当前时间：2026年4月；我已经满仓，回撤12%。"
```

## 验证

```bash
python3 -m unittest discover -s tests
```

## 如何启动 Web UI

先确保已经至少跑过 `build-index`，否则页面虽然能打开，但无法真正生成提问包。

启动命令：

```bash
python3 -m hymy_rag serve
```

默认地址是 `http://127.0.0.1:8765`。

页面里可以直接填写问题、上下文、persona 文件名和多选过滤条件，然后一键生成。生成结果会同时显示在页面上，并继续写入根目录下的 `clipboard.md`。
