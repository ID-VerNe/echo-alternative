# echo替代

把一本书"编译"成一套可复用的 LLM 说话工作流。

输入一个作者的一批书，离线产出风格资产包（`style_spec.json` + `style_anchors.jsonl`），在线用固定的内容/风格解耦管线生成回复，让模型在新话题上也用作者的方式说话——而不是复述书里的内容，也不是套一层形容词人设。

- 设计方案：[`docs/issues/0001-book-to-style-pipeline.md`](docs/issues/0001-book-to-style-pipeline.md)（与 Issue #1 同源）
- 实验记录：Issue [#1](https://github.com/ID-VerNe/echo-alternative/issues/1) 的评论
- 语料：`图书语料库/`（本地目录，含版权素材，已在 `.gitignore` 中排除）

## 管线

```
离线  pipeline/corpus.py   md → 清洗 → 段落 + 元数据
      pipeline/anchors.py  风格强度打分 + 表达功能打标（rant/argument/anecdote/selfmock）
      pipeline/spec.py     LLM 归纳 + 硬统计注入 → style_spec.json（含 compiled_prompt）
      pipeline/stats.py    句长/标点/词汇多样性（纯代码计算）

在线  pipeline/generate.py  意图分类 → 事实清单（不成句）→ 3 候选风格转译 → rubric rerank → 规则质检
      pipeline/metrics.py   Cosine Delta、句长 JS、标点余弦、n-gram 重叠、事实保留
      pipeline/stylevec.py  内容无关风格向量（虚词 n-gram + 句长转移 + 标点 + 句首词）
```

实验结果与结论见 [`docs/REPORT.md`](docs/REPORT.md)。

## 目录约定

```
图书语料库/
  pdf/        原始 PDF 分卷（按书目分目录）
  markdown/   转换后的 markdown（按书目分目录，与 pdf 同名）
pipeline/     管线代码
experiments/  e01…e06 每步实验脚本（可重跑）
artifacts/    产出物（jsonl 与中间结果已 gitignore，spec 入库）
outputs/      评测结果与日志（gitignore）
docs/issues/  本地留档的 issue / RFC
```

## 复现

```powershell
python experiments/e01_audit.py     # 语料体检 + 清洗落盘
python experiments/e04_anchors.py   # 锚点库
python experiments/e05_spec.py      # 生成 style_spec
python experiments/e06_eval.py      # 端到端评测 + 盲评
```

依赖：Python 3.11+、numpy；LLM 走 OpenAI 兼容接口，密钥放 `.secrets.json`（`{"api_key": "..."}`，已 gitignore）。
