# echo替代

把一本书"编译"成一套可复用的 LLM 说话工作流。

目标：输入任意一本书（或同一作者的一批书），离线产出一个风格资产包（`style_spec.json` + `style_anchors.db`），在线用固定的内容/风格解耦管线生成回复，让模型在新话题上也用作者的方式说话——而不是复述书里的内容，也不是套一层形容词人设。

- 设计方案：[`docs/issues/0001-book-to-style-pipeline.md`](docs/issues/0001-book-to-style-pipeline.md)（与 Issue #1 同源）
- 语料：`图书语料库/`（本地目录，含版权素材，已在 `.gitignore` 中排除）

## 目录约定

```
图书语料库/
  pdf/        原始 PDF 分卷（按书目分目录）
  markdown/   转换后的 markdown（按书目分目录，与 pdf 同名）
docs/issues/  本地留档的 issue / RFC
```

## 状态

- [x] 仓库初始化
- [x] 方案记录（Issue #1）
- [ ] 语料 markdown 转换与校验
- [ ] 离线提取管线 MVP
- [ ] 在线生成管线 MVP
- [ ] 评估指标栈
