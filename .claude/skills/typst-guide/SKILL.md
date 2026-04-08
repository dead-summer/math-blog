---
name: typst-guide
description: |
  Assist in creating and editing Typst math notes that conform to this project's
  conventions. Covers template structure, typesetting elements, math macros,
  cross-reference rules, and file organization for the math-blog project.
allowed-tools:
  - Read
  - Write
  - Edit
  - Grep
  - Glob
  - AskUserQuestion
---

# Typst Guide: math-blog Typst 数学笔记辅助

## 1. 任务说明

本 Skill 用于在 `math-blog` 项目中**创建和编辑** Typst 格式的数学笔记。执行本 Skill 时，你应当：

- 按照下述文件结构约定放置新文件
- 使用项目标准模板头部
- 遵循排版元素规范（标题、公式、图表、参考文献等）
- 复用项目已有的数学宏定义

如果用户未指定具体操作，询问用户是要**创建新笔记**还是**编辑已有笔记**。

## 2. 文件结构约定

新笔记涉及以下目录（`{slug}` 为笔记的英文短标识，使用 kebab-case）：

| 类型     | 路径                                   |
| -------- | -------------------------------------- |
| 文章源码 | `content/article/{slug}.typ`           |
| 图片资源 | `public/images/{slug}/`                |
| 参考文献 | `public/reference/{slug}.bib`          |

- 文章源码为必须；图片目录和 `.bib` 文件按需创建。
- 所有路径在 Typst 源码中以 `/` 开头表示项目根目录（如 `/public/images/...`）。

## 3. 模板结构

每篇文章以如下标准头部开始：

```typst
#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "文章标题",
  author: "summer",
  desc: [文章简要描述],
  date: "YYYY-MM-DD",
  tags: (
    blog-tags.xxx,
    blog-tags.yyy,
  ),
  show-outline: true,
)
```

### 可用标签

标签定义于 `typ/templates/mod.typ` 的 `blog-tags` 字典中：

- `scientific-computing` — Scientific Computing
- `machine-learning` — Machine Learning
- `numerical-methods` — Numerical Methods
- `pde` — PDE
- `mechanics` — Mechanics
- `mathematics` — Mathematics
- `misc` — Miscellaneous

使用方式：`blog-tags.scientific-computing`。如需新标签，在 `typ/templates/mod.typ` 的 `blog-tags` 字典中添加。

### 语言变体

- 默认（英文）：`main.with(...)`
- 中文笔记：`main-zh.with(...)`（自动设置 `lang: "zh", region: "cn"`）

## 4. 排版元素规范

### 4.1 标题层级

```typst
= 一级标题
== 二级标题
=== 三级标题
==== 四级标题
```

标题自动编号（格式 `1.1`），无需手动添加编号。

### 4.2 数学公式

**行内公式**：用 `$...$` 包裹，紧贴美元符号无空格。

```typst
其中 $bold(x) in RR^3$ 是空间坐标。
```

**行间公式**：美元符号与内容之间用换行或空格分隔。行间公式是语句的一部分，**必须包含适当的标点符号**（逗号、句号等）。

```typst
定义能量泛函为
$
  I(u) = integral_Omega ( 1/2 |nabla u|^2 - f u ) dif bold(x).
$
```

**带标签的公式**（可交叉引用）：

```typst
$ -Delta u(bold(x)) = f(bold(x)), quad bold(x) in Omega, $<eq:poisson>
```

**引用公式**：`@eq:poisson`

**公式编号规则**：仅带 `<eq:label>` 标签的行间公式会自动编号，无标签的行间公式不编号。

### 4.3 图片

```typst
#figure(
  image("/public/images/{slug}/figure-name.png"),
  caption: [图片说明文字],
) <fig:figure-name>
```

- 引用：`@fig:figure-name`
- 可选参数 `width`：`image("...", width: 90%)`

### 4.4 表格（三线表）

```typst
#figure(
  three-line-table(
    columns: N,
    align: (right, left, ...),
  )[
    | 列1 | 列2 | ... |
    |-----|-----|-----|
    | 数据 | 数据 | ... |
  ],
  caption: [表格说明文字],
) <tb:table-name>
```

- 引用：`@tb:table-name`
- `three-line-table` 来自 `tablem` 包，已在模板中导入
- `columns` 指定列数，`align` 指定每列对齐方式

### 4.5 参考文献

**引用**：`@citekey`（citekey 对应 `.bib` 文件中的条目标识符）

**文末添加参考文献列表**：

```typst
#bibliography("/public/reference/{slug}.bib")
```

### 4.6 定理环境

来自 `theorion` 包（`@preview/theorion:0.4.1`），已在模板中通过 `macros.typ` 导入。

```typst
#proposition[
  命题内容
]

#proof[
  证明过程
]
```

其他可用环境：`#theorem`, `#lemma`, `#corollary`, `#definition`, `#remark`, `#example` 等。

### 4.7 交叉引用约定

| 元素类型 | 标签格式           | 引用方式          |
| -------- | ------------------ | ----------------- |
| 公式     | `<eq:name>`        | `@eq:name`        |
| 图片     | `<fig:name>`       | `@fig:name`       |
| 表格     | `<tb:name>`        | `@tb:name`        |

### 4.8 其他排版要素

- **强调**：`*斜体强调*`
- **有序列表**：`+ 项目` 或 `1. 项目`
- **无序列表**：`- 项目`
- **代码行内**：`` `code` ``
- **代码块**：` ```lang ... ``` `

## 5. 数学符号

### Typst 常用数学语法

| 语法                  | 效果说明              |
| --------------------- | --------------------- |
| `bold(x)`             | 粗体向量              |
| `hat(u)`              | 上标帽子              |
| `tilde(u)`            | 上标波浪              |
| `nabla`               | 梯度算子              |
| `Delta`               | 拉普拉斯算子          |
| `partial`             | 偏导符号              |
| `integral_Omega`      | 积分                  |
| `sum_(i=1)^n`         | 求和                  |
| `dif bold(x)`         | 微分元素              |
| `norm(x)_2`           | 范数                  |
| `abs(x)`              | 绝对值                |
| `RR`, `NN`, `ZZ`      | 数集                  |
| `quad`                | 空白间距              |
| `cases(...)`          | 分段函数/方程组       |
| `mat(...)`            | 矩阵                  |

### 项目自定义宏

定义于 `typ/templates/macros.typ`：

| 宏名        | 含义               |
| ----------- | ------------------ |
| `indicator` | 指示函数 **1**     |
| `div`       | 散度 div           |
| `span`      | 张成 span          |
| `diag`      | 对角 diag          |

如需新的数学符号宏，在 `typ/templates/macros.typ` 中按相同格式添加。

## 6. 完整示例

以下为一篇最小可运行的笔记模板：

```typst
#import "/typ/templates/blog.typ": *
#show: main.with(
  title: "示例笔记标题",
  author: "summer",
  desc: [这是一篇示例笔记],
  date: "2026-03-09",
  tags: (
    blog-tags.mathematics,
  ),
  show-outline: true,
)

= 引言

本文讨论了 $f(x) = 0$ 的求解方法。

= 问题描述

考虑如下方程：
$ -Delta u(bold(x)) = f(bold(x)), quad bold(x) in Omega, $<eq:poisson>
其中 $Omega subset RR^d$ 是计算域。

@eq:poisson 是经典的 Poisson 方程。

== 实验设置

#figure(
  three-line-table(
    columns: 2,
    align: (right, left),
  )[
    | 参数 | 说明 |
    |------|------|
    | 问题域 | $Omega = [0, 1]^2$ |
    | 网格 | $64 times 64$ |
  ],
  caption: [实验参数设置],
) <tb:setup>

如 @tb:setup 所示，我们采用标准配置。

== 实验结果

#figure(
  image("/public/images/example/result.png"),
  caption: [实验结果可视化],
) <fig:result>

@fig:result 展示了最终的数值解。

= 总结

本文验证了方法的有效性。
```
