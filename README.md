# ThemeStartPredictor

基于 BERT 的主题起始位置预测模型，用于从文本中判断主要内容的开始位置（字符级索引）。该项目适用于 Markdown 文章等内容结构清晰的文本，可用于摘要、分段或内容识别任务。

---

## 项目结构

```

.
├── config.py                 # 预训练模型路径配置等常量
├── train.py                 # 训练主程序，支持命令行参数
├── inference.py             # 推理脚本，支持输入文本或文本文件
├── dataset.py               # 自定义 Dataset 构建逻辑
├── models.py                # 模型结构定义（基于 BERT）
├── utils.py                 # 文本读取等辅助函数
├── data/
│   ├── train/               # 训练用 Markdown 文本
│   └── test/                # 测试用 Markdown 文本
└── weights/                 # 模型保存目录

````

---

## 功能简介

- 返回预测的主题起始字符位置，并显示片段内容
- **目前最大支持token长度512**

---

## 安装依赖

建议使用 Python 3.9+，先创建虚拟环境，然后安装：

**注意：** 默认的依赖是torch+cuda12.8 可能需要依据实际情况修改，如：或者直接使用cpu版本，具体可以参考torch官网

```bash
pip install -r requirements.txt
````

---

## 推理使用（`inference.py`）

### 方式一：输入文本字符串

```bash
python inference.py \
  --model_path weights/theme_start_model.pt \
  --text "这是文章的开头部分。模型应该找出主要内容从哪里开始。"
```

### 方式二：读取文件内容进行预测

```bash
python inference.py \
  --model_path weights/theme_start_model.pt \
  --file ./data/test/sample1.md
```

### 示例输出

```
📍 预测字符起始位置: 18
📎 预测片段:
本文主要介绍如何使用 BERT 模型完成文本主体提取任务。
```

---

## 训练模型（`train.py`）

支持以下参数：

* `--max_batches`：训练总批次数（如：100）
* `--batch_size`：每批样本数量（如：4）
* `--save_path`：保存模型的路径（如：weights/model.pt）
* `--resume`：是否加载已有模型继续训练

### 示例命令：

```bash
python train.py \
  --max_batches 200 \
  --batch_size 4 \
  --save_path weights/theme_start_model.pt
```

### 继续训练：

```bash
python train.py \
  --max_batches 50 \
  --batch_size 4 \
  --save_path weights/theme_start_model.pt \
  --resume
```

---

## 数据格式说明

* 每个 `.md` 文件表示一条训练数据
* 文本中需包含 `<ai-theme>` 标签，表示模型需要学习预测的起始位置
* 示例：

```markdown
 大家好，今天是个风和日丽的好日子，在此时刻，我想给大家分享一下大模型的部署：
 <ai-theme>
 ## 本地大模型部署指引
 #### 目录
 1. 准备环境
 2. 下载模型
 ....
 </ai-theme>
```

模型训练时会根据该标签自动生成标注索引，推理时无需该标签。
