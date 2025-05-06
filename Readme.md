<h1 align="center">Transformer 模型实现</h1>

本项目是论文 "[Attention is All You Need](https://arxiv.org/pdf/1706.03762)" 的代码复现，使用 PyTorch 框架实现了 Transformer 模型的核心组件。

## 项目概述

Transformer 是一种基于自注意力机制的神经网络架构，摒弃了传统的循环神经网络结构，完全依赖注意力机制来捕获输入序列的全局依赖关系。本项目实现了 Transformer 的核心组件，包括：

- 多头自注意力机制
- 位置编码
- 前馈神经网络
- 残差连接和层归一化

## 模型架构

### 嵌入层 (Embedding)

嵌入层由三部分组成：

1. **词嵌入 (WordEmbedding)**: 将输入的词索引转换为词向量表示
2. **位置编码 (PositionalEmbedding)**: 为序列中的每个位置生成唯一的编码
3. **分段嵌入 (SegmentEmbedding)**: 用于区分不同的文本段落

位置编码使用正弦和余弦函数生成，确保模型能够感知序列中词的相对位置信息。

### 注意力机制 (Attention)

实现了多头自注意力机制，包括：

1. **QKV注意力计算 (QKVAttention)**: 计算注意力权重并应用于值矩阵
2. **多头自注意力 (MultiHeadSelfAttention)**: 将输入分割为多个头，分别计算注意力，然后合并结果

论文中的映射层是使用线性层实现的，这里使用卷积层代替。

### Transformer 块 (TransformerBlock)

每个 Transformer 块包含：

1. 多头自注意力层
2. 逐位前馈网络层
3. 残差连接和层归一化