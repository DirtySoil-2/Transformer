import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, List


class WordEmbedding(nn.Module):
    """
    词嵌入模块，将输入的词索引转换为词向量表示。
    Args:
        vocab_size: 词表大小。
        embedding_size: 词嵌入维度。
        padding_idx: 用于填充的索引值。
    """
    def __init__(self, vocab_size: int, embedding_size: int, padding_idx: int = 0):
        super(WordEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.word_embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=padding_idx)
        self.word_embedding.weight.data.normal_(0.0, self.embedding_size ** -0.5) 

    def forward(self, word_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            word_ids: 输入的词索引张量
        Returns:
            word_embedding: 缩放后的词嵌入向量
        """
        word_embedding = self.word_embedding(word)
        word_embedding = self.embedding_size ** 0.5 * word_emb
        return word_embedding
    

class SegmentEmbedding(nn.Module):
    """
    分段嵌入模块，用于区分不同的文本段落。
    Args:
        segment_size: 分段大小。
        embedding_size: 嵌入维度。
    """
    def __init__(self, segment_size: int, embedding_size: int):
        super(SegmentEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.segment_embedding = nn.Embedding(segment_size, embedding_size)
        self.segment_embedding.weight.data.normal_(0.0, self.embedding_size ** -0.5)

    def forward(self, segment_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            segment_ids: 输入的分段索引张量
        Returns:
            segment_embedding: 缩放后的分段嵌入向量
        """
        segment_embedding = self.segment_embedding(segment_ids)
        segment_embedding = self.embedding_size ** 0.5 * segment_embedding
        return segment_embedding


def get_sinusoid_encoding(position_size: int, hidden_size: int) -> np.ndarray:
    """
    生成正弦位置编码。
    Args:   
        position_size: 位置编码的最大长度
        hidden_size: 隐藏层维度
    Returns:
        sinusoid: 正弦位置编码矩阵
    """
    def calculate_angle(position_id: int, hidden_idx: int) -> float:
        # i = hidden_idx // 2
        return position_id / np.power(10000, 2 * (hidden_idx // 2) / hidden_size)

    def get_position_angle_vector(position_id: int) -> List:
        return [calculate_angle(position_id, hidden_id) for hidden_id in range(hidden_size)]

    sinusoid = np.array([get_position_angle_vector(position_id) for position_id in range(position_size)])
    sinusoid[:, 0::2] = np.sin(sinusoid[:, 0::2])
    sinusoid[:, 1::2] = np.cos(sinusoid[:, 1::2])
    return sinusoid.astype("float32")


class PositionalEmbedding(nn.Module):
    """
    位置编码模块，为序列中的每个位置生成唯一的编码。
    Args:
        max_length: 位置编码的最大长度
        embedding_size: 嵌入维度
    """
    def __init__(self, max_length: int, embedding_size: int):
        super(PositionalEmbedding, self).__init__()
        self.embedding_size = embedding_size
        self.position_encoder = nn.Embedding(max_length, embedding_size)
        encoding = get_sinusoid_encoding(max_length, embedding_size)
        encoding = torch.from_numpy(encoding)
        self.position_encoder.weight.data.copy_(encoding)
    
    def forward(self, position_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            position_ids: 输入的位置索引张量
        Returns:
            position_embedding: 位置嵌入向量
        """
        position_embedding = self.position_encoder(position_ids)
        position_embedding = position_embedding.detach()
        position_embedding.requires_grad_(False)
        return position_embedding


class TransformerEmbeddings(nn.Module):
    """
    Transformer嵌入层，整合词嵌入、位置编码和分段编码。
    Args:
        vocab_size: 词表大小
        hidden_size: 隐藏层维度
        hidden_dropout_prob: dropout 概率
        position_size: 位置编码最大长度
        segment_size: 分割词表大小
    """
    def __init__(
        self, 
        vocab_size: int, 
        hidden_size: int = 768, 
        hidden_dropout_prob: float = 0.1, 
        position_size: int = 512,
        segment_size: int = 2
    ):
        super(TransformerEmbeddings, self).__init__()
        self.word_embeddings = WordEmbedding(vocab_size, hidden_size)
        self.position_embeddings = PositionalEmbedding(position_size, hidden_size)
        self.segment_embeddings = SegmentEmbedding(segment_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(hidden_dropout_prob)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: Optional[torch.Tensor] = None, 
        position_ids: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 输入词索引张量
            segment_ids: 分段索引张量
            position_ids: 位置索引张量
        Returns:
            embeddings: 整合后的嵌入向量
        """
        if position_ids is None:
            ones = torch.ones_like(input_ids, dtype=torch.int64)
            seq_length = torch.cumsum(ones, dim=-1)
            position_ids = seq_length - ones
            position_ids.requires_grad_(False)
        input_embeddings = self.word_embeddings(input_ids)
        segment_embeddings = self.segment_embeddings(segment_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = input_embeddings + segment_embeddings + position_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings
    