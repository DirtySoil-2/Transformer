"""
Transformer 模型实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Union

from embedding import TransformerEmbeddings


device = torch.device('cuda')


class QKVAttention(nn.Module):
    """
    多头注意力QKV计算。
    计算注意力权重并应用于值矩阵，实现注意力机制的核心计算。
    """
    def __init__(self, head_size: int, device: Optional[torch.device] = None):
        super(QKVAttention, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        head_size = torch.tensor([head_size], dtype=torch.float32, device=self.device)
        self.sqrt_size = torch.sqrt(head_size)

    def forward(self, Q, K, V, valid_lens):
        """
        Args:
            Q: 查询矩阵, [batch_size, heads_num, seq_len, head_size]
            K: 键矩阵, [batch_size, heads_num, seq_len, head_size]
            V: 值矩阵, [batch_size, heads_num, seq_len, head_size]
            valid_len: 有效长度, [batch_size, seq_len]
        Returns:
            context: 注意力计算结果, [batch_size, heads_num, seq_len, head_size]
            attention_weights: 注意力权重, [batch_size, heads_num, heads_num, seq_len, seq_len]
        """
        batch_size, heads_num, seq_len, head_size = Q.size()
        # Q:[batch_size, heads_num, 1, seq_len, head_size]
        Q = Q.reshape(batch_size, heads_num, 1, seq_len, head_size)
        # K:[batch_size, heads_num*heads_num, seq_len, head_size]
        K = K.repeat([1, heads_num, 1, 1])
        # K:[batch_size, heads_num, heads_num, seq_len, head_size]
        K = K.reshape(batch_size, heads_num, heads_num, seq_len, head_size)
        # score:[batch_size, heads_num, heads_num, seq_len, seq_len]
        score = torch.matmul(Q, K.transpose(3, 4)) / self.sqrt_size
        # attention_weights:[batch_size, heads_num, heads_num, seq_len, seq_len]
        attention_weights = F.softmax(score, -1)
        self._attention_weights = attention_weights
        # 加权平均
        # V:[batch_size, heads_num*heads_num, seq_len, head_size]
        V = V.repeat([1, heads_num, 1, 1])
        # K:[batch_size, heads_num, heads_num, seq_len, head_size]
        V = V.reshape(batch_size, heads_num, heads_num, seq_len, head_size)
        # B:[batch_size, heads_num, heads_num, seq_len, head_size]
        B = torch.matmul(attention_weights, V)
        # context:[batch_size, heads_num, seq_len, head_size]
        context = torch.sum(B, dim=2)
        # context:[batch_size, seq_len, heads_num, head_size]
        context = context.permute(0, 2, 1, 3)
        # context:[batch_size, seq_len, heads_num*head_size]
        context = context.reshape(batch_size, seq_len, heads_num*head_size)
        return context, attention_weights
    

class MultiHeadSelfAttention(nn.Module):
    """
    多头自注意力机制。
    将输入分割为多个头，分别计算注意力，然后合并结果。
    """
    def __init__(self, inputs_size: int, heads_num: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super(MultiHeadSelfAttention, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.heads_num = heads_num
        self.head_size = inputs_size // heads_num
        self.middle_size = int(4 * inputs_size / 3)
        assert(self.head_size * heads_num == inputs_size), "输入维度必须能被头数整除"
        self.Q_proj = self._create_projection_layer()
        self.K_proj = self._create_projection_layer()
        self.V_proj = self._create_projection_layer()
        self.out_proj = self._create_projection_layer()
        self.attention = QKVAttention(self.head_size, device=self.device)
        
    def _create_projection_layer(self) -> nn.Sequential:
        """创建投影层，使用卷积网络处理输入"""
        return nn.Sequential(
            nn.Conv2d(1, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        ).to(self.device)

    def split_head_reshape(self, X: torch.Tensor, heads_num: int, head_size: int) -> torch.Tensor:
        """
        Args:
            X: 输入张量, [batch_size, 1, seq_len, hidden_size]
            heads_num: 头数
            head_size: 每个头的维度
        Returns:
            X: 多头重组后的张量, [batch_size, heads_num, seq_len, head_size]
        """
        batch_size, i, seq_len, hidden_size = X.shape
        # X:[batch_size, seq_len, hidden_size]
        X = X.reshape(batch_size*i, seq_len, hidden_size)
        # 多头重组
        # X:[batch_size, seq_len, heads_num, head_size]
        X = torch.reshape(X, shape=[batch_size, seq_len, heads_num, head_size])
        # 形状重组
        # X:[batch_size, heads_num, seq_len, head_size]
        X = X.permute(0, 2, 1, 3)
        return X 


    def forward(self, X: torch.Tensor, valid_lens: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: 输入矩阵, [batch_size,seq_len,hidden_size]
            valid_lens: 有效长度矩阵, [batch_size]
        输出:
            out:输出矩阵, 多头注意力的结果
            atten_weights: 注意力权重
        """
        self.batch_size, self.seq_len, self.hidden_size = X.shape
        # X:[batch_size, 1, seq_len, hidden_size]
        X = X.reshape(self.batch_size, 1, self.seq_len, self.hidden_size)
        # Q, K, V:[batch_size, 1, seq_len, hidden_size]
        Q, K, V = self.Q_proj(X), self.K_proj(X), self.V_proj(X)
        # Q, K, V:[batch_size, heads_num, seq_len, head_size]
        Q = self.split_head_reshape(Q, self.heads_num, self.head_size)
        K = self.split_head_reshape(K, self.heads_num, self.head_size)
        V = self.split_head_reshape(V, self.heads_num, self.head_size)
        # out:[batch_size, seq_len, heads_num*head_size]
        out, atten_weights = self.attention(Q, K, V, valid_lens)
        batch_size, seq_len, hidden_size = out.shape
        # out:[batch_size, 1, seq_len, heads_num*head_size]
        out = out.reshape(batch_size, 1, seq_len, hidden_size)
        out = self.out_proj(out)
        # out:[batch_size, seq_len, heads_num*head_size]
        out = out.reshape(batch_size, seq_len, hidden_size)
        return out, atten_weights


class PositionwiseFeedForward(nn.Module):
    """
    逐位前馈层。
    对序列中的每个位置独立应用相同的前馈网络。
    """
    def __init__(self, input_size: int, mid_size: int, dropout: float = 0.1, device: Optional[torch.device] = None):
        super(PositionwiseFeedForward, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.features = nn.Sequential(
            nn.Linear(input_size, mid_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mid_size, input_size)
        ).to(self.device)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: 输入张量
        Returns:
            output: 前馈网络输出
        """
        return self.features(X)


class AddNorm(nn.Module):
    """
    加与规范化层。
    实现残差连接和层归一化。
    """
    def __init__(self,  size: int, dropout: float, device: Optional[torch.device] = None):
        super(AddNorm, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.layer_norm = nn.LayerNorm(size).to(self.device)
        self.dropout = nn.Dropout(dropout).to(self.device)

    def forward(self, X: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """
        Args:
            X: 残差连接的输入
            H: 子层的输出
        Returns:
            out: 加与规范化的输出
        """
        H = X + self.dropout(H)
        return self.layer_norm(H)
    

class TransformerBlock(nn.Module):
    """
    Transformer 编码器块。
    包含多头自注意力机制和前馈网络，以及残差连接和层归一化。
    """
    def __init__(
        self, 
        input_size: int, 
        head_num: int, 
        pff_size: int, 
        an_dropout: float = 0.1, 
        attention_dropout: Optional[float] = None, 
        pff_dropout: Optional[float] = None, 
        device: Optional[torch.device] = None
    ):
        """
        Args:
            input_size: 输入数据维度
            head_num: 多头自注意力多头个数
            pff_size: 逐位前馈层的大小
            an_dropout: 加与规范化 dropout 参数
            attn_dropout: 多头注意力的 dropout 参数
            ppf_dropout: 逐位前馈层的 dropout 参数
        """
        super(TransformerBlock, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.attention_dropout = an_dropout if attention_dropout is None else attention_dropout
        self.pff_dropout = an_dropout if pff_dropout is None else pff_dropout
        # 多头自注意力机制
        self.multi_head_attention = MultiHeadSelfAttention(input_size, head_num, dropout=self.attention_dropout, device=self.device)
        self.pff = PositionwiseFeedForward(input_size, pff_size, dropout=self.pff_dropout, device=self.device)  
        self.addnorm1 = AddNorm(input_size, an_dropout, device=self.device)
        self.addnorm2 = AddNorm(input_size, an_dropout, device=self.device)

    def forward(self, X: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            X: 输入张量
            src_mask: 源序列的掩码
        Returns:
            X: 编码器块的输出
            attention_weights: 多头注意力的权重
        """
        X_atten, attention_weights = self.multi_head_attention(X, src_mask)  # 多头注意力
        X = self.addnorm1(X, X_atten)  # 加与规范化
        X_pff = self.pff(X)  # 前馈层
        X = self.addnorm2(X, X_pff)  # 加与规范化
        return X, attention_weights
    

class Transformer(nn.Module):
    """
    Transformer 编码器模型。
    包含多个Transformer编码器块，用于序列编码和分类任务。
    """
    def __init__(
        self,
        vocab_size: int, 
        n_block: int = 2, 
        hidden_size: int = 768, 
        heads_num: int = 4, 
        intermediate_size: int = 3072, 
        hidden_dropout: float = 0.1, 
        attention_dropout: float = 0.1, 
        pff_dropout: float = 0, 
        position_size: int = 512, 
        num_classes: int = 2, 
        padding_idx: int = 0, 
        device: Optional[torch.device] = None
    ):
        """
        Args:
            vocab_size: 词表大小。
            n_block: Transformer 编码器数目。
            hidden_size: 每个词映射成稠密向量的维度。
            heads_num: 头的数量。
            intermediate_size: 逐位前馈层的维度。
            hidden_dropout: Embedding 层的dropout。
            attention_dropout: 多头注意力的 dropout。
            position_size: 位置编码大小。
            num_classes: 类别数。
            pff_dropout: 逐位前馈层的 dropout。
            padding_idx: 填充字符的id。
        """
        super(Transformer, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.padding_idx = padding_idx
        # 嵌入层
        self.embeddings = TransformerEmbeddings(vocab_size, hidden_size, hidden_dropout, position_size).to(self.device)
        # Transformer的编码器
        self.layers = nn.ModuleList([
            TransformerBlock(
                hidden_size, heads_num, intermediate_size, 
                an_dropout=hidden_dropout, attention_dropout=attention_dropout, 
                pff_dropout=pff_dropout, device=self.device
            ) for _ in range(n_block)
        ])
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_classes)
        ).to(self.device)
        
        self.to(self.device)

    def forward(
        self, 
        input_ids: torch.Tensor, 
        segment_ids: torch.Tensor, 
        position_ids: Optional[torch.Tensor] = None, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            input_ids: 输入词索引
            segment_ids: 分段索引
            position_ids: 位置索引
            attention_mask: 注意力掩码
        Returns:
            logits: 分类logits
        """
        # 构建Mask矩阵
        if attention_mask is None:
            attention_mask = (input_ids == self.padding_idx) * -1e9
            attention_mask = attention_mask.float()
        # 抽取特征向量
        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids, segment_ids=segment_ids)
        sequence_output = embedding_output
        self._attention_weights = []
        # Transformer的输出和注意力权重的输出
        for encoder_layer in self.layers:
            sequence_output, atten_weights = encoder_layer(sequence_output, src_mask=attention_mask)
            self._attention_weights.append(atten_weights)
        # 选择第0个位置的向量作为句向量
        first_token_tensor = sequence_output[:, 0]
        # 分类器
        return self.classifier(first_token_tensor)
    
    @property
    def attention_weights(self) -> List[torch.Tensor]:
        return self._attention_weights
