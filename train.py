import torch
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
import pandas as pd

# 加载数据文件并进行预处理
file_path = 'data/cmn.txt'  # 请确保数据文件位于该路径下

# 读取文件并处理每一行，提取英文和中文句子
data = []
with open(file_path, 'r', encoding='utf-8') as file:
    for line in file:
        # 每行数据使用制表符分割，提取英文和中文部分
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            english_sentence = parts[0].strip()
            chinese_sentence = parts[1].strip()
            data.append([english_sentence, chinese_sentence])

# 创建 DataFrame 保存提取的句子
df = pd.DataFrame(data, columns=['English', 'Chinese'])

# 将处理后的英文和中文句子分别保存为两个文件
df['English'].to_csv('data/english_sentences.txt', index=False, header=False)
df['Chinese'].to_csv('data/chinese_sentences.txt', index=False, header=False)
# 定义英文和中文的分词器
tokenizer_en = get_tokenizer('basic_english')

# 中文分词器：将每个汉字作为一个 token
def tokenizer_zh(text):

    return list(text)


# 构建词汇表函数
def build_vocab(sentences, tokenizer):
    def yield_tokens(sentences):
        for sentence in sentences:
            yield tokenizer(sentence)
    vocab = build_vocab_from_iterator(yield_tokens(sentences), specials=['<unk>', '<pad>', '<bos>', '<eos>'])
    vocab.set_default_index(vocab['<unk>'])
    return vocab

# 从文件中加载句子
with open('data/english_sentences.txt', 'r', encoding='utf-8') as f:
    english_sentences = [line.strip() for line in f]


with open('data/chinese_sentences.txt', 'r', encoding='utf-8') as f:
    chinese_sentences = [line.strip() for line in f]

# 构建词汇表
en_vocab = build_vocab(english_sentences, tokenizer_en)
zh_vocab = build_vocab(chinese_sentences, tokenizer_zh)

print(f'英文词汇表大小：{len(en_vocab)}')
print(f'中文词汇表大小：{len(zh_vocab)}')

# 将句子转换为索引序列，并添加 <bos> 和 <eos>
def process_sentence(sentence, tokenizer, vocab):
    tokens = tokenizer(sentence)
    tokens = ['<bos>'] + tokens + ['<eos>']

    indices = [vocab[token] for token in tokens]
    return indices




en_sequences = [process_sentence(sentence, tokenizer_en, en_vocab) for sentence in english_sentences]
zh_sequences = [process_sentence(sentence, tokenizer_zh, zh_vocab) for sentence in chinese_sentences]

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

class TranslationDataset(Dataset):
    def __init__(self, src_sequences, trg_sequences):
        self.src_sequences = src_sequences
        self.trg_sequences = trg_sequences

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.src_sequences[idx]), torch.tensor(self.trg_sequences[idx])

def collate_fn(batch):
    src_batch, trg_batch = [], []
    for src_sample, trg_sample in batch:
        src_batch.append(src_sample)
        trg_batch.append(trg_sample)
    src_batch = pad_sequence(src_batch, padding_value=en_vocab['<pad>'], batch_first=True)
    trg_batch = pad_sequence(trg_batch, padding_value=zh_vocab['<pad>'], batch_first=True)
    return src_batch, trg_batch

# 创建数据集
dataset = TranslationDataset(en_sequences, zh_sequences)
import sklearn
# 划分训练集和验证集
from sklearn.model_selection import train_test_split
train_data, val_data = train_test_split(dataset, test_size=0.1)

# 创建数据加载器
batch_size = 32
train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
import torch.nn as nn
import torch


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.scale = d_k ** -0.5  # 缩放因子

    def forward(self, Q, K, V, mask=None):
        """
        :param Q: [batch_size, num_heads, seq_len, d_k]
        :param K: [batch_size, num_heads, seq_len, d_k]
        :param V: [batch_size, num_heads, seq_len, d_k]
        :param mask: [batch_size, 1, 1, seq_len] 或 [batch_size, 1, seq_len, seq_len]
        :return: 注意力输出和注意力权重
        """
        # 计算注意力分数

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [batch_size, num_heads, seq_len, seq_len]




        attn = torch.softmax(scores, dim=-1)  # [batch_size, num_heads, seq_len, seq_len]

        # 计算注意力输出
        output = torch.matmul(attn, V)  # [batch_size, num_heads, seq_len, d_k]


        return output, attn
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0, "d_model 必须能被 num_heads 整除"
        self.d_k = self.d_v = d_model // num_heads
        self.num_heads = num_heads

        # 定义线性变换层
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.fc = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(self.d_k)
        self.dropout = nn.Dropout(0.1)

    def forward(self, Q, K, V, mask=None):
        batch_size = Q.size(0)
        seq_len_Q = Q.size(1)
        seq_len_K = K.size(1)
        seq_len_V = V.size(1)

        # 确保 K 和 V 的序列长度与 Q 一致
        if seq_len_K != seq_len_Q or seq_len_V != seq_len_Q:
            raise ValueError(f"Q, K, V 的序列长度必须一致。Q: {seq_len_Q}, K: {seq_len_K}, V: {seq_len_V}")

        # 线性变换并分头
        Q = self.w_q(Q).view(batch_size, seq_len_Q, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(K).view(batch_size, seq_len_K, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(V).view(batch_size, seq_len_V, self.num_heads, self.d_k).transpose(1, 2)

        # 计算注意力
        if mask is not None:
            mask = mask.unsqueeze(1)
        output, attn = self.attention(Q, K, V, mask=mask)

        # 拼接多头的输出
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_Q, self.num_heads * self.d_k)
        output = self.fc(output)

        return output
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.relu(self.fc1(x))))
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 创建一个 [max_len, d_model] 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        # 奇数和偶数位置分别使用 sin 和 cos
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数位置
        pe = pe.unsqueeze(0)  # 增加 batch 维度
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch_size, seq_len, d_model]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # 自注意力子层
        attn_output = self.self_attn(x, x, x, mask)

        x = x + self.dropout(attn_output)

        x = self.norm1(x)

        # 前馈神经网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm2(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, src_mask=None, trg_mask=None):
        # 掩码多头自注意力子层
        self_attn_output = self.self_attn(x, x, x, trg_mask)
        x = x + self.dropout(self_attn_output)
        x = self.norm1(x)
        # 编码器-解码器注意力子层
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)
        # 前馈神经网络子层
        ffn_output = self.ffn(x)
        x = x + self.dropout(ffn_output)
        x = self.norm3(x)
        return x
class Encoder(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_mask=None):
        # src: [batch_size, src_len]
        x = self.embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, src_mask)
        return x
class Decoder(nn.Module):
    def __init__(self, output_dim, d_model, num_heads, d_ff, num_layers, dropout):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(output_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, output_dim)

    def forward(self, trg, enc_output, src_mask=None, trg_mask=None):
        # trg: [batch_size, trg_len]
        x = self.embedding(trg) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)

        # 确保 enc_output 的序列长度与 x 一致
        if enc_output.size(1) != x.size(1):
            # 如果 enc_output 的序列长度小于 x 的序列长度，填充 enc_output
            if enc_output.size(1) < x.size(1):
                padding = torch.zeros(enc_output.size(0), x.size(1) - enc_output.size(1), enc_output.size(2), device=enc_output.device)
                enc_output = torch.cat([enc_output, padding], dim=1)
            # 如果 enc_output 的序列长度大于 x 的序列长度，截断 enc_output
            else:
                enc_output = enc_output[:, :x.size(1), :]

        for layer in self.layers:
            x = layer(x, enc_output, src_mask, trg_mask)
        output = self.fc_out(x)
        return output

class Transformer(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def make_src_mask(self, src):
        # 生成源序列的掩码，屏蔽填充位置
        src_mask = (src != en_vocab['<pad>']).unsqueeze(1).unsqueeze(2)
        return src_mask  # [batch_size, 1, 1, src_len]

    def make_trg_mask(self, trg):
        # 生成目标序列的掩码，包含填充位置和未来信息
        trg_pad_mask = (trg != zh_vocab['<pad>']).unsqueeze(1).unsqueeze(2)  # [batch_size, 1, 1, trg_len]
        trg_len = trg.size(1)
        trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), device=trg.device)).bool()  # [trg_len, trg_len]
        trg_mask = trg_pad_mask & trg_sub_mask  # [batch_size, 1, trg_len, trg_len]
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_output = self.encoder(src, src_mask)
        output = self.decoder(trg, enc_output, src_mask, trg_mask)
        return output

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# 定义模型参数
input_dim = len(en_vocab)
output_dim = len(zh_vocab)
d_model = 512
num_heads = 8
d_ff = 2048
num_layers = 3
dropout = 0.1

# 实例化编码器、解码器和 Transformer 模型
encoder = Encoder(input_dim, d_model, num_heads, d_ff, num_layers, dropout)
decoder = Decoder(output_dim, d_model, num_heads, d_ff, num_layers, dropout)
model = Transformer(encoder, decoder).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss(ignore_index=zh_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# 定义训练函数
def train(model, dataloader, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for src, trg in dataloader:
        src = src.to(device)
        trg = trg.to(device)
        optimizer.zero_grad()
        output = model(src, trg[:, :-1])  # 输入不包括最后一个词
        output_dim = output.shape[-1]
        output = output.contiguous().view(-1, output_dim)
        trg = trg[:, 1:].contiguous().view(-1)  # 目标不包括第一个词
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 定义验证函数
def evaluate(model, dataloader, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)
            output = model(src, trg[:, :-1])
            output_dim = output.shape[-1]
            output = output.contiguous().view(-1, output_dim)
            trg = trg[:, 1:].contiguous().view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(dataloader)

# 开始训练
n_epochs = 10

for epoch in range(n_epochs):
    train_loss = train(model, train_dataloader, optimizer, criterion)
    val_loss = evaluate(model, val_dataloader, criterion)
    print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
torch.save(model.state_dict(), 'transformer_model.pth')
print("模型已保存到 transformer_model.pth")