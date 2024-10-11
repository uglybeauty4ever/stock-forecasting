import pandas as pd
import re
import jieba
from collections import defaultdict
from gensim.models import Word2Vec
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
import json


# 1. 数据清洗与去噪
def clean_text(text):
    text = re.sub('<[^>]*>', '', text)  # 去除HTML标签
    text = re.sub(r'[^\w\s]', '', text)  # 去除特殊符号
    return text




def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f)


def remove_stopwords_and_punct(text, stopwords):
    words = jieba.lcut(text)
    return [word for word in words if word not in stopwords]


# 2. 加载情感词典、否定词、副词等
def load_sentiment_resources(sentiment_file, negation_file, degree_file):
    # 加载情感词典
    sentiment_dict = defaultdict(float)  # 默认值为0.0
    with open(sentiment_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()  # 使用split()而不是split(' ')
            if len(parts) == 2:  # 确保每行正好有两个部分
                word, score = parts
                sentiment_dict[word] = float(score)
            else:
                print(f"Warning: Line '{line.strip()}' is invalid and will be skipped.")

    # 加载否定词
    with open(negation_file, 'r', encoding='utf-8') as f:
        negation_words = [line.strip() for line in f]

    # 加载程度副词
    degree_dict = defaultdict(float)  # 默认值为1.0
    with open(degree_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 2:
                word, weight = parts
                degree_dict[word] = float(weight)
            else:
                print(f"Warning: Line '{line.strip()}' is invalid and will be skipped.")

    return sentiment_dict, negation_words, degree_dict


# 3. 分类词汇为情感词、否定词和程度副词
def classify_and_score_words(word_list, sentiment_dict, negation_words, degree_dict):
    sentiment_words = {}  # 情感词
    negation_words_positions = {}  # 否定词
    degree_words_positions = {}  # 程度副词

    for i, word in enumerate(word_list):
        if word in sentiment_dict:
            sentiment_words[i] = sentiment_dict[word]
        elif word in negation_words:
            negation_words_positions[i] = -1  # 否定词反转
        elif word in degree_dict:
            degree_words_positions[i] = degree_dict[word]  # 程度副词权重

    return calculate_weighted_score(sentiment_words, negation_words_positions, degree_words_positions, word_list)


# 4. 计算加权情感得分
def calculate_weighted_score(sentiment_words, negation_words, degree_words, word_list):
    W = 1  # 初始化权重
    score = 0
    sentiment_indices = list(sentiment_words.keys())

    for i in range(len(word_list)):
        if i in sentiment_words:
            score += W * sentiment_words[i]  # 权重 * 情感词得分
            if sentiment_indices.index(i) < len(sentiment_indices) - 1:
                next_sentiment_index = sentiment_indices[sentiment_indices.index(i) + 1]
                # 在当前情感词和下一个情感词之间查找否定词或程度副词
                for j in range(i + 1, next_sentiment_index):
                    if j in negation_words:
                        W *= -1  # 否定词反转权重
                    elif j in degree_words:
                        W *= degree_words[j]  # 程度副词调整权重

    return score


# # 5. 训练Word2Vec模型并扩展情感词典
# def train_word2vec_and_expand(sentiment_dict, corpus_file, seed_words, top_n=10):
#     # 加载语料库
#     with open(corpus_file, 'r', encoding='utf-8') as f:
#         corpus = [line.strip().split() for line in f.readlines()]
#
#     # 训练Word2Vec模型
#     model = Word2Vec(sentences=corpus, vector_size=200, window=5, min_count=5, sg=1)
#     model.save("word2vec_model.model")

    # 提取与种子词相似的词
    similar_words = {}
    for word in seed_words:
        if word in model.wv:
            similar_words[word] = model.wv.most_similar(word, topn=top_n)

    # 扩展情感词典
    for seed, words in similar_words.items():
        for word, similarity in words:
            if similarity > 0.6:  # 设定相似度阈值
                sentiment_dict[word] = "情感"  # 假设情感词典中每个词语都有标签
    return sentiment_dict


def extend_sentiment_dict_with_model(sentiment_dict, seed_words, model_file='word2vec_model.model', top_n=10):
    # 加载已保存的Word2Vec模型
    model = Word2Vec.load(model_file)

    # 扩展情感词典
    for word in seed_words:
        if word in model.wv:
            similar_words = model.wv.most_similar(word, topn=top_n)
            for sim_word, similarity in similar_words:
                if similarity > 0.6:  # 根据相似度阈值过滤
                    sentiment_dict[sim_word] = "情感"  # 假设你需要将相似的词归类为情感词
                    print(f"Adding {sim_word} to sentiment dictionary with similarity: {similarity:.2f}")

    return sentiment_dict

# 6. 预处理主函数
def preprocess_comments(csv_file, stopwords_file, user_dict_file, unwanted_authors, sentiment_file, negation_file,
                        degree_file, seed_words, word2vec_model_file='word2vec_model.model'):
    df = pd.read_csv(csv_file)

    # 加载停用词
    stopwords = load_stopwords(stopwords_file)

    # 加载金融领域分词词典
    jieba.load_userdict(user_dict_file)

    # 加载情感词典、否定词和程度副词
    sentiment_dict, negation_words, degree_dict = load_sentiment_resources(sentiment_file, negation_file, degree_file)

    # 数据清洗
    df['cleaned_text'] = df['comment'].apply(clean_text)

    # 去噪
    #df = remove_noise(df, unwanted_authors)

    # 分词并去停用词
    df['processed_text'] = df['cleaned_text'].apply(lambda text: remove_stopwords_and_punct(text, stopwords))

    # 检查是否有保存的Word2Vec模型，如果有，加载模型并扩展情感词典
    try:
        print("Loading Word2Vec model from:", word2vec_model_file)
        model = Word2Vec.load(word2vec_model_file)

        # 扩展情感词典
        expanded_sentiment_dict = extend_sentiment_dict_with_model(sentiment_dict, seed_words, word2vec_model_file)
    except FileNotFoundError:
        print("Word2Vec model not found, training a new one.")
        # 如果没有保存的模型，则重新训练并保存模型
        #expanded_sentiment_dict = train_word2vec_and_expand(sentiment_dict, 'corpus.txt', seed_words)

    # 计算情感得分
    df['sentiment_score'] = df['processed_text'].apply(
        lambda word_list: classify_and_score_words(word_list, expanded_sentiment_dict, negation_words, degree_dict)
    )

    # 保存预处理后的结果
    df.to_csv('processed_comments_with_sentiment.csv', index=False)
    return df


# 7. 合并情感得分与股票数据
def combine_sentiment_with_stock_data(sentiment_file, stock_data_file):
    sentiment_df = pd.read_csv(sentiment_file)
    stock_df = pd.read_csv(stock_data_file)
    combined_df = pd.merge(stock_df, sentiment_df[['date', 'sentiment_score']], on='date', how='inner')
    combined_df.to_csv('combined_stock_data.csv', index=False)
    return combined_df


# 8. Transformer模型定义
class StockDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.scaler = MinMaxScaler()  # 数据归一化
        self.data[['open', 'high', 'low', 'volume', 'sentiment_score']] = self.scaler.fit_transform(
            self.data[['open', 'high', 'low', 'volume', 'sentiment_score']]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data.iloc[idx]
        features = torch.tensor(item[['open', 'high', 'low', 'volume', 'sentiment_score']].values, dtype=torch.float32)
        label = torch.tensor(item['next_day_close'], dtype=torch.float32)  # 假设目标是预测第二天的收盘价
        return features, label


class TransformerModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, nhead=4, dropout=0.1):
        super(TransformerModel, self).__init__()
        # 定义Transformer的编码层
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        # 定义线性层，用于将Transformer的输出映射到最终的输出
        self.fc = nn.Linear(input_size, 1)  # 假设我们预测单个值：下一天的收盘价

    def forward(self, src):
        # 输入数据的形状 (seq_len, batch_size, input_size)
        src = src.permute(1, 0, 2)  # Transformer expects (sequence_length, batch_size, input_size)

        # 通过Transformer编码层
        transformer_out = self.transformer_encoder(src)

        # 只取最后一个时间步的输出进行预测
        output = transformer_out[-1, :, :]

        # 通过线性层输出预测值
        output = self.fc(output)

        return output

# 9. 训练与评估函数
def train_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()  # 使用均方误差来衡量股票价格预测的误差
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)

    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式
        train_loss = 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)

            optimizer.zero_grad()  # 梯度清零

            # 前向传播
            output = model(features.unsqueeze(0))  # 由于模型期望有时间维度，这里需要加一个维度
            loss = criterion(output.squeeze(), labels)  # 计算损失

            # 反向传播与优化
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # 在每个 epoch 结束后输出训练损失
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.4f}")

        # 验证模型
        val_loss = evaluate_model(model, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {val_loss:.4f}")

def evaluate_model(model, val_loader, criterion, device):
    model.eval()  # 设置模型为评估模式
    val_loss = 0
    with torch.no_grad():  # 禁用梯度计算
        for features, labels in val_loader:
            features, labels = features.to(device), labels.to(device)
            output = model(features.unsqueeze(0))
            loss = criterion(output.squeeze(), labels)
            val_loss += loss.item()
    return val_loss / len(val_loader)


# 10. 主函数
def main():
    # 1. 加载并预处理评论数据
    csv_file = 'comments.csv'
    stopwords_file = 'stopwords.txt'
    user_dict_file = 'user_dict.txt'
    unwanted_authors = ['unknown', 'bot']
    sentiment_file = 'sentiment_dict.txt'
    negation_file = 'negation_words.txt'
    degree_file = 'degree_words.txt'
    seed_words = ['上涨', '下跌', '盈利']  # 种子词，用于扩展情感词典
    word2vec_model_file = 'word2vec_model.model'  # 模型文件路径

    # 预处理评论数据并计算情感得分
    processed_comments = preprocess_comments(csv_file, stopwords_file, user_dict_file, unwanted_authors,
                                             sentiment_file, negation_file, degree_file, seed_words, word2vec_model_file)

    # 2. 加载股票数据并合并情感得分
    stock_data_file = 'stock_data.csv'
    combined_data = combine_sentiment_with_stock_data('processed_comments_with_sentiment.csv', stock_data_file)

    # 3. 将数据拆分为训练集和验证集
    train_size = int(0.8 * len(combined_data))
    train_data = combined_data[:train_size]
    val_data = combined_data[train_size:]

    # 4. 创建数据集和数据加载器
    train_dataset = StockDataset(train_data)
    val_dataset = StockDataset(val_data)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    # 5. 定义模型和训练参数
    input_size = 5  # open, high, low, volume, sentiment_score
    hidden_size = 128
    num_layers = 2
    num_epochs = 20
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TransformerModel(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)

    # 6. 训练模型
    train_model(model, train_loader, val_loader, num_epochs, learning_rate, device)

    # 7. 进行预测（可选）
    test_data = combined_data[train_size:]  # 用于测试的部分数据
    test_loader = DataLoader(StockDataset(test_data), batch_size=1, shuffle=False)

    model.eval()
    predictions = []
    with torch.no_grad():
        for features, _ in test_loader:
            features = features.to(device)
            output = model(features.unsqueeze(0))
            predictions.append(output.item())

    # 输出预测结果
    print("Predictions:", predictions)

if __name__ == '__main__':
    main()