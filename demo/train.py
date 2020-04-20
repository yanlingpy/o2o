import gensim
import numpy as np
import pandas as pd

from collections  import defaultdict
from tensorflow import keras
from sklearn.model_selection import train_test_split

def build_embeddings_matrix(word_vec_model):
    # 初始化词向量矩阵
    embeddings_matrix = np.random.random((len(word_vec_model.wv.vocab)+1, 128))

    # 初始化词索引字典
    word_index = defaultdict(dict)

    for index, word in enumerate(word_vec_model.index2word):
        word_index[word] = index + 1

        # 预留０行给查不到的词
        embeddings_matrix[index+1] = word_vec_model.get_vector(word)


    return word_index, embeddings_matrix

def build_model(word_index, embeddings_matrix):
    # 建立模型
    model = keras.Sequential()
    model.add(keras.layers.Embedding(input_dim=len(word_index)+1, output_dim=128, weights=[embeddings_matrix],input_length=20, trainable=False))  # 嵌入层
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(100, dropout=0.1,recurrent_dropout=0.5, return_sequences=True)))  # ｌｓｔｍ层
    model.add(keras.layers.Bidirectional(keras.layers.LSTM(50,dropout=0.1,recurrent_dropout=0.5)))
    # model.add(keras.layers.GlobalAveragePooling1D())
    # model.add(keras.layers.Dense(32,activation="relu"))
    model.add(keras.layers.Dense(1, activation="sigmoid"))
    model.compile(optimizer="rmsprop", loss = 'binary_crossentropy', metrics=["accuracy"])
    model.summary()
    return model

def train_data(word_index):
    df = pd.read_csv("data/new_train.csv", names=["label", "comment"])
    df["word_index"] = df["comment"].astype("str").map(lambda x: np.array([word_index[i] for i in x.split(" ")]))
    # 填充以及截断
    train = keras.preprocessing.sequence.pad_sequences(df["word_index"].values, maxlen=20)
    x_train, x_test, y_train, y_test = train_test_split(train, df["label"].values, test_size=0.2, random_state=1)

    # 分出验证集
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.15)

    return x_train, x_val, x_test, y_train, y_val, y_test


if __name__ == '__main__':
    # 加载词向量
    word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("data/word_vec.txt", binary=False)

    # 建立词索引
    word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)


    # 生成训练集，测试集，验证集
    x_train, x_val, x_test, y_train, y_val, y_test = train_data(word_index)
    # 建立模型
    model = build_model(word_index, embeddings_matrix)
    # 训练
    model.fit(x_train, y_train, epochs=50, validation_data=(x_val, y_val))
    # 评估
    ret = model.evaluate(x_test, y_test)
    print(f"损失: {ret[0]}, 准确率: {ret[1]}")
    # 保存模型
    model.save_weights("model/o2o_model")
