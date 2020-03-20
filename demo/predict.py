import jieba
import gensim
from tensorflow import keras

from train import build_model, build_embeddings_matrix

if __name__ == '__main__':
    word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("data/word_vec.txt", binary=False)
    word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
    model = build_model(word_index, embeddings_matrix)
    model.load_weights("./model/o2o_model")

    while True:
        text = input("请输入一句话: ")
        text = [word_index.get(word, 0) for word in jieba.cut(text)]
        text = keras.preprocessing.sequence.pad_sequences([text], maxlen=20, padding="post", truncating="post", dtype="float32")
        res = model.predict(text)[0][0]

        print(res)

        # if res == 0:
        #     print("无关")
        # else:
        #     print("有关")

        print()