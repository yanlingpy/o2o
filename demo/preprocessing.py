import pandas as pd
import jieba as jb


def stop_words():
    """
    停用词
    :return:stop_words
    """
    with open("data/stop_words.txt", "r", encoding="utf-8") as fr:
        stopwords = set([word.strip() for word in fr])

    return stopwords


if __name__ == '__main__':
    # 加载停用词
    stop_words = stop_words()

    # 读取训练集数据
    df = pd.read_csv("data/train.csv", sep="\t")

    # 分词并去除停用词
    df["comment"] = df["comment"].map(lambda x: " ".join([i for i in jb.cut(x) if i not in stop_words]))

    # 保存文件，成新的训练集
    df.to_csv("data/new_train.csv", index=False, header=False, columns=["label", "comment"])
