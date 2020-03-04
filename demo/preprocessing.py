import pandas as pd
import jieba as jb


def stop_words():
    """
    停用词
    :return:stop_words
    """
    with open("stopwords/stop_words") as fr:
        stop_words = set([word.strip() for word in fr])

    return stop_words


if __name__ == '__main__':
    # 加载停用词
    stop_words = stop_words()

    # 读取训练集数据
    df = pd.read_csv("data/train.csv", sep="\t")

    # 分词并去除停用词
    df["comment"] = df["comment"].map(lambda x: " ".join([i for i in jb.cut(x) if i not in stop_words]))

    # 保存文件，成新的训练集
    df.to_csv("./o2ocomment.csv", index=False, header=False, columns=["label", "comment"])