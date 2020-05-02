import re
import gensim
import jieba
from tensorflow import keras
from flask import Flask,render_template,request
from wtforms import Form,TextAreaField,validators
from train import build_model, build_embeddings_matrix


app = Flask(__name__)
app.debug = True

def classify_review(text):
    text = re.sub("<[^>]*>", "", text)
    word_vec_model = gensim.models.KeyedVectors.load_word2vec_format("data/word_vec.txt", binary=False)
    word_index, embeddings_matrix = build_embeddings_matrix(word_vec_model)
    model = build_model(word_index, embeddings_matrix)
    model.load_weights("./model/o2o_model")
    text = [word_index.get(word, 0) for word in jieba.cut(text)]
    text = keras.preprocessing.sequence.pad_sequences([text], maxlen=20, padding="post", truncating="post",
                                                    dtype="float32")
    res = model.predict(text)[0][0]
    if res < 0.5:
        lable = "与食品安全无关"
    else:
        lable = "与食品安全相关"

    return lable

@app.route('/')
def index():
    #验证用户输入的文本是否有效
    form = ReviewForm(request.form)
    # print("HAHAHA")
    return render_template("index.html",form=form)


@app.route("/main",methods=["GET", "POST"])
def main():
    form = ReviewForm(request.form)
    if request.method == "GET":
        return render_template("index.html")
    else:
        if form.validate():
            # print("xiixiixxi")
            # 获取表单提交的评论
            review_text = request.form["review"]
            # 获取评论的分类结果,类标、概率
            lable = classify_review(review_text)
            # 将分类结果返回给界面进行显示
            return render_template("reviewform.html",review=review_text,label=lable)
        else:
            return "验证失败"


class ReviewForm(Form):
    # 表单验证
    review = TextAreaField("",[validators.DataRequired()])


if __name__ == '__main__':
    app.run(debug=True)
