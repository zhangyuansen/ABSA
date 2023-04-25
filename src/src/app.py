from flask import Flask,render_template, request,url_for
from run_predict import API

app = Flask(__name__)
api=API()

@app.route('/',methods=['GET',"POST"])
def index():
    if request.method == "GET":
        return render_template("index.html",input_text ="",output="输出结果将显示在这里")
    if request.method == "POST":
        input_str = request.form['input_text']
        result = api.api(input_str)
        return render_template("index.html",input_text =input_str,output=result)

if __name__ == '__main__':

    app.run(host='0.0.0.0', port=8080)