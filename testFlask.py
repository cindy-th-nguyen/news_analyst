from unittest import result
from flask import Flask,render_template,request

app = Flask(__name__)

@app.route("/",methods=["POST","GET"])
def home():
    todoRes = "noVar"
    if request.method == "POST":
        todo = request.form.get("todo")
        print(todo)
    return render_template('testFlask.html', result=todoRes)

if __name__ == '__main__':
	app.run(debug=True)
	
