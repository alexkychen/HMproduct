from flask import Flask, jsonify, request, render_template, render_template_string
import json
import numpy as np
import pickle
import pandas as pd


with open("basic_with_bestseller_information.csv", "rb") as f:
    df = pd.read_csv(f)

print(df.shape)
app = Flask(__name__)


@app.route('/', methods=["GET", "POST"])
def index():
    pred = ""
    if request.method == "POST":
        customer_id = request.form["customer_id"]
        pred = df.loc[df['customer_id'] == customer_id, 'prediction'].iloc[0]
    return render_template("index.html", pred=pred)


if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5000)
