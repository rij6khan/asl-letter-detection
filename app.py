from flask import Flask

app = Flask(__name__)

@app.route("/")

def website():
    return "<p>Test</p>"