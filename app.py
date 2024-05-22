from flask import Flask, render_template, request
import openai
from config import OPENAI_API_KEY

app = Flask(__name__)
openai.api_key = OPENAI_API_KEY

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']
    response = openai.Completion.create(
        engine="davinci-codex",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text

if __name__ == '__main__':
    app.run(debug=True)
