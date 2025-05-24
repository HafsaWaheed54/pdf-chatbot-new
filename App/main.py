from flask import Flask, request, render_template, jsonify
from chatbot import PDFChatbot 
from pdf_reader import extract_text_from_pdf

app = Flask(__name__)
bot = PDFChatbot()

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/upload', methods=['POST'])
def upload():
    pdf = request.files['pdf']
    text = extract_text_from_pdf(pdf)
    bot.ingest_text(text)
    return "PDF uploaded and processed."

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = bot.get_response(user_input)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(debug=True)