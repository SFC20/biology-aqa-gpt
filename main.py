import os

from flask import Flask, jsonify, render_template, request
from openai import OpenAI
from load_documents import extract_text_from_pdf

# Load the AQA Biology specification once when the app starts
SPEC_TEXT = extract_text_from_pdf("documents/AQA-7401-7402-SP-2015.PDF")

app = Flask(__name__)

client = OpenAI()


def ai_bot_response(prompt):
    messages = build_spec_aware_prompt(prompt)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=300,
        temperature=0.4,
    )
    return response.choices[0].message.content.strip()


def build_spec_aware_prompt(user_question):
    # Get the most relevant parts of the spec (for now just include the whole thing)
    context = SPEC_TEXT[:3000]  # limit characters so GPT doesn't get overloaded

    return [
        {"role": "system", "content": "You are an AQA A-level Biology tutor. Answer questions strictly using the provided specification content."},
        {"role": "user", "content": f"The following is from the AQA Biology spec:\n\n{context}\n\nStudent question: {user_question}"}
    ]


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get_response", methods=["POST"])
def get_response():
    data = request.json
    user_input = data.get("user_input")
    bot_response = ai_bot_response(user_input)
    return jsonify({"response": bot_response})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))  # Render assigns PORT dynamically
    app.run(host="0.0.0.0", port=port)

