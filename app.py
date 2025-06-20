import os
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import requests

# Load API keys
load_dotenv("api_key.env")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")

app = Flask(__name__)

# === AI Model Wrappers ===

def ask_groq(question):
    return make_request("Groq", "https://api.groq.com/openai/v1/chat/completions",
                        {"model": "llama3-70b-8192", "messages": [{"role": "user", "content": question}]},
                        GROQ_API_KEY)

def ask_gemini(question):
    return make_request("Gemini", "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent",
                        {"contents": [{"parts": [{"text": question}]}]},
                        GOOGLE_API_KEY)

def ask_mistral(question):
    return make_request("Mistral", "https://api.together.xyz/v1/chat/completions",
                        {"model": "mistralai/Mixtral-8x7B-Instruct-v0.1", "messages": [{"role": "user", "content": question}]},
                        TOGETHER_API_KEY)

def make_request(model_name, url, data, api_key):
    try:
        headers = {"Content-Type": "application/json"}
        full_url = url if model_name != "Gemini" else f"{url}?key={api_key}"
        if model_name != "Gemini":
            headers["Authorization"] = f"Bearer {api_key}"

        res = requests.post(full_url, headers=headers, json=data)
        if res.status_code == 200:
            if model_name == "Gemini":
                return res.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
            else:
                return res.json()["choices"][0]["message"]["content"].strip()
        return f"Error from {model_name}: {res.status_code} - {res.text}"
    except Exception as e:
        return f"Error while connecting to {model_name}: {str(e)}"

# === Verifier ===

def verify_majority_answer(question, answers):
    prompt = f"""
You are an AI judge. Your job is to read three AI answers and output only the most trustworthy final answer to the question — with no explanation.

Instructions:
- Read the question and the 3 AI answers.
- Choose the best and most accurate answer.
- Then return only the final answer in this exact format: ✅ Final Answer: [short conclusive answer]

Do not explain your decision. Do not write paragraphs. Only output one clear, short, and final answer.

Question: "{question}"

Groq: "{answers['Groq']}"
Gemini: "{answers['Gemini']}"
Mistral: "{answers['Mistral']}"

Again, respond ONLY with: ✅ Final Answer: [short clear verdict]
"""
    return ask_mistral(prompt)

# === Semantic Match Checker ===

def semantic_match(answer, final_answer):
    prompt = f"""
You are an AI semantic verifier. Your task is to check whether the following AI answer means the same thing as the final answer.

If the meaning is the same, return only this: ✅ Match  
If it is not the same, return only this: ❌ Mismatch

Final Answer: "{final_answer}"
AI Answer: "{answer}"

Now reply only with ✅ Match or ❌ Mismatch.
"""
    return ask_mistral(prompt)

# === Routes ===

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/ask_ai", methods=["POST"])
def ask_ai():
    data = request.json
    model = data.get("model")
    question = data.get("question")

    if not model or not question:
        return jsonify({"error": "Missing model or question"}), 400

    try:
        if model == "Groq":
            answer = ask_groq(question)
        elif model == "Gemini":
            answer = ask_gemini(question)
        elif model == "Mistral":
            answer = ask_mistral(question)
        else:
            return jsonify({"error": "Unknown model"}), 400
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/verify", methods=["POST"])
def verify():
    data = request.json
    question = data.get("question")
    answers = data.get("answers")

    if not question or not answers:
        return jsonify({"error": "Missing question or answers"}), 400

    try:
        result = verify_majority_answer(question, answers)
        return jsonify({"final": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/semantic_match", methods=["POST"])
def semantic():
    data = request.json
    answer = data.get("answer")
    final = data.get("final")

    if not answer or not final:
        return jsonify({"error": "Missing answer or final"}), 400

    try:
        verdict = semantic_match(answer, final)
        return jsonify({"verdict": verdict})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)