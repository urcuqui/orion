from flask import Flask, render_template, request, jsonify
import ollama
from libs.utils import clean_response_deepseek
from libs.agent_wrap import get_conversational_model
from prompts import system

app = Flask(__name__)

def get_ollama_response(prompt):
    #llm = agent.get_model()

    #cleaned_response = clean_response_deepseek(ollama.chat(
    #model = "deepseek-r1:7b",    
    
    #)["message"]["content"])    
    llm = get_conversational_model()
    #cleaned_response = clean_response_deepseek(llm.invoke({"question":prompt})["answer"])
    
    response_two = clean_response_deepseek(ollama.chat(
    model = "deepseek-r1:7b",
    messages=[
        {"role": "system", "content": system.SYSTEM_ATLAS},
        #{"role": "user", "content": "I need a set of tools and frameworks for {}".format(prompt)}
        {"role": "user", "content": prompt}
    ]
    )["message"]["content"])
    print("First response sent")
    
    return response_two

    #return cleaned_response


def get_deepseek_response(prompt):
    # Placeholder function - Replace with actual DeepSeek API call
    return f"DeepSeek: {prompt}"

@app.route('/know-your-enemy.html')
def enemey():
    return render_template('know-your-enemy.html')

# @app.route('/know-yourself.html')
# def enemey():
#     return render_template('know-your-self.html')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/red-pill.html')
def red_pill():
    return render_template('red-pill.html')

@app.route('/chat_phishing', methods=['POST'])
def chat_phishing():    
    bot_response = get_ollama_response("Create one phishing email in English.")
    return jsonify({"response": bot_response})

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json.get("message", "")
    #model = request.json.get("model", "ollama")
    
    
    bot_response = get_ollama_response(user_input)
    
    
    return jsonify({"response": bot_response})

if __name__ == '__main__':
    app.run(debug=False)
