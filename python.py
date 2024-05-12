from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

# Load the model (this should be done once, not on every request)
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Luna-AI-Llama2-Uncensored-GGML", model_file="luna-ai-llama2-uncensored.ggmlv3.q4_K_M.bin", gpu_layers=50)

@app.route('/generate', methods=['POST'])
def generate_text():
    content = request.json
    prompt = content['prompt']
    response = llm(prompt)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
