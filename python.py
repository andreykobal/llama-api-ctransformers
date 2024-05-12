from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM

app = Flask(__name__)

# Load the model (this should be done once, not on every request)
llm = AutoModelForCausalLM.from_pretrained("TheBloke/Llama-2-7B-Chat-GGML", model_file="llama-2-7b-chat.ggmlv3.q3_K_M.bin", gpu_layers=50)

@app.route('/generate', methods=['POST'])
def generate_text():
    content = request.json
    prompt = content['prompt']
    response = llm(prompt)
    print(response)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
