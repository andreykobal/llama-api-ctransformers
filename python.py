from flask import Flask, request, jsonify
from ctransformers import AutoModelForCausalLM
import threading

app = Flask(__name__)

# Load the model (this should be done once, not on every request)
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/LLaMA2-13B-Tiefighter-GGUF",
    gpu_layers=50,
    context_length=4096, 
    max_new_tokens=4096
)

# Create a lock
model_lock = threading.Lock()

@app.route('/generate', methods=['POST'])
def generate_text():
    content = request.json
    prompt = content['prompt']
    
    # Acquire the lock before accessing the model
    with model_lock:
        response = llm(prompt)
    
    print(response)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
