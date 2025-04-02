from flask import Flask, render_template, request, jsonify
from hook_manager import HookManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time

app = Flask(__name__)

model_name = "roneneldan/TinyStories-33M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

layer = 2

vector_names = ['cat', 'dog', 'dragon', 'mountain', 'forest', 'cake']

name_to_idx = {name: idx for idx, name in enumerate(vector_names)}
name_to_idx['none'] = None

input_ids = tokenizer.encode('Once upon a time', return_tensors="pt")

steering_vectors = [
    torch.load(f'steering_vectors/tinystories_vectors/{vector_name}/layer_2.pt', weights_only=True)
    for vector_name in vector_names
]

steering_vectors_normed = [
    steering_vector / torch.norm(steering_vector)
    for steering_vector in steering_vectors
]

current_vector = None

def print_color(color):
    return print(color)

@app.route('/')
def index():
    return render_template('index.html')

# New endpoint to receive the color from the frontend
@app.route('/print_color', methods=['POST'])
def receive_color():
    global current_vector
    data = request.get_json()
    current_vector = name_to_idx[data.get('color', None)]
    return jsonify({'status': 'success'})


@app.route('/get_next_word', methods=['GET'])
def generate_text():
    global input_ids
    with torch.no_grad():
        next_token = steer_model(model, input_ids, layer, steering_vectors_normed, current_vector, 5)
        
        # Add the token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if input_ids.shape[1] > 50:
            input_ids = input_ids[:, 1:]
        
        # Decode the new token
        token_text = tokenizer.decode(next_token[0])
        
    return jsonify({'token': token_text}), 200
    
    
@app.route('/reinit', methods=['POST'])
def reinit():
    global input_ids
    input_ids = tokenizer.encode('Once upon a time', return_tensors="pt")
    return jsonify({'status': 'success'})


def steer_model(model, input_ids, layer, steering_vectors, current_vector, scalar):

    with HookManager(model) as hook_manager:
        if current_vector != None:
            hook_manager.attach_steering_vector(layer, steering_vectors[current_vector], scalar)
            print(current_vector)

        outputs = model(input_ids)

        next_token_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)

        probs[probs < 0.01] = 0
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token



# def generate_text(prompt, model, tokenizer, max_length, steering_vectors, layer):
#     global current_vector
    
#     # Tokenize the prompt
#     input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
#     # Print initial prompt
#     print(f"\nPrompt: {prompt}")

#     # Track generated text
#     generated_text = ""
    
#     # Generate token by token
#     for _ in range(max_length):

#         with torch.no_grad():
#             next_token = steer_model(model, input_ids, layer, steering_vectors, current_vector, 5)
        
#         # Add the token to the input_ids
#         input_ids = torch.cat([input_ids, next_token], dim=-1)
#         if input_ids.shape[1] > 50:
#             input_ids = input_ids[:, 1:]
        
#         # Decode the new token
#         token_text = tokenizer.decode(next_token[0])
#         generated_text += token_text
        
#         # Print the token
#         print(token_text, end="", flush=True)
        
#         # Small delay to make generation visible
#         time.sleep(0.2)

if __name__ == '__main__':
    app.run(debug=True)
