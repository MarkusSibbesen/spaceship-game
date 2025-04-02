import torch
from hook_manager import HookManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import keyboard
import threading
import time

vector_names = ['cat', 'dog', 'dragon', 'mountain', 'forest', 'cake']
current_vector = None

def key_listener():
    """Thread function to listen for keypresses"""
    global current_vector

    keyboard_presses = []

    for i in range(len(vector_names)):
        def create_press_function(index=i):  # capture i's value via default argument
            def on_press(event):
                global current_vector
                if event.event_type == keyboard.KEY_DOWN and event.name == str(index + 1):
                    if current_vector != index:
                        print(f' [{vector_names[index]}]', end='')
                    current_vector = index
            return on_press

        keyboard_presses.append(create_press_function()) 
    
    for func in keyboard_presses:
        keyboard.hook(func)

    # Keep the thread alive
    time.sleep(0.1)


def color_text(text, scalar):

    scalar = max(-50, min(50, scalar))
    
    # Calculate RGB values based on scalar
    # Blue (-10) to white (0) to red (10)
    if scalar < 0:
        # From blue to white
        normalized = 1 + scalar / 50  # 0 at -10, 1 at 0
        r = int(255 * normalized)
        g = int(255 * normalized)
        b = 255
    else:
        # From white to red
        normalized = scalar / 50  # 0 at 0, 1 at 10
        r = 255
        g = int(255 * (1 - normalized))
        b = int(255 * (1 - normalized))
    
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def steer_model(model, input_ids, layer, steering_vectors, current_vector, scalar):

    with HookManager(model) as hook_manager:
        if current_vector != None:
            hook_manager.attach_steering_vector(layer, steering_vectors[current_vector], scalar)

        outputs = model(input_ids)

        next_token_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)

        probs[probs < 0.01] = 0
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token



def generate_text(prompt, model, tokenizer, max_length, steering_vectors, layer):
    global current_vector
    
    # Start the key listener thread
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Print initial prompt
    print(f"\nPrompt: {prompt}")

    # Track generated text
    generated_text = ""
    
    # Generate token by token
    for _ in range(max_length):

        with torch.no_grad():
            next_token = steer_model(model, input_ids, layer, steering_vectors, current_vector, 5)
        
        # Add the token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        if input_ids.shape[1] > 50:
            input_ids = input_ids[:, 1:]
        
        # Decode the new token
        token_text = tokenizer.decode(next_token[0])
        generated_text += token_text
        
        # Print the token
        print(token_text, end="", flush=True)
        
        # Small delay to make generation visible
        time.sleep(0.2)



def main():
    # Load model and tokenizer (use a smaller model for speed)
    print("Loading model and tokenizer...")
    model_name = "roneneldan/TinyStories-33M"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Start the key listener thread
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.daemon = True
    listener_thread.start()

    steering_vectors = [
        torch.load(f'steering_vectors/tinystories_vectors/{vector_name}/layer_2.pt', weights_only=True)
        for vector_name in vector_names
    ]
    # steering_vector = torch.rand(768)

    steering_vectors_normed = [
        steering_vector / torch.norm(steering_vector)
        for steering_vector in steering_vectors
    ]
    
    print("\nInteractive Text Generation")
    print("---------------------------")
    print("Controls:")
    print("  - UP/DOWN: adjust steering lambda")
    
    prompt = input("\nEnter a prompt: ")
    
    generate_text(prompt, model, tokenizer, 1000, steering_vectors_normed, 2)


if __name__ == '__main__':
    main()