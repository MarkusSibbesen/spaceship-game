import torch
from hook_manager import HookManager
from transformers import AutoModelForCausalLM, AutoTokenizer
import keyboard
import threading
import time

scalar_value = 0

def key_listener():
    """Thread function to listen for keypresses"""
    global modification_active, scalar_value

    def on_up_press(event):
        global scalar_value
        if event.event_type == keyboard.KEY_DOWN and event.name == 'pil op':
            scalar_value = min(20, scalar_value + 1)
    
    def on_down_press(event):
        global scalar_value
        if event.event_type == keyboard.KEY_DOWN and event.name == 'pil ned':
            scalar_value = max(-20, scalar_value - 1)
    
    
    # Register the hotkeys
    keyboard.hook(on_up_press)
    keyboard.hook(on_down_press)
    
    # Keep the thread alive
    time.sleep(0.1)


def color_text(text, scalar):

    scalar = max(-10, min(10, scalar))
    
    # Calculate RGB values based on scalar
    # Blue (-10) to white (0) to red (10)
    if scalar < 0:
        # From blue to white
        normalized = 1 + scalar / 10  # 0 at -10, 1 at 0
        r = int(255 * normalized)
        g = int(255 * normalized)
        b = 255
    else:
        # From white to red
        normalized = scalar / 10  # 0 at 0, 1 at 10
        r = 255
        g = int(255 * (1 - normalized))
        b = int(255 * (1 - normalized))
    
    return f"\033[48;2;{r};{g};{b}m{text}\033[0m"


def steer_model(model, input_ids, layer, steering_vector, scalar, past_key_values=None):

    with HookManager(model) as hook_manager:
        hook_manager.attach_steering_vector(layer, steering_vector, scalar)

        outputs = model(input_ids, use_cache=True)

        next_token_logits = outputs.logits[:, -1, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token



def generate_text(prompt, model, tokenizer, max_length, steering_vector, layer):
    global modification_active, scalar_value
    
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
            next_token = steer_model(model, input_ids, layer, steering_vector, scalar_value)
        
        # Add the token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Decode the new token
        token_text = tokenizer.decode(next_token[0])
        generated_text += token_text
        
        # Print the token
        print(color_text(token_text, scalar=scalar_value), end="", flush=True)
        
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

    steering_vector = torch.load('steering_vectors/raw_means/layer3_label3.pt', weights_only=True)
    # steering_vector = torch.rand(768)
    
    print("\nInteractive Text Generation")
    print("---------------------------")
    print("Controls:")
    print("  - UP/DOWN: adjust steering lambda")
    
    prompt = input("\nEnter a prompt: ")
    
    generate_text(prompt, model, tokenizer, 1000, steering_vector, 3)


if __name__ == '__main__':
    main()