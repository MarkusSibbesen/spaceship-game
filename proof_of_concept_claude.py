
print("importing torch")
import torch
print("importing keyboard")
import keyboard
print("importing transformers")
from transformers import AutoModelForCausalLM, AutoTokenizer
print("importing threading")
import threading
print("importing time")
import time

# Global variables to track state
modification_active = False
temperature_value = 1.0
stop_generation = False

def key_listener():
    """Thread function to listen for keypresses"""
    global modification_active, temperature_value, stop_generation
    
    def on_space_press(event):
        global modification_active
        if event.event_type == keyboard.KEY_DOWN and event.name == 'space':
            modification_active = True
        elif event.event_type == keyboard.KEY_UP and event.name == 'space':
            modification_active = False
    
    def on_up_press(event):
        global temperature_value
        if event.event_type == keyboard.KEY_DOWN and event.name == 'pil op':
            temperature_value = min(2.0, temperature_value + 0.1)
            print(f"\n[TEMPERATURE INCREASED: {temperature_value:.1f}]")
    
    def on_down_press(event):
        global temperature_value
        if event.event_type == keyboard.KEY_DOWN and event.name == 'pil ned':
            temperature_value = max(0.1, temperature_value - 0.1)
            print(f"\n[TEMPERATURE DECREASED: {temperature_value:.1f}]")
    
    def on_esc_press(event):
        global stop_generation
        if event.event_type == keyboard.KEY_DOWN and event.name == 'esc':
            stop_generation = True
            print("\n[STOPPING GENERATION]")
    
    # Register the hotkeys
    keyboard.hook(on_space_press)
    keyboard.hook(on_up_press)
    keyboard.hook(on_down_press)
    keyboard.hook(on_esc_press)
    
    # Keep the thread alive
    while not stop_generation:
        time.sleep(0.1)


def generate_text_interactive(prompt, model, tokenizer, max_length=1000):
    """Generate text token by token with interactive modifications"""
    global modification_active, temperature_value, stop_generation
    
    # Start the key listener thread
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    
    # Print initial prompt
    print(f"\nPrompt: {prompt}")
    print("\nGeneration (press SPACE to modify, UP/DOWN to change temperature, ESC to stop):")
    
    # Track generated text
    generated_text = ""
    
    # Generate token by token
    for _ in range(max_length):
        if stop_generation:
            break
        
        # Adjust temperature based on modification state
        temp = temperature_value * 2 if modification_active else temperature_value
        
        # Generate a single token
        with torch.no_grad():
            outputs = model(input_ids)
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature
            next_token_logits = next_token_logits / temp
            
            # Sample from the distribution
            probs = torch.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        
        # Add the token to the input_ids
        input_ids = torch.cat([input_ids, next_token], dim=-1)
        
        # Decode the new token
        token_text = tokenizer.decode(next_token[0])
        generated_text += token_text
        
        # Print the token
        print(token_text, end="", flush=True)
        
        # Small delay to make generation visible
        time.sleep(0.1)
    
    print("\n\nGeneration completed!")
    stop_generation = True
    return generated_text


def main():
    # Load model and tokenizer (use a smaller model for speed)
    print("Loading model and tokenizer...")
    model_name = "EleutherAI/pythia-14m"  # You can use any model you prefer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # Start the key listener thread
    listener_thread = threading.Thread(target=key_listener)
    listener_thread.daemon = True
    listener_thread.start()
    
    print("\nInteractive Text Generation")
    print("---------------------------")
    print("Controls:")
    print("  - SPACE: Hold to modify generation (increases temperature)")
    print("  - UP/DOWN: Adjust base temperature")
    print("  - ESC: Stop generation")
    
    prompt = input("\nEnter a prompt: ")
    
    generate_text_interactive(prompt, model, tokenizer)


if __name__ == "__main__":
    main()