import torch

class HookManager():
    def __init__(self, model):
        self.model = model
        self.hooks = []

    def attach_residstream_hook(self, layer):

        hookpoint = f'transformer.h.{layer}.attn'
        
        extracted_output = []
        def residstream_hook(module, input, output):
            extracted_output.append(input[0].squeeze(0).detach())

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_hook(residstream_hook)
        )

        return extracted_output


    def attach_steering_vector(self, layer, steering_vector, scalar = 1):


        hookpoint = f'transformer.h.{layer}.attn'
        
        def steering_hook(module, input):
            activation = input[0]
            activation = activation + steering_vector * scalar
            return activation

        self.hooks.append(
            self.model.get_submodule(hookpoint).register_forward_pre_hook(steering_hook)
        )

        

    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()