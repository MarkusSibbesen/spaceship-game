from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, GPTNeoXForCausalLM
#from classes.datahandling import TextClassificationDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
#from utils.probe_confidence_intervals import get_activations
import torch
from collections import defaultdict
import torch

def model_setup(model_name:str) -> tuple[AutoModelForCausalLM,AutoTokenizer, str]:
    """loads a huggingface model

    Args:
        model_name (str): the huggingface name of a model. Example: AI-Sweden-Models/gpt-sw3-356m

    Returns:
        tuple[AutoModelForCausalLM,AutoTokenizer, str]: model, tokenizer, device
    """

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    # Initialize Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    model.to(device)
    print("found device:",device)
    return model, tokenizer, device

