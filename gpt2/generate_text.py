import torch
import tiktoken
from gpt2model import GPT2Model

configure = {
    'vocabulary': 50257,      # Vocabulary size
    'd_model': 768,           # Embedding dimension
    'context_length': 1024,   # Max context length 
    'n_heads': 12,             # Number of attention heads
    'n_layers': 12,           # Number of transformer layers 
    'drop_rate': 0.1,         # Dropout rate
    'qkv_bias': False         # QKV bias
}

model = GPT2Model(configure)

def generate_text(model, input_tokens, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        input_tokens_cond = input_tokens[:, -context_size:]  # get the last context_size tokens 
        with torch.no_grad():
            logits = model(input_tokens_cond)
        
        logits = logits[:, -1, :]  # the last time step 
        print(f"logits dimension: {logits.shape}")
        probs = torch.softmax(logits, dim=-1)
        print(f"probs dimension: {probs.shape}")
        next_token = torch.argmax(probs, dim=-1, keepdim=True)
        input_tokens = torch.cat([input_tokens, next_token], dim=1)

    return input_tokens

context = 'Hello, I am'
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(context)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) 
print(f"Encoded: {encoded_tensor}")
print(f"Encoded shape: {encoded_tensor.shape}")

model.eval()
out = generate_text(model, encoded_tensor, max_new_tokens=1, context_size=configure['context_length'])
print(f"Output: {out}")

decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)
