from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

# Load the fine-tuned model and tokenizer
model_path = "./my_fine_tuned_model"
model = GPT2LMHeadModel.from_pretrained(model_path)
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

def generate_text(prompt, max_length=100):
    # Encode the input and create attention mask
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device)
    
    # Generate text
    output = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        pad_token_id=tokenizer.eos_token_id,
    )
    
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Example usage
prompt = "The main topic of the video is"  # Adjust this based on your video content
generated_text = generate_text(prompt)
print(f"Prompt: {prompt}")
print(f"Generated text: {generated_text}")

# Interactive mode
while True:
    user_prompt = input("Enter a prompt (or 'quit' to exit): ").strip()
    if user_prompt.lower() == 'quit':
        break
    if not user_prompt:
        print("Please enter a non-empty prompt.")
        continue
    generated_text = generate_text(user_prompt)
    print(f"Generated text: {generated_text}")