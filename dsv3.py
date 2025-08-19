from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

model_id = "deepseek-ai/DeepSeek-V3"

tokenizer = AutoTokenizer.from_pretrained(model_id)
config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    config=config,
    trust_remote_code=True,
    torch_dtype="auto",
)

# output_path = f"golden_data_llama4-17b-{model_size_to_num_experts}.jsonl"

# # prompt_texts = ["I love to"]
# # prompt_texts = ['Hello, my name is']
# prompt_texts = ["What is the chemical symbol for gold?"]
# all_data_to_save = []

# for prompt_text in prompt_texts:
#   input_ids = tokenizer.encode(prompt_text, return_tensors="pt")
#   print(f"Input ids are {input_ids}")
#   with torch.no_grad():
#     generate_ids = model.generate(input_ids, do_sample=False, max_length=30)
#     out_tokens = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#     print("Output_tokens:\n", out_tokens)
