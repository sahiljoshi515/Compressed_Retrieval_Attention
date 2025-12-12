import torch
from transformers import AutoTokenizer

from hf_sparse_infer.model import LLM


def main():
    device = "cuda"
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = LLM(
        model_name=model_name,
        batch_size=1,
        max_length=4096,
        use_lsh_server=False,  # IMPORTANT: dense-only
        device=device,
        dtype=torch.bfloat16,
    )

    context = "You are a helpful assistant."
    question = "What is the capital of France?"

    ctx_ids = tokenizer(context, return_tensors="pt").input_ids.to(device)
    q_ids = tokenizer(question, return_tensors="pt").input_ids.to(device)

    tokens, stats = model.generate(
        context_ids=ctx_ids,
        question_ids=q_ids,
        max_tokens=32,
        temperature=0.7,
        verbose=True,
    )

    print("\nGenerated tokens:", tokens)
    print("Decoded text:")
    print(tokenizer.decode(tokens))


if __name__ == "__main__":
    main()
