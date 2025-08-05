# app/llm.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from typing import List

MODEL_ID = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

generator = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1,
    model_kwargs={"torch_dtype": "float32"}
)

def generate_answer(question: str, context_chunks: List[str]) -> str:
    context = "\n".join(context_chunks)

    # ✂ Truncate long context
    if len(context) > 1000:
        context = context[:1000]

    prompt = f"Context: {context}\n\nQuestion: {question}\nAnswer:"

    result = generator(
        prompt,
        max_new_tokens=200,
        do_sample=False,
        truncation=True
    )

    return result[0]["generated_text"]