import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

df = pd.read_csv("/home/hushiy/scripts/Medical.csv")


with open("/home/hushiy/scripts/Ethics_principle.txt", "r") as file:
    retrieved_doc = file.read().strip()


model_name = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu")


llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    temperature=0.0,     
    do_sample=False,    
    return_full_text=False
)


outputs = []
for idx, row in df.iterrows():
    question = row['Question']

    prompt = f"""You are a medical professional answering a clinical ethics multiple-choice question.

Use the following medical ethics context to inform your decision:
{retrieved_doc}

Now, read the scenario below and choose the most ethically appropriate response:
{question}

Answer with JUST the letter of the most ethical choice:
Answer:"""

    response = llm(prompt)[0]['generated_text'].strip()
    
    outputs.append({
        "question": question,
        "llm_output": response
    })


if outputs:
    pd.DataFrame(outputs).to_csv("mistral_rag.csv", index=False)
    print("\n Saved RAG-enhanced outputs to mistral_rag.csv")
else:
    print("\n⚠️ No outputs were generated.")
