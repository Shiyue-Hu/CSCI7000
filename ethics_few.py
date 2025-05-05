import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch


few_shot_df = pd.read_csv("few_shot.csv")

few_shot_examples = ""
for idx, row in few_shot_df.iterrows():
    question = row['Question']
    answer_reasoning = row['Answer and Reasoning']
    
    if pd.notna(answer_reasoning):
        few_shot_examples += f"Question: {question}\nAnswer: {answer_reasoning}\n\n"
    else:
        few_shot_examples += f"Question: {question}\nAnswer: [No answer provided]\n\n"


df = pd.read_csv("/home/hushiy/scripts/Medical.csv")


model_name = "BioMistral/BioMistral-7B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
model.to("cuda" if torch.cuda.is_available() else "cpu") 

llm = pipeline("text-generation", model=model, tokenizer=tokenizer, temperature=0.0, do_sample=False)

outputs = []
for idx, row in df.iterrows():
    prompt = f"""You are a medical professional. Answer with JUST the letter of the most ethical choice:

Here are some examples for you to learn from:
{few_shot_examples}
Now, consider the following case:

{row['Question']}
Please answer with your choice below
Answer:"""

    
    response = llm(prompt)[0]['generated_text']
    answer = response[len(prompt):].strip()  
    
    outputs.append({"question": row['Question'], "llm_output": answer})

if outputs:
    pd.DataFrame(outputs).to_csv("event_few.csv", index=False)
    print("\nSaved outputs to mistral.csv")
else:
    print("\nNo outputs generated!")