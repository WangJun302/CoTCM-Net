import pandas as pd
import torch
import numpy as np
import json
import faiss
from transformers import T5Tokenizer, T5Config
from test3 import HFMoET5Model 

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
tokenizer = T5Tokenizer.from_pretrained('t5-base')

config = T5Config.from_pretrained("t5-small")
config.num_experts = 6
config.use_cache = True

model = HFMoET5Model(config)
model.load_state_dict(
        torch.load(r'D:\Users\admin\Desktop\ChatMol-main1\ChatMol-main\save_model\moe_t5_small_6cengMOE.pt',
                   weights_only=False))
model.to(device)
model.eval()


def encode_texts(texts, tokenizer, model, device="cuda", batch_size=16):
    vectors = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            tokens = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            outputs = model.encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
            vecs = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # 取第一个 token 的表示
            vectors.append(vecs)
    return np.vstack(vectors)


df = pd.read_excel(r"D:\Users\admin\Desktop\ChatMol-main1\ChatMol-main\chufang.xlsx")

unique_functions = df["Function"].dropna().drop_duplicates().astype(str).tolist()

print(f"提取唯一 Function 功效数量: {len(unique_functions)}")


vectors = encode_texts(unique_functions, tokenizer, model, device=device)


index = faiss.IndexFlatL2(vectors.shape[1])
index.add(vectors)

faiss.write_index(index, "function_vectors.faiss")


with open("function_texts.json", "w", encoding="utf-8") as f:
    json.dump(unique_functions, f, indent=2, ensure_ascii=False)




index = faiss.read_index("function_vectors.faiss")

with open("function_texts.json", "r", encoding="utf-8") as f:
    function_texts = json.load(f)

def retrieve_similar_functions(input_function, tokenizer, model, index, function_texts, top_k=3):
    model.eval()
    with torch.no_grad():
        tokens = tokenizer([input_function], return_tensors="pt", padding=True, truncation=True).to(device)
        outputs = model.encoder(input_ids=tokens.input_ids, attention_mask=tokens.attention_mask)
        query_vector = outputs.last_hidden_state[:, 0, :].detach().cpu().numpy()  # shape: (1, hidden_dim)


    distances, indices = index.search(query_vector, top_k)
    top_functions = [function_texts[i] for i in indices[0]]
    return top_functions

def generate_multiple_smiles(func, tokenizer, model, device, num_return_sequences=5, max_length=64):
    input_tokens = tokenizer(func, return_tensors="pt", truncation=True, padding=True).to(device)

    output_ids = model.generate(
                input_ids=input_tokens.input_ids,
                attention_mask=input_tokens.attention_mask,
                max_new_tokens=256,
                num_beams=1, 
                do_sample=True, 
                top_k=50,  
                top_p=0.95, 
                temperature=0.8,

                no_repeat_ngram_size=2,
                pad_token_id=tokenizer.pad_token_id,

                decoder_start_token_id=model.config.decoder_start_token_id,
                num_return_sequences=num_return_sequences,
            )

    decoded = [tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
    return decoded

input_func = "It has the effects of warming meridians and dredging collaterals, releasing the exterior, dispersing wind-cold, and unblocking yang."

similar_funcs = retrieve_similar_functions(input_func, tokenizer, model, index, function_texts, top_k=3)
print("Top-3 相似功效：", similar_funcs)


all_generated = []
for func in similar_funcs:
    gen = generate_multiple_smiles(func, tokenizer, model, device, num_return_sequences=5)
    all_generated.extend(gen)

print(f"\n共生成 {len(all_generated)} 个 SMILES 候选：")
for i, s in enumerate(all_generated):
    print(f"{i+1}. {s}")

import numpy as np
import faiss
import torch

def smiles_to_vector(smiles_list, tokenizer, device='cpu'):
    
    inputs = tokenizer(smiles_list, return_tensors='pt', padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.encoder(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask)
        embeddings = outputs.last_hidden_state[:, 0, :] 

def get_topk_similar_smiles(smiles_list, tokenizer, model, device, topk=4):
    
    vectors = smiles_to_vector(smiles_list, tokenizer, device)
    centroid = np.mean(vectors, axis=0, keepdims=True)  
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    D, I = index.search(centroid, topk)
    topk_smiles = [smiles_list[i] for i in I[0]]
    return topk_smiles

top5 = get_topk_similar_smiles(
    smiles_list=all_generated,
    tokenizer=tokenizer,
    model=model,
    device=device,
    topk=4,
)

print("🎯 Top-4 最相似结构 SMILES:")
for i, s in enumerate(top5, 1):
    print(f"{s}")
