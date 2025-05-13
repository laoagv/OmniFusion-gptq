import asyncio
from websockets.asyncio.server import serve
from transformers import AutoTokenizer
from gptqmodel import GPTQModel, QuantizeConfig, get_best_device
from urllib.request import urlopen

import ast
import json
import time
import torch
from PIL import Image
import torch.nn as nn
import models
import base64
from io import BytesIO
import sys


DEVICE = sys.argv[1]


startTime = time.time()
model = GPTQModel.load("./omnifusion-gptq-8bit", device_map=DEVICE)
model.model = torch.compile(model.model, mode="default", fullgraph=False)

print(time.time()-startTime)
print("model loaded")

# DEVICE = "cpu"
PROMPT = "Это диалог с ИИ-помощником. Используй русский язык.\n"

tokenizer = model.tokenizer

# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/projection.pt", local_dir='./')
# hf_hub_download(repo_id="AIRI-Institute/OmniFusion", filename="OmniMistral-v1_1/special_embeddings.pt", local_dir='./')
projection = torch.load("projection.pt", map_location=DEVICE, weights_only=False)
special_embs = torch.load("special_embeddings.pt", map_location=DEVICE)
print(time.time()-startTime)
print("projection and spec embs loaded")
clip = models.CLIPVisionTower("./clip-vit-large")
clip.load_model()
clip = clip.to(device=DEVICE, dtype=torch.bfloat16)
print(time.time()-startTime)
print("clip model loaded")



def gen_answer(model, tokenizer, clip, projection, query, special_embs, image=None):
    bad_words_ids = tokenizer(["\n", "</s>", ":"], add_special_tokens=False).input_ids + [[13]]
    gen_params = {
            "do_sample": False,
            "max_new_tokens": 350,
            "early_stopping": False,
            "num_beams": 1,
            "num_return_sequences": 1,
            "repetition_penalty": 1.0,
            "remove_invalid_values": True,
            "eos_token_id": 2,
            "pad_token_id": 2,
            "forced_eos_token_id": 2,
            "use_cache": True,
            "no_repeat_ngram_size": 4,
            "bad_words_ids": bad_words_ids,
            # "max_length": 300,
        }

    print(time.time()-startTime)
    print("emb creates started")

    if image!=None:
        image_features = clip.image_processor(image, return_tensors='pt')

        print(time.time()-startTime)
        print("first clip run ended")

        image_embedding = clip(image_features['pixel_values']).to(device=DEVICE, dtype=torch.bfloat16)

        print(time.time()-startTime)
        print("second clip run ended")

        projected_vision_embeddings = projection(image_embedding).to(device=DEVICE, dtype=torch.bfloat16)

        print(time.time()-startTime)
        print("projection run ended")

    prompt_ids = tokenizer.encode(f"{PROMPT}", add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

    question_ids = tokenizer.encode(query, add_special_tokens=False, return_tensors="pt").to(device=DEVICE)

    prompt_embeddings = model.model.model.embed_tokens(prompt_ids).to(torch.bfloat16)

    question_embeddings = model.model.model.embed_tokens(question_ids).to(torch.bfloat16)

    print(time.time()-startTime)
    print("text embs created")
    if image !=None:
        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                projected_vision_embeddings,
                special_embs['EOI'][None, None, ...],
                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...],
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    else:
        embeddings = torch.cat(
            [
                prompt_embeddings,
                special_embs['SOI'][None, None, ...],
                special_embs['EOI'][None, None, ...],
                special_embs['USER'][None, None, ...],
                question_embeddings,
                special_embs['BOT'][None, None, ...],
            ],
            dim=1,
        ).to(dtype=torch.bfloat16, device=DEVICE)
    print(time.time()-startTime)
    print("embds cuted")

    out = model.model.generate(inputs_embeds=embeddings, **gen_params)

    print(time.time()-startTime)
    print("answer generated")

    out = out[:, 0:]
    generated_texts = tokenizer.batch_decode(out)[:3]
    return generated_texts

async def echo(websocket):
    async for message in websocket:
        request = json.loads(message)
        question = request['text']
        try:
            image = request['image']
            if image.startswith("data:image/"):
                image = image.split(",")[1]
            decoded_bytes = base64.b64decode(image)
            byte_io = BytesIO(decoded_bytes)
            img = Image.open(byte_io)
        except:
            print("img error")
            img=None

        answer = gen_answer(model,tokenizer,clip,projection,query=question,special_embs=special_embs,image=img)
        if isinstance(my_variable, list):
            answer = " ".join(answer)

        await websocket.send(answer)

async def main():
    print("server run")
    async with serve(echo, "0.0.0.0", 8765):
        await asyncio.get_running_loop().create_future()  # run forever

asyncio.run(main())