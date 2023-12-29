import cv2
import llama
import torch
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"

llama_dir = "/home/alpaco/nsm/LLaMA-Adapter/llama_adapter_v2_multimodal7b/llama_model_weights/"

# choose from BIAS-7B, LORA-BIAS-7B, LORA-BIAS-7B-v21
model, preprocess = llama.load("LORA-BIAS-7B-v21", llama_dir, llama_type="7B", device=device)
model.eval()

prompt = llama.format_prompt("Tell me four words for the color, occasion, style and type of this outfit.")
imgs_dir = "/home/alpaco/nsm/LLaMA-Adapter/llama_adapter_v2_multimodal7b/docs/cloth/00006_00.jpg"
img = Image.fromarray(cv2.imread(imgs_dir))
img = preprocess(img).unsqueeze(0).to(device)

result = model.generate(img, [prompt])[0]


print('{}결과:'.format(imgs_dir.split('/')[-1]),result)