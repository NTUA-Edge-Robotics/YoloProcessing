from pathlib import Path
import torch
from time import perf_counter

images = Path("images").rglob("*")

model = torch.hub.load("ultralytics/yolov5", "yolov5x6")

def infere(model, image:str):
    results = model(image)
    #results.print()

start_time = perf_counter()

for image in images:
    infere(model, image,)

end_time = perf_counter()

print(f"It took {end_time- start_time: 0.2f} second(s) to complete.")
