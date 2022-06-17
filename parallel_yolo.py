from pathlib import Path
import torch
import random
from threading import Thread
from time import sleep, perf_counter

images = Path("people").rglob("*")

model = torch.hub.load("ultralytics/yolov5", "yolov5n6")

def infere(model, image:str):
    results = model(image)
    #results.print()

start_time = perf_counter()

threads = []

for image in images:
    #sleep(random.uniform(0.01,0.05))
    t = Thread(target=infere, args=(model, image,))
    threads.append(t)
    t.start()

for t in threads:
    t.join()

end_time = perf_counter()

print(f"It took {end_time- start_time: 0.2f} second(s) to complete.")
