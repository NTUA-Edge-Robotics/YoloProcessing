from pathlib import Path
import torch
from threading import Thread
from time import perf_counter

from infere import infere

def parallel_yolo(images, model_name:str):
    model = torch.hub.load("ultralytics/yolov5", model_name)
    times = []
    threads = []

    start_time = perf_counter()

    for image in images:
        t = Thread(target=infere, args=(model, image, times,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = perf_counter()
    total_time = end_time - start_time

    return total_time, times
