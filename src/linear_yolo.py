import torch
from time import perf_counter

from infere import infere

def linear_yolo(images, model_name:str):
    model = torch.hub.load("ultralytics/yolov5", model_name)
    times = []

    start_time = perf_counter()

    for image in images:
        infere(model, image, times)

    end_time = perf_counter()
    total_time = end_time - start_time

    return total_time, times
