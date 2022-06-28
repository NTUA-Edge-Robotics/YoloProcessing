import torch
from time import perf_counter_ns

from infere import infere

def linear_yolo(images, model_name:str, the_device:str):
    model = torch.hub.load("ultralytics/yolov5", model_name, device=the_device)
    times = []

    start_time = perf_counter_ns()

    for image in images:
        infere(model, image, times)

    end_time = perf_counter_ns()
    total_time = (end_time - start_time) / 1000000

    return total_time, times
