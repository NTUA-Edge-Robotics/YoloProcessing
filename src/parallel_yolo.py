import torch
from threading import Thread
from time import perf_counter_ns

from infere import infere

def parallel_yolo(images, model_name:str, the_device:str):
    model = torch.hub.load("ultralytics/yolov5", model_name, device=the_device)
    times = []
    threads = []

    start_time = perf_counter_ns()

    for image in images:
        t = Thread(target=infere, args=(model, image, times,))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    end_time = perf_counter_ns()
    total_time = (end_time - start_time) / 1000000

    return total_time, times
