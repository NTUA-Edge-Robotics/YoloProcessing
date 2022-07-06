import torch
from threading import Thread
from time import perf_counter_ns

from infere import infere

def parallel_yolo(images, model_name:str, use_cpu:bool):
    """Runs the inference in parallel

    Args:
        images (_type_): The images to process
        model_name (str): The name of the YOLOv5 model
        use_cpu (bool): True to use the CPU instead of the GPU

    Returns:
        _type_: The total inference time and the individual inference time for each image
    """
    if use_cpu:
        model = torch.hub.load("ultralytics/yolov5", model_name, device="cpu")
    else:
        model = torch.hub.load("ultralytics/yolov5", model_name)
    
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

    torch.cuda.empty_cache()

    return total_time, times
