import torch
from time import perf_counter_ns

from infere import infere

def linear_yolo(images, model_name:str, use_cpu:bool):
    """Runs the inference linearly

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

    start_time = perf_counter_ns()

    for image in images:
        infere(model, image, times)

    end_time = perf_counter_ns()
    total_time = (end_time - start_time) / 1000000

    torch.cuda.empty_cache()

    return total_time, times
