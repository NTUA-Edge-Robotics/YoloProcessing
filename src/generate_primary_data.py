from pathlib import Path
from time import sleep
import pandas

from linear_yolo import linear_yolo
from parallel_yolo import parallel_yolo

all_images = list(Path("images").rglob("*"))
model = "yolov5x"
mode = "parallel" # linear
device = "gpu" # cpu
batches = [1] + list(range(5, 101, 5))
results = pandas.DataFrame()

for num_images in batches:
    images_to_infer = all_images[:num_images]

    if (mode == "linear"):
        total_time, times = linear_yolo(images_to_infer, model, device)
    elif (mode == "parallel"):
        total_time, times = parallel_yolo(images_to_infer, model, device)
    
    frame = pandas.DataFrame(data = times, columns=["processing", "inference", "nms"])

    frame["experiment_time"] = total_time
    frame["num_images"] = num_images

    results = pandas.concat([results, frame], ignore_index=True)

    sleep(3)

results.to_csv("results/results_parallel_9.csv", index=False)
