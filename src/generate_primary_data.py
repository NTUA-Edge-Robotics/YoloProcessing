from email.mime import image
from pathlib import Path
import pandas

from linear_yolo import linear_yolo
from parallel_yolo import parallel_yolo

all_images = list(Path("images").rglob("*"))
model = "yolov5x"
mode = "parallel"
batches = range(100, 101, 5)
results = pandas.DataFrame()

for num_images in batches:
    images_to_infer = all_images[:num_images]

    if (mode == "linear"):
        total_time, times = linear_yolo(images_to_infer, model)
    elif (mode == "parallel"):
        total_time, times = parallel_yolo(images_to_infer, model)
    
    frame = pandas.DataFrame(data = times, columns=["processing", "inference", "nms"])

    frame["experiment_time"] = total_time
    frame["num_images"] = num_images

    results = pandas.concat([results, frame], ignore_index=True)

results.to_csv("results/results_parallel1.csv")
