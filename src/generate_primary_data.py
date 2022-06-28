import argparse
import os
from pathlib import Path
from time import sleep
import pandas

from linear_yolo import linear_yolo
from parallel_yolo import parallel_yolo

# Parse the arguments of the command line
parser = argparse.ArgumentParser(description="Benchmark YOLO in parallel or linearly")

parser.add_argument("images", help="The directory containing the images")
parser.add_argument("mode", choices=["parallel", "linear"], help="Parallel or linear mode")
parser.add_argument("model", help="The weights used to run the inference")
parser.add_argument("results", help="Path to save the CSV results")
parser.add_argument("-cpu", action="store_true", help="Run inference on the CPU", dest="use_cpu")

args = parser.parse_args()

all_images = list(Path(args.images).rglob("*"))
model = args.model
mode = args.mode
batches = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
results = pandas.DataFrame()

# Create results directory if not exists
os.makedirs(os.path.dirname(args.results), exist_ok=True)

for batch_size in batches:
    images_to_infer = all_images[:batch_size]

    if (mode == "linear"):
        total_time, times = linear_yolo(images_to_infer, model, args.use_cpu)
    elif (mode == "parallel"):
        total_time, times = parallel_yolo(images_to_infer, model, args.use_cpu)
    
    frame = pandas.DataFrame(data = times, columns=["processing", "inference", "nms"])

    frame["experiment_time"] = total_time
    frame["batch_size"] = batch_size

    results = pandas.concat([results, frame], ignore_index=True)

results.to_csv(args.results, index=False)
