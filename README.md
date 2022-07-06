# YoloParallel

The project aims to benchmark the performance of [YOLOv5](https://github.com/ultralytics/yolov5) using parallel and linear processing.

## Installation

1. Install the dependencies with `pip install -r requirements.txt`

## Primary Data Generation

The API of the script can be found using `python src/generate_primary_data.py -h`. The script follows these steps :

1. Run the inference with YOLOv5 with the specified batch sizes (number of images)
1. Get the total inference time and the inference time for each image
1. Save the results in a CSV file

## Inference Time Visualization

The following script will produce a graphic of the total inference time according to the batch size. The API of the script can be found using `python src/visualize_total_inference_time.py -h`.

## What could be improved

- Log and handle errors
- Automated tests
