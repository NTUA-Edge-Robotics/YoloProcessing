import argparse
import pandas
from matplotlib import pyplot

from preprocess import preprocess

# Parse the arguments of the command line
parser = argparse.ArgumentParser(description="Produces two graphics of the total inference time according to the batch size for linear and parallel processing in GPU and CPU.")

parser.add_argument("parallel", help="The results of the GPU parallel processing")
parser.add_argument("parallel_label", help="The label of the parallel data")
parser.add_argument("parallel_color", help="The color of the parallel data")
parser.add_argument("linear", help="The results of the GPU linear processing")
parser.add_argument("linear_label", help="The label of the linear data")
parser.add_argument("linear_color", help="The color of the linear data")
parser.add_argument("figure", help="Path and filename of the resulting figure")

args = parser.parse_args()

parallel_gpu = pandas.read_csv(args.parallel)
parallel_gpu = preprocess(parallel_gpu)
x1 = parallel_gpu["batch_size"]
y1 = parallel_gpu["experiment_time"]

linear_gpu = pandas.read_csv(args.linear)
linear_gpu = preprocess(linear_gpu)
x2 = linear_gpu["batch_size"]
y2 = linear_gpu["experiment_time"]

pyplot.figure(figsize=(3.4, 3.4))
pyplot.scatter(x1, y1, edgecolors=args.parallel_color, facecolors="none", label=args.parallel_label)
pyplot.scatter(x2, y2, edgecolors=args.linear_color, facecolors="none", label=args.linear_label)

pyplot.xlabel("Batch size")
pyplot.xticks(x1)
pyplot.ylabel("Total inference time (s)")
pyplot.legend()
pyplot.grid(alpha=0.2)

pyplot.tick_params(axis="both", which="both", labelsize=8)
pyplot.savefig(args.figure, bbox_inches="tight")
