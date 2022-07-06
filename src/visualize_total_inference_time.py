import pandas
from matplotlib import pyplot

from preprocess import preprocess

parallel_gpu = pandas.read_csv("results/drone_parallel_gpu_batch_60.csv")
parallel_gpu = preprocess(parallel_gpu)
x1 = parallel_gpu["batch_size"]
y1 = parallel_gpu["experiment_time"]

linear_gpu = pandas.read_csv("results/drone_linear_gpu_batch_60.csv")
linear_gpu = preprocess(linear_gpu)
x2 = linear_gpu["batch_size"]
y2 = linear_gpu["experiment_time"]

parallel_cpu = pandas.read_csv("results/drone_parallel_cpu_batch_60.csv")
parallel_cpu = preprocess(parallel_cpu)
x3 = parallel_cpu["batch_size"]
y3 = parallel_cpu["experiment_time"]

linear_cpu = pandas.read_csv("results/drone_linear_cpu_batch_60.csv")
linear_cpu = preprocess(linear_cpu)
x4 = linear_cpu["batch_size"]
y4 = linear_cpu["experiment_time"]

pyplot.figure()
#pyplot.scatter(x4, y4, edgecolors="C3", facecolors="none", label="Linear (Robot CPU)")
#pyplot.scatter(x3, y3, edgecolors="C2", facecolors="none", label="Parallel (Robot CPU)")
pyplot.scatter(x2, y2, edgecolors="C1", facecolors="none", label="Linear (Edge GPU)")
pyplot.scatter(x1, y1, edgecolors="C0", facecolors="none", label="Parallel (Edge GPU)")

pyplot.xlabel("Batch size")
pyplot.xticks(x1)
pyplot.ylabel("Total inference time (s)")
#pyplot.yticks([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55])
pyplot.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5])
pyplot.legend()
pyplot.grid(alpha=0.2)

pyplot.tick_params(axis="both", which="both", labelsize=8)
pyplot.gcf().set_size_inches(3.4, 3.4)

pyplot.savefig("results/inf_time_batch_size_gpu.svg", bbox_inches="tight")
