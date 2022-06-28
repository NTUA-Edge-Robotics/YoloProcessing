import pandas
from matplotlib import pyplot

def preprocess(frame:pandas.DataFrame):
    frame["average_time"] = (frame["processing"] + frame["inference"] + frame["nms"]) / 1000
    frame["experiment_time"] = frame["experiment_time"] / 1000
    frame = frame.groupby("batch_size", as_index=False).mean()

    return frame

parallel = pandas.read_csv("results/drone_parallel_gpu_batch_60.csv")
parallel = preprocess(parallel)
x1 = parallel["batch_size"]
y1 = parallel["experiment_time"]

linear = pandas.read_csv("results/drone_linear_gpu_batch_60.csv")
linear = preprocess(linear)
x2 = linear["batch_size"]
y2 = linear["experiment_time"]

pyplot.figure()
pyplot.scatter(x2, y2, edgecolors="C1", facecolors="none", label="Linear")
pyplot.scatter(x1, y1, edgecolors="C0", facecolors="none", label="Parallel")

pyplot.xlabel("Batch size")
pyplot.xticks(x1)
pyplot.ylabel("Total inference time (s)")
pyplot.yticks([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4])
pyplot.legend()
pyplot.grid(alpha=0.2)

pyplot.tick_params(axis="both", which="both", labelsize=8)
pyplot.gcf().set_size_inches(3.4, 3.4)

pyplot.savefig("results/inf_time_batch_size.svg", bbox_inches="tight")
#pyplot.show()
