import pandas

from preprocess import preprocess

parallel = pandas.read_csv("results/drone_parallel_gpu_batch_60.csv")
parallel = preprocess(parallel)

print(parallel[["batch_size", "average_time"]])

linear = pandas.read_csv("results/drone_linear_gpu_batch_60.csv")
linear = preprocess(linear)

print(linear[["batch_size", "average_time"]])
