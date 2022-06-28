import pandas

frame = pandas.read_csv("results/results_parallel_9.csv")
frame["average_time"] = (frame["processing"] + frame["inference"] + frame["nms"]) / 1000
frame["experiment_time"] = frame["experiment_time"] / 1000
frame = frame.groupby("num_images", as_index=False).mean()

results = frame[["num_images", "average_time", "experiment_time"]]

print(results[["num_images", "average_time", "experiment_time"]])
