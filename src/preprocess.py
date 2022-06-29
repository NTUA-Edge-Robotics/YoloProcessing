import pandas

def preprocess(frame:pandas.DataFrame):
    frame["average_time"] = (frame["processing"] + frame["inference"] + frame["nms"]) / 1000
    frame["experiment_time"] = frame["experiment_time"] / 1000
    frame = frame.groupby("batch_size", as_index=False).mean()

    return frame
