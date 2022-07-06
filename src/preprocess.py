import pandas

def preprocess(frame:pandas.DataFrame):
    """Calculates the average time in seconds and convert the experiment time to seconds.

    The average time is calculated by adding the processing time, the inference time and the NMS time.

    Args:
        frame (pandas.DataFrame): The data frame to process

    Returns:
        pandas.DataFrame: The modifed data frame
    """
    frame["average_time"] = (frame["processing"] + frame["inference"] + frame["nms"]) / 1000
    frame["experiment_time"] = frame["experiment_time"] / 1000
    frame = frame.groupby("batch_size", as_index=False).mean()

    return frame
