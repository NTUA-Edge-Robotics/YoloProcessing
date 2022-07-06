def infere(model, image:str, times:list):
    """Runs the inference using YOLOv5 on an image using a specific model.

    Appends the inference time in the times list.

    Args:
        model (_type_): The YOLOv5 model
        image (str): The path to the image
        times (list): The inference time will be appended in this list
    """
    results = model(image)
    times.append(results.t)
