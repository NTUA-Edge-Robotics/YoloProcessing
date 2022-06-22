def infere(model, image:str, times:list):
    results = model(image)
    times.append(results.t)
