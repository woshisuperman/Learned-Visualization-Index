import numpy as np

def lon_generate():
    data = np.load("./trainingset/lon/longitude.npy").tolist()
    #print(len(data))
    trainset_V = []
    trainset_X = []
    temp_X = []
    for i in range(len(data)):
        temp_X.append([i])
    start = 0
    for i in range(20000):
        if start+1680>len(data):
            trainset_V.append(data[start:len(data)])
            trainset_X.append(temp_X[start:len(data)])
            break
        trainset_V.append(data[start:1680+start])
        trainset_X.append(temp_X[start:1680 + start])
        start+=1680

    for i in range(len(trainset_V)):
        trainset_V[i] = np.asarray(trainset_V[i]).reshape(len(trainset_V[i]),1).tolist()
    #print(trainset_V[19972])
    #print(trainset_X[19972])
    #print(len(trainset_X))
    np.save("./trainingset/lon/trainingset_X.npy",np.asarray(trainset_X))
    np.save("./trainingset/lon/trainingset_V.npy",np.asarray(trainset_V))


def lat_generate():
    data = np.load("./trainingset/lat/latitude.npy").tolist()
    #print(len(data))
    trainset_V = []
    trainset_X = []
    temp_X = []
    for i in range(len(data)):
        temp_X.append([i])
    start = 0
    for i in range(20000):
        if start + 1680 > len(data):
            trainset_V.append(data[start:len(data)])
            trainset_X.append(temp_X[start:len(data)])
            break
        trainset_V.append(data[start:1680 + start])
        trainset_X.append(temp_X[start:1680 + start])
        start += 1680

    for i in range(len(trainset_V)):
        trainset_V[i] = np.asarray(trainset_V[i]).reshape(len(trainset_V[i]), 1).tolist()
    #print(trainset_V[19972])
    #print(trainset_X[19972])
    #print(len(trainset_X))
    np.save("./trainingset/lat/trainingset_X.npy", np.asarray(trainset_X))
    np.save("./trainingset/lat/trainingset_V.npy", np.asarray(trainset_V))

def lon_stand():
    data = np.load("./trainingset/lon/trainingset_V.npy",allow_pickle=True).tolist()
    #print(len(data))
    minn = []
    for i in range(len(data)):
        minn.append(min(data[i])[0])
    np.save("./models/lon/min.npy",minn)


def lat_stand():
    data = np.load("./trainingset/lat/trainingset_V.npy",allow_pickle=True).tolist()
    #print(len(data))
    minn = []
    for i in range(len(data)):
        minn.append(min(data[i])[0])
    np.save("./models/lat/min.npy",minn)



lon_generate()
lat_generate()
lon_stand()
lat_stand()
print("Done...")