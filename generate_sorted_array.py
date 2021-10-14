import numpy as np


raw_data = np.load("./sorted_arrays/cells.npy").tolist()
for i in range(len(raw_data)):
    raw_data[i].append(i)
#print(raw_data[0:100])
raw_data = np.asarray(raw_data)
#print(len(raw_data))


def cmp(elem):
    return elem[0]

def sort_lon():
    lon_index = raw_data[:,[0, 6]]
    lon_index = lon_index.tolist()
    lon_index.sort(key=cmp)
    lon_index = np.asarray(lon_index)
    lon_index = lon_index[:, 1]
    np.save("./sorted_arrays/longitude.npy", lon_index)

def sort_lat():
    lat_index = raw_data[:, [1, 6]]
    #print(lat_index)
    lat_index = lat_index.tolist()
    lat_index.sort(key=cmp)
    #print(lat_index)
    lat_index = np.asarray(lat_index)
    lat_index = lat_index[:, 1]
    np.save("./sorted_arrays/latitude.npy", lat_index)

def sort_week():
    week_index = raw_data[:, [4, 6]]
    week_index = week_index.tolist()
    week_index.sort(key=cmp)
    week_index = np.asarray(week_index)
    week_index = week_index[:, 1]
    np.save("./sorted_arrays/week.npy", week_index)

def sort_day():
    day_index = raw_data[:, [2, 6]]
    day_index = day_index.tolist()
    day_index.sort(key=cmp)
    day_index = np.asarray(day_index)
    day_index = day_index[:, 1]
    np.save("./sorted_arrays/time.npy", day_index)

def sort_hour():
    hour_index = raw_data[:, [3, 6]]
    hour_index = hour_index.tolist()
    hour_index.sort(key=cmp)
    hour_index = np.asarray(hour_index)
    hour_index = hour_index[:, 1]
    np.save("./sorted_arrays/hour.npy", hour_index)



def weeksplit():
    weeksort = np.load("./sorted_arrays/week.npy")
    week = [0]*7
    count = 1
    for i in range(len(weeksort)-1):
        if raw_data[weeksort[i]][4]!=raw_data[weeksort[i+1]][4]:
            week[count] = i+1
            count+=1
    week.append(len(weeksort))
    np.save("./models/week.npy",np.asarray(week))

def hoursplit():
    hoursort = np.load("./sorted_arrays/hour.npy")
    hour = [0]*24
    count = 1
    for i in range(len(hoursort)-1):
        if raw_data[hoursort[i]][3]!=raw_data[hoursort[i+1]][3]:
            hour[count] = i+1
            count+=1
    hour.append(len(hoursort))
    np.save("./models/hour.npy",np.asarray(hour))

def daysplit():
    daysort = np.load("./sorted_arrays/time.npy")
    day = [0] * 942
    count = 1
    for i in range(len(daysort) - 1):
        if raw_data[daysort[i]][2] != raw_data[daysort[i + 1]][2]:
            day[count] = i + 1
            count += 1
    day.append(len(daysort))
    np.save("./models/time.npy", np.asarray(day))
    #print(np.asarray(day))

def lonsplit():
    lonsort = np.load("./sorted_arrays/longitude.npy")
    lon = [0] * 2**25
    count = 0
    while count<=raw_data[lonsort[0]][0]:
        lon[count] = 0
        count+=1
    for i in range(len(lonsort) - 1):
        if raw_data[lonsort[i]][0] != raw_data[lonsort[i + 1]][0]:
            while count<=raw_data[lonsort[i+1]][0]:
                lon[count] = i+1
                count+=1
    if count<=2**25-1:
        while count<=2**25-1:
            lon[count] = len(lonsort)
            count+=1
    np.save("./trainingset/lon/longitude.npy", np.asarray(lon))
    #print(np.asarray(lon))

def latsplit():
    latsort = np.load("./sorted_arrays/latitude.npy")
    lat = [0] * 2**25
    count = 0
    while count<=raw_data[latsort[0]][1]:
        lat[count] = 0
        count+=1
    for i in range(len(latsort) - 1):
        if raw_data[latsort[i]][1] != raw_data[latsort[i + 1]][1]:
            while count<=raw_data[latsort[i+1]][1]:
                lat[count] = i+1
                count+=1
    if count<=2**25-1:
        while count<=2**25-1:
            lat[count] = len(latsort)
            count+=1
    np.save("./trainingset/lat/latitude.npy", np.asarray(lat))
    #print(np.asarray(lat))


sort_lat()
sort_lon()
sort_day()
sort_hour()
sort_week()


latsplit()
lonsplit()
daysplit()
hoursplit()
weeksplit()


print("Done...")