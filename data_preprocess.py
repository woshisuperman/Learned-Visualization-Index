#!/usr/bin/python
# -*- coding: UTF-8
import numpy as np
import pandas as pd
import time as tm
import datetime

n = 2**25      #分辨率倍数
latitude_resolution = int(n)   #像素精度
longitude_resolution = int(n)
dateline = 942  #从2008-3-21到2010-10-18一共942天
lat_bias = int(90 * n)
lon_bias = int(180 * n)
file_path = './raw_data/Brightkite_totalCheckins.txt'
count_dictionary = {}
count_data = []
filename = 'bkdata_trainSet'

#read the os time
def now():
    return str(tm.strftime('%Y-%m-%d %H:%M:%S'))

#calculate time difference
def dateDiffInHours(t1, t2):
    td = t2 - t1
    return td.days * 24 + td.seconds/3600

#read bk data
def read_data(filepath):
    global max_n
    data = pd.read_csv(filepath, names=['useless_variable1','time','latitute','longitude','useless_variable2'],sep='\t')

    time = data['time']
    time = np.array(time)
    lat = data['latitute']
    lat = np.array(lat)
    lon = data['longitude']
    lon = np.array(lon)
    max_n = len(data)

    return time,lat,lon

def process1(time,lat,lon):
    global count_data, count_dictionary, longitude_resolution, latitude_resolution, n
    for i in range(len(time)):
        cur_time = str(time[i])
        if(cur_time == 'nan'):
            continue
        cur_year,cur_month,cur_day = int(cur_time[0:4]),int(cur_time[5:7]),int(cur_time[8:10])
        cur_h = int(cur_time[11:13])
        #print('this is the data in line {}, hour {}'.format(i, cur_h))
        timeline = (datetime.datetime(cur_year,cur_month,cur_day) - datetime.datetime(2008, 3, 21)).days     #从2008.3.21开始
        weekday = int(datetime.datetime(cur_year, cur_month, cur_day).strftime("%w"))  # 星期天为一个星期的开始
        cur_lat = float(lat[i])
        cur_lon = float(lon[i])
        #print("this is the {}th day ,and lon is {} ,lat is {}".format(anyday,cur_lon,cur_lat))
        if (cur_lat == 90 or cur_lat == -90):
            #print(cur_lon, cur_lat)
            continue
        if (cur_lon == 180 or cur_lon == -180):
            #print(cur_lon, cur_lat)
            continue
        if (cur_lon == 0 and cur_lat == 0):
            continue
        cur_lat = int(float(lat[i] + 90) / 180 * n)
        cur_lon = int(float(lon[i] + 180) / 360 * n)
        if (cur_lat == latitude_resolution):
            cur_lat = 0
        if (cur_lon == longitude_resolution):
            cur_lon = 0
        if(cur_lat > latitude_resolution or cur_lat < 0):
            #print("!!")
            cur_lat = int(float(lon[i] + 90) / 180 * n)
            cur_lon = int(float(lat[i] + 180) / 360 * n)
            #print(cur_lon, cur_lat)
        if str(cur_lon)+"+"+str(cur_lat)+"+"+str(timeline)+"+"+str(cur_h)+"+"+str(weekday) not in count_dictionary:
            count_dictionary[str(cur_lon)+"+"+str(cur_lat)+"+"+str(timeline)+"+"+str(cur_h)+"+"+str(weekday)] = 1
        else:
            count_dictionary[str(cur_lon) + "+" + str(cur_lat) + "+" + str(timeline) + "+" + str(cur_h) + "+" + str(weekday)] += 1
        if i%100000==0:
            print(f"{i} pieces of data")
        #print("the count_data's value is {}".format(count_data[cur_lon][cur_lat][anyday]))

def process2():
    global count_dictionary, count_data
    count = 0
    for key in count_dictionary.keys():
        count+=1
        piece = []
        for i in key.split('+'):
            piece.append(int(i))
        piece.append(count_dictionary[key])
        count_data.append(piece)
        if count%100000==0:
            print(f"{count} pieces of data into list")

def main():
    global max_n, count_dictionary, count_data
    time, lat, lon = read_data(file_path)
    #print(max_n)
    process1(time,lat,lon)
    process2()
    np.save("./sorted_arrays/cells.npy", np.asarray(count_data))
    #f = h5py.File("./processed_data/trainSet_big_3d_{}.h5".format(hour), "w")
    #f.create_dataset(filename,data = train_data)
    #print(f"length of dic: {len(count_dictionary)}")
    #print(f"length of list: {len(count_data)}")
    #print(f"length of one piece: {len(count_data[0])}")
    print(f"{now()} already saved file")
    print("Done...")

if __name__ == '__main__':
    main()
