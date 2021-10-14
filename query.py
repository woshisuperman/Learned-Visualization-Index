import numpy as np
import torch
import time

#load cells and sorted array and models
raw = np.load("./sorted_arrays/cells.npy",allow_pickle=True)
lon_sorted = np.load("./sorted_arrays/longitude.npy",allow_pickle=True)
lat_sorted = np.load("./sorted_arrays/latitude.npy",allow_pickle=True)
day_sorted = np.load("./sorted_arrays/time.npy",allow_pickle=True)
hour_sorted = np.load("./sorted_arrays/hour.npy",allow_pickle=True)
week_sorted = np.load("./sorted_arrays/week.npy",allow_pickle=True)
min_lat = np.load("./models/lat/min.npy",allow_pickle=True)
min_lon = np.load("./models/lon/min.npy",allow_pickle=True)
daysplit = np.load("./models/time.npy",allow_pickle=True)
hoursplit = np.load("./models/hour.npy",allow_pickle=True)
weeksplit = np.load("./models/week.npy",allow_pickle=True)
model = torch.load("./models/structure/model.pkl", map_location='cpu')


def HPSSearch(threshold, input_value, type):  #type--- 1 means lon 2 means lat
    if input_value >= 2**25:
        return len(raw)

    file_path = ""
    min_value = 0


    model_num = input_value//1680     #选择模型
    model_input = input_value%1680  #模型的输入

    #print("choose model ",model_num)


    if type == 1:
        file_path = "./models/lon/"
        min_value = min_lon[model_num]
    else:
        file_path = "./models/lat/"
        min_value = min_lat[model_num]

    #aa = time.clock()
    model.load_state_dict(torch.load(file_path+"net_"+str(model_num)+"_bk.pkl"))

    start_search = int(min_value + model(torch.tensor([model_input], dtype=torch.float32)).item())
    #print(start_search)
    #print("model output is ", model(torch.tensor([model_input], dtype=torch.float32)).item())
    #bb = time.clock()

    #print("model time is ",bb-aa)


    min_border = start_search - threshold
    max_border = start_search + threshold


    if min_border < 0:
        min_border = 0
    if max_border >= len(raw):
        max_border = len(raw) - 1
    if start_search < 0:
        start_search = 0
    if start_search >= len(raw):
        start_search = len(raw) - 1
    #print(start_search, min_border, max_border)

    # If it is higher than the maximum or less than the minimum, directly expand the search range
    if type == 1:
        if raw[lon_sorted[min_border]][0] > input_value or raw[lon_sorted[max_border]][0] < input_value:
            return -1
    else:
        if raw[lat_sorted[min_border]][1] > input_value or raw[lat_sorted[max_border]][1] < input_value:
            return -1

    if type == 1:
        while min_border <= max_border:
            if raw[lon_sorted[start_search]][0] == input_value:
                break
            if raw[lon_sorted[start_search]][0] > input_value:
                max_border = start_search - 1
            elif raw[lon_sorted[start_search]][0] < input_value:
                min_border = start_search + 1
            start_search = (min_border + max_border) // 2
        if start_search<0:
            return 0
        #keep looking until the last value which equals to the input value
        if min_border <= max_border:
            while start_search-1>=0 and raw[lon_sorted[start_search-1]][0]>=input_value:
                start_search -= 1
        elif min_border > max_border:
            #print(start_search, min_border, max_border)
            if raw[lon_sorted[start_search]][0]<input_value:
                while start_search+1<len(raw) and raw[lon_sorted[start_search + 1]][0] < input_value:
                    start_search += 1
                start_search+=1
            elif raw[lon_sorted[start_search]][0]>input_value:
                print(start_search)
                while start_search-1>=0 and raw[lon_sorted[start_search - 1]][0] > input_value:
                    start_search -= 1
        return start_search

    if type == 2:
        while min_border <= max_border:
            if raw[lat_sorted[start_search]][1] == input_value:
                break
            if raw[lat_sorted[start_search]][1] > input_value:
                max_border = start_search - 1
            elif raw[lat_sorted[start_search]][1] < input_value:
                min_border = start_search + 1
            start_search = (min_border + max_border) // 2
        if start_search < 0:
            return 0
        # keep looking until the last value which equals to the input value
        if min_border <= max_border:
            while start_search - 1 >= 0 and raw[lat_sorted[start_search - 1]][1] >= input_value:
                start_search -= 1
        elif min_border > max_border:
            if raw[lat_sorted[start_search]][1] < input_value:
                while start_search + 1 < len(raw) and raw[lat_sorted[start_search + 1]][1] < input_value:
                    start_search += 1
                start_search += 1
            elif raw[lat_sorted[start_search]][1] > input_value:
                while start_search - 1 >= 0 and raw[lat_sorted[start_search - 1]][1] > input_value:
                    start_search -= 1
        return start_search

def lon_search(lower, upper):
    threshold = 1024
    type = 1
    lower_pos = HPSSearch(threshold, lower, type)
    upper_pos = HPSSearch(threshold, upper, type)
    while lower_pos == -1:
        threshold = threshold + 512
        lower_pos = HPSSearch(threshold, lower, type)
    threshold = 1024
    while upper_pos == -1:
        threshold = threshold + 512
        upper_pos = HPSSearch(threshold, upper, type)
    print("The position in the sorted array of lon:     [", lower_pos, ",", upper_pos, "]")


def lat_search(lower, upper):
    #print(lower, upper)
    threshold = 1024
    type = 2
    lower_pos = HPSSearch(threshold, lower, type)
    upper_pos = HPSSearch(threshold, upper, type)
    while lower_pos == -1:
        threshold = threshold + 512
        lower_pos = HPSSearch(threshold, lower, type)
    threshold = 1024
    while upper_pos == -1:
        threshold = threshold + 512
        upper_pos = HPSSearch(threshold, upper, type)
    print("The position in the sorted array of lat:     [", lower_pos, ",", upper_pos, "]")

def day_search(lower, upper):
    print("The position in the sorted array of day:     [", daysplit[int(lower)], ",", daysplit[int(upper+1)]-1, "]")

def hour_search(lower, upper):
    print("The position in the sorted array of hour:    [", hoursplit[int(lower)], ",", hoursplit[int(upper+1)]-1, "]")

def week_search(lower, upper):
    print("The position in the sorted array of week:    [", weeksplit[int(lower)], ",", weeksplit[int(upper+1)]-1, "]")

def get_range(attribute):
    a = attribute.index('[')
    b = attribute.index(',')
    c = attribute.index(']')
    lower = float(attribute[a + 1:b])
    upper = float(attribute[b + 1:c])
    return lower,upper

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def judge(attribute):
    if '[' in attribute:
        a = attribute.index('[')
    else:
        print("Please enter the correct query format")
        return -1
    if ',' in attribute:
        b = attribute.index(',')
    else:
        print("Please enter the correct query format")
        return -1
    if ']' in attribute:
        c = attribute.index(']')
    else:
        print("Please enter the correct query format")
        return -1
    if a>b or a>c or b>c:
        print("Please enter the correct query format")
        return -1
    lower = attribute[a+1:b]
    upper = attribute[b+1:c]
    if not is_number(lower):
        print("Please enter the correct query format")
        return -1
    if not is_number(upper):
        print("Please enter the correct query format")
        return -1

print("**************************************************************************************")

'''print("Please input a range of an attribute for a search on sorted array, lon for longtitude, lat for latitude, day for timeline, hour for 24-hour, week for weekday \n"
                      "**************************************************************************************\n"
                      "Please note that the range of lon is [-180,180], the range of lat is [-90,90], the range of day is [0,941], the range of hour is [0,23], and the range of week is [0,6]\n"
                      "**************************************************************************************\n"
                      "Here are two example:  "
                      "lat[20.5, 45.6] is a command to query the start and end positions of the sorted array with latitude between 20.5 and 45.6\n"
                      "week[1,5] is a command to query the start and end positions of the sorted array with weekday between Monday and Friday\n"
                      "**************************************************************************************\n"
                      "Type 'quit' to exit\n"
                      "\n")'''
print("Input examples: lon[-120.8765, 123.7936], lat[34.5693, 25.7826], week[0,1], hour[18,23], time[0,807]\n"
      "**************************************************************************************\n"
      "Type 'quit' to exit\n"
      "\n"
      )

while True:
    time.sleep(3)
    print("\n","\n")
    print("**************************************************************************************")
    attribute = input("Please enter the query command:\n")
    attribute = attribute.replace(" ","").replace("\t","").strip()
    if attribute == "quit" or attribute == "QUIT":
        break
    elif attribute[0:3] == "lon":
        if judge(attribute) == -1:
            continue
        lower, upper = get_range(attribute)
        if lower<upper and lower>=-180 and lower<=180 and upper<=180 and upper>=-180:
            #print("lon[",lower,',',upper,']')
            lower = int((lower + 180) / 360 * 2 ** 25)
            upper = int((upper + 180) / 360 * 2 ** 25) + 1
            lon_search(lower, upper)
        else:
            print("Input out of range bounds")
            continue
    elif attribute[0:3] == "lat":
        if judge(attribute) == -1:
            continue
        lower, upper = get_range(attribute)
        if lower < upper and lower >= -180 and lower <= 180 and upper <= 180 and upper >= -180:
            #print("lat[", lower, ',', upper, ']')
            lower = int((lower + 90) / 180 * 2 ** 25)
            upper = int((upper + 90) / 180 * 2 ** 25) + 1
            lat_search(lower, upper)
        else:
            print("Input out of range bounds")
            continue
    elif attribute[0:3] == "day":
        if judge(attribute) == -1:
            continue
        lower, upper = get_range(attribute)
        if lower<upper and lower>=0 and lower<=941 and upper<=941 and upper>=0:
            #print("day[",lower,',',upper,']')
            day_search(lower, upper)
        else:
            print("Input out of range bounds")
            continue
    elif attribute[0:4] == "hour":
        if judge(attribute) == -1:
            continue
        lower, upper = get_range(attribute)
        if lower < upper and lower >= 0 and lower <= 23 and upper <= 23 and upper >= 0:
            #print("hour[", lower, ',', upper, ']')
            hour_search(lower, upper)
        else:
            print("Input out of range bounds")
            continue
    elif attribute[0:4] == "week":
        if judge(attribute) == -1:
            continue
        lower, upper = get_range(attribute)
        if lower < upper and lower >= 0 and lower <= 6 and upper <= 6 and upper >= 0:
            #print("week[", lower, ',', upper, ']')
            week_search(lower, upper)
        else:
            print("Input out of range bounds")
            continue
    else:
        print("Please enter the right attribute")