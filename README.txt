We show the building and execution methods of LVI using the Brightkite dataset. The dataset can be downloaded from https://snap.stanford.edu/data/loc-brightkite_totalCheckins.txt.gz

For constructing LVI by yourselves, please download and unzip the dataset and put it in the raw_data directory. Users can skip the download process and directly execute queries using the models and sorted arrays uploaded together with the code. 


Steps for executing queries:
1. Enter command line.
2. Execute “python query.py”.
3. Input query commands.


Steps for constructing LVI:
1. Enter command line.
2. Execute “python dara_preprocess.py” to generate cells.
3. Execute “python generate_sorted_array.py” to generate sorted arrays and hashmaps (time, week, hour).
4. Execute “generate_trainingset.py” to generate training sets (longitude and latitude).
5. Execute “python training.py” to train models (longitude and latitude).


*query command format: 
	attribute[lower boundary, upper boundary]
*attribute: 
	lon, lat, week, hour, time
*attribute ranges: 
	lon[-180.0, 180.0] (float)
	lat[-90.0, 90.0] (float)
	time[0,941] (int)
	hour[0,23] (int)
	week[0, 6] (int) 


Training environment:
1. Python 3.7+
2. Libraries: torch (GPU version), torch, numpy, pandas, datetime, time, functools, sklearn
3. Windows10 (Ryzen 3700X, 16G, GTX 1650) | CentOS7 (XEON E5-2680, 196G, GTX 2080Ti)
* We have developed and tested in such an environment. Other similar environments may be also possible.


Files descriptions:
1.“models” contains all trained models (longitude and latitude) and hashmaps (time, week, hour).
2.“raw_data” contains the original data.
3.“sorted arrays” contains all generated sorted arrays (all 5 attributes)
4.“trainingset” contains the training sets (longitude and latitude).
5.“config.py” is the configurations of all models.
6.“model2.py” is the specification of all models.







