### These instructions are not user friendly - they act as a reminder for myself



## To generate 30-minute temperature forecasts:
	# First go to Temperature forecasts/
	# The csv has been pre-prepared with summary data from the 'Spatial climate data'
	# you can run this yourself by running get_temperature.py from Temperature forecasts/Climate spatial data/
	
	# Run gen_weather.py (changing the state and name of database each time)
	# You should now have all the temperature forecasts for each state to 2050

## To examime temperature forecasts use the following:
	# sqlite3 <state.db>
	# .headers on
	# .mode csv
	# .output <state_4.5_temps.csv>
	# select * from FORECAST where RCP = "4.5"
	# .output <state_8.5_temps.csv>
	# select * from FORECAST where RCP = "8.5"

## To generate historical temperature/demand/spot datasets for inputs to modelling scripts:
	# Navigate to Market and weather historic data/
	# Run grab_spot.py to get the latest spot/demand prices
	# Run clean_spot.py
	# Run clean_weather.py (assuming the weather csv's are saved in Weather/ from rp5.ru)
	# sqlite3 dataset.db
	# .headers on
	# .mode csv
	# .output <state_history.csv>
	# 
	select w.obs_date, w.hour, w.minute, w.curr_temp, s.demand,s.spot_price 
	from weather as w 
	left join spot_data as s 
	on w.state = s.state 
	and w.obs_date = s.obs_date 
	and w.hour = s.hour 
	and w.minute = s.minute 
	where w.city="sydney"
	and w.obs_date like '2018%';
	# Copy the resulting csv files into the Modelling/historical/ folder

## Go to Modelling/
	# copy in the saved <state.db> to the datasets/ folder
	# run demand_modeller.py (first change to correct files and db)
	# run price_modeller.py (first change to correct files and db)

## To examime demand and spot forecasts use the following:
	# sqlite3 <state.db>
	# .headers on
	# .mode csv
	# .output <state_4.5_results.csv>
	# select * from <TABLE> where RCP = "4.5"
	# .output <state_8.5_results.csv>
	# select * from <TABLE> where RCP = "8.5"