### These instructions are not very user friendly sorry (more a reminder for myself) - feel free to reach out if you want to learn more about using these datasets or scripts


## To generate 30-minute temperature forecasts:
	# First go to Temperature forecasts/
	# The csv has been pre-prepared with summary data from the 'Spatial climate data'
	# you can run this yourself by running get_temperature.py from Temperature forecasts/Climate spatial data/
	# Run gen_weather.py (changing the state and name of database each time)
	# You should now have all the temperature forecasts for each state to 2050
	# copy the saved databases <state.db> to the Modelling/datasets/ folder

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
	# select w.obs_date, w.hour, w.minute, w.curr_temp, s.demand,s.spot_price 
	# from weather as w 
	# left join spot_data as s 
	# on w.state = s.state 
	# and w.obs_date = s.obs_date 
	# and w.hour = s.hour 
	# and w.minute = s.minute 
	# where w.city="sydney"
	# and w.obs_date like '2018%';

	# Copy the resulting csv files into the Modelling/datasets/ folder

## Go to Modelling/
	# Ensure the <state.db> datasets are saved in the datasets/ folder
	# Ensure the <state_history.csv> files are saved in the datasets/ folder
	# run demand_modeller.py (first change to correct files and db)
	# run price_modeller.py (first change to correct files and db)

## To examime demand and spot forecasts use the following:
	# sqlite3 <state_report.db>
	# .headers on
	# .mode csv

	# .output <state_4.5_results.csv>
	# select year, season, week, day, hour, minute, curr_temp, rfr_demand, knn_demand, rfr_spot, knn_spot 
	# from spot_rfr 
	# where rcp="4.5";

	# .output <state_8.5_results.csv>
	# select year, season, week, day, hour, minute, curr_temp, rfr_demand, knn_demand, rfr_spot, knn_spot 
	# from spot_rfr 
	# where rcp="8.5";

## Results of the analysis are available in Results/