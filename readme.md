### These instructions are not user friendly - they act as a reminder for myself



## First go to Temperature forecasts/
	# The csv has been pre-prepared with summary data from the 'Spatial climate data'
	# you can run this yourself by running get_temperature.py from Temperature forecasts/Climate spatial data/
	
	# Run gen_weather.py (changing the state and name of database each time)
	# You should now have all the temperature forecasts for each state to 2050

## To examime temperature forecasts use the following:
	# sqlite3 <state.db>
	# .headers on
	# .mode csv
	# .output <state_4.5_temps.csv>
	# select * from <TABLE> where RCP = "4.5"
	# .output <state_8.5_temps.csv>
	# select * from <TABLE> where RCP = "8.5"

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