import os, csv, sqlite3, sys, random
import pandas as pd
import numpy as np
from numpy import random
import scipy.stats as stats

states = ["Tas"]									# "NSW","QLD","SA","Vic","Tas"	
seasons = ["Summer","Autumn","Winter","Spring"]
weeks = list(map(int,range(1,54)))
years = list(map(int,range(2020,2051)))
days = list(map(int,range(1,8)))
hours = list(map(int,range(0,24)))
# change 2 to number of (iterations - 1)
iterations = list(map(int,range(1,2)))
RCPs = ["4.5", "8.5"]					


def create_tables(cxn):
	"""
	Function to create tables in sqlite3
	:param cxn: the connection to the sqlite3 database
	:return:
	"""
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS FORECAST")
	cursor.execute("""CREATE TABLE IF NOT EXISTS FORECAST(
		iteration int,
		state varchar(3),
		RCP varchar(3),
		year int,
		season varchar(6),
		week int,
		day int,
		hour int,
		minute int, 
		temp_forecast float
		)""")
	cursor.close()



def dbexecute(cxn, query, payload):
	"""
	Function to execute an sqlite3 table insertion
	:param cxn: connection to the sqlite3 database
	:param query: the query to be run
	:param payload: the payload for any query parameters
	:return:
	"""
	cursor = cxn.cursor()
	if not payload:
		cursor.execute(query)
	else:
		cursor.execute(query, payload)



def generate_weather(seasonal_data, weekly_data, climate_data, num_calcs_remaining):

	for iteration in iterations:

		print ("\nITERATION # " + str(iteration) + "\n")
	
		for state in states:

			for RCP_number in RCPs:

				for year in years:		

					print ("Generating forecast for: " + str(state) + ", " + str(RCP_number) + ", "+ str(year) + "; " + str(num_calcs_remaining) + " remaining.")

					for week in weeks:

						# find season given week
						if week < 9:
							season = "Summer"
						elif 9 <= week < 23:
							season = "Autumn"
						elif 23 <= week < 36:
							season = "Winter"
						elif 36 <= week < 49:
							season = "Spring"
						else:
							season = "Summer"

						# get seasonal variance given state, season
						seasonal_var = float(seasonal_data.loc[(seasonal_data['State'] == state) & (seasonal_data['Season'] == season)]['Std dev'].values[0])
						
						# get seasonal temp increase given RCP
						if year < 2031:
							rcp_year = "2030"
						else:
							rcp_year = "2050" 
						temp_incr = float(climate_data.loc[(climate_data['State'] == state) & (climate_data['Year'] == rcp_year) & (climate_data['RCP'] == RCP_number)]['Value'].values[0])
						if year == 2030:
							temp_incr_2030 = temp_incr
						if year < 2031:
							scaled_temp_incr = temp_incr - (float(rcp_year) - year)/10*temp_incr
						else:
							temp_incr_relative = temp_incr - temp_incr_2030
							scaled_temp_incr = temp_incr - (float(rcp_year) - year)/20*temp_incr_relative

						# get weekly average given state, week
						week_average = float(weekly_data.loc[(weekly_data['State'] == state) & (weekly_data['Week'] == str(week))]['Monthly Average'].values[0])

						todays_variance = None
						yesterdays_variance = None
						for day in days:

							# select days variance using normal dist of variance subject to upper and lower limits
							lower = 0 - 1.5*seasonal_var
							upper = 0 + 50*seasonal_var

							# day_variance = random.normal(0, seasonal_var)
							if todays_variance is None:
								todays_variance = float(stats.truncnorm(lower/seasonal_var, upper/seasonal_var, 0, seasonal_var).rvs(1))
							tomorrows_variance = float(stats.truncnorm(lower/seasonal_var, upper/seasonal_var, 0, seasonal_var).rvs(1))

							for hour in hours:

								# weekly average +/- daily variance + seasonal temp increase
								hourly_scale = float(hourly_data.loc[(hourly_data['State'] == state) & (hourly_data['Season'] == season) & (hourly_data['Hour'] == str(hour))]['Adj_average'].values[0])
								# scale the average and variance based on hourly characteristics
								hourly_average = hourly_scale * week_average

								# smooth variance between days
								if yesterdays_variance is not None:
									if 21 <= hour <= 22:
										smoothed_variance = .8*hourly_scale*todays_variance + .2*hourly_scale*tomorrows_variance
									elif hour == 23:
										smoothed_variance = .7*hourly_scale*todays_variance + .3*hourly_scale*tomorrows_variance
									elif hour == 0:
										smoothed_variance = .5*hourly_scale*todays_variance + .5*hourly_scale*yesterdays_variance
									elif hour == 1:
										moothed_variance = .7*hourly_scale*todays_variance + .3*hourly_scale*yesterdays_variance
									elif 2 <= hour <= 3:
										smoothed_variance = .8*hourly_scale*todays_variance + .2*hourly_scale*yesterdays_variance
									else:
										smoothed_variance = todays_variance
								else:
									smoothed_variance = todays_variance

								temp_forecast = hourly_average + smoothed_variance + scaled_temp_incr
								
								# store into database (30-minute intervals)
								minute = 0
								query = """INSERT OR IGNORE INTO FORECAST(iteration, state, RCP, year, season, week, day, hour, minute, temp_forecast) VALUES (?,?,?,?,?,?,?,?,?,?)"""
								payload = (iteration, state, RCP_number, year, season, week, day, hour, minute, temp_forecast)
								dbexecute(cxn, query, payload)
								minute = 30
								query = """INSERT OR IGNORE INTO FORECAST(iteration, state, RCP, year, season, week, day, hour, minute, temp_forecast) VALUES (?,?,?,?,?,?,?,?,?,?)"""
								payload = (iteration, state, RCP_number, year, season, week, day, hour, minute, temp_forecast)
								dbexecute(cxn, query, payload)

							todays_variance = tomorrows_variance
							yesterdays_variance = todays_variance

					num_calcs_remaining -= 1


if __name__ == '__main__':

	# set up the locations for data retrieval and storage, connect to db and create tables
	DATABASE = "test.db"					##### CHANGE THIS LINE
	cxn = sqlite3.connect(DATABASE)
	create_tables(cxn)

	file_name = "weather_for_script.csv"	
	df = pd.read_csv(file_name)

	# modify csv tables into pandas dataframes
	seasonal_data, weekly_data, climate_data, hourly_data = df[0:20], df[22:287], df[289:309], df[311:]
	seasonal_data.drop(["Unnamed: 5","Unnamed: 6"], axis=1, inplace=True)

	weekly_data.columns = df[21:22].values[0]
	weekly_data.reset_index(inplace=True)
	weekly_data.drop([np.nan,"index"], axis=1, inplace=True)

	climate_data.columns = df[288:289].values[0]
	climate_data.reset_index(inplace=True)
	climate_data.drop([np.nan,"index"], axis=1, inplace=True)

	hourly_data.columns = df[310:311].values[0]
	hourly_data.reset_index(inplace=True)

	num_calcs_remaining = len(states) * len(RCPs) * len(years) * len(iterations)

	generate_weather(seasonal_data, weekly_data, climate_data, num_calcs_remaining)

	cxn.commit()
	cxn.close()