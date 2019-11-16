import os, glob, csv, sqlite3, sys

"""
T: air temp 2m above ground
Po: atmospheric pressure mm mercury
Pa: pressure tendency (changes in atmospheric pressure over last 3 hours mm memrcury)
U: relative humidity
DD: mean wind direction for past 10 mins
Ff: mean wind speed (m/s)
ff10: maximum gust value (m/s)
ww: special weather phenomenon
C: total cloud cover
RRR: amount of precipitation (mm)
"""

"""
	SQLITE3 to csv for analysis
	.headers on
	.mode csv
	.output state_weather.csv
	select * from weather where (obs_date like "2018%" and state="NSW");
"""

def create_tables(cxn):
	"""
	Function to create tables in sqlite3
	:param cxn: the connection to the sqlite3 database
	:return:
	"""
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS WEATHER")
	cursor.execute("""CREATE TABLE IF NOT EXISTS WEATHER(
		state varchar(3),
		city varchar (15),
		obs_date varchar(12),
		hour int,
		minute int,
		curr_temp float,
		unique(state, city, obs_date, hour, minute)
		)""")
	cursor.close()



def dbselect(cxn, query, payload):
	"""
	Function to select data from an sqlite3 table
	:param cxn: connection to the sqlite3 database
	:param query: the query to be run
	:param payload: the payload for any query parameters
	:return results: the results of the search
	"""
	cursor = cxn.cursor()
	if not payload:
		rows = cursor.execute(query)
	else:
		rows = cursor.execute(query,payload)
	results = []
	for row in rows:
		results.append(row)
	cursor.close()
	return results



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


def csv_to_sql(files):
	"""
	"""

	state_summary = {}

	for file in glob.glob(files):
		state, city = file.split("/")[2][:-4].split("_")

		# state_summary[state] = {}

		# read each csv file by transcribing rows to columns for simpler extraction
		with open(file,'r', encoding='utf-8') as enc_in:
			reader = csv.reader(enc_in)

			# skip header rows
			for i in range(0,7):
				next(reader, None)

			# store data rows
			for row in reader:

				data = row[0].replace('"','').split(";")
				year = data[0][6:10]
				month = data[0][3:5]
				day = data[0][0:2]
				obs_date = str(year) + "/" + str(month) + "/" + str(day)
				hour = data[0][11:13]
				minute = data[0][14:16]
				if (state == "sa"):
					minute = 0
				else:
					if minute != "00":
						continue
				curr_temp = data[1]

				# state_summary[state][obs_date] = {}

				for i in range(2):
					if i == 0:
						minute = 0
					else:
						minute = 30
					obs_time = str(hour) + ":" + str(minute)

					# state_summary[state][obs_date][obs_time] = curr_temp

					query = """INSERT OR IGNORE INTO WEATHER(state, city, obs_date, hour, minute, curr_temp)
					VALUES (?,?,?,?,?,?)"""
					payload = (state, city, obs_date, hour, minute, curr_temp)
					dbexecute(cxn, query, payload)
		


if __name__ == '__main__':

	# set up the locations for data retrieval and storage, connect to db and create tables
	DATABASE = "dataset.db"
	cxn = sqlite3.connect(DATABASE)
	create_tables(cxn)

	weather_data_path = 'Weather/2018/'

	files = os.path.join(weather_data_path,"*")

	csv_to_sql(files)

	cxn.commit()
	cxn.close()
	



