import os, glob, csv, sqlite3, sys
from geopy.geocoders import Nominatim

"""
	SQLITE3 to csv for analysis
	.headers on
	.mode csv
	.output state_weather.csv
	select * from weather where (obs_date like "2018%" and state="NSW");
"""

nsw_list = ["NSW", "New South Wales"]
qld_list = ["QLD", "Queensland"]
vic_list = ["VIC", "Victoria"]
sa_list = ["SA", "South Australia"]
tas_list = ["TAS", "Tasmania"]
nt_list = ["NT", "Northern Territory"]
wa_list = ["WA", "Western Australia"]

def create_tables(cxn):
	"""
	Function to create tables in sqlite3
	:param cxn: the connection to the sqlite3 database
	:return:
	"""
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS CLIMATE")
	cursor.execute("""CREATE TABLE IF NOT EXISTS CLIMATE(
		obs_year int,
		rcp varchar (3),
		season varchar(3),
		latitude float,
		longitude float,
		state varchar(3),
		temp_type varchar(3),
		temp_value float
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

	# initialise geolocator
	geolocator = Nominatim()
	file_count = 1
	state_list = []

	for file in glob.glob(files):

		state_count = 0

		obs_year, rcp, temp_type, season = file.split("/")[1][:-4].split(" ")
		temp_type = temp_type[0:3]

		# read each csv file by transcribing rows to columns for simpler extraction
		with open(file,'r', encoding='utf-8') as enc_in:
			reader = csv.reader(enc_in)

			# skip header rows
			for i in range(0,2):
				next(reader, None)

			# store data rows
			for row in reader:

				temp_value = row[3]
				latitude = row[1]
				longitude = row[2]

				# skip data that is outside of the Australian bounds
				if (float(longitude) < 109.69 or float(longitude) > 157.50):
					continue
				if (float(latitude) < -46.04 or float(latitude) > -6.98):
					continue

				# find the state of the data point
				if file_count == 1:
					lat_long = str(latitude) + "," + str(longitude)
					location = geolocator.reverse(lat_long, timeout=10)
					print (location.address)
					if location.address:
						address = location.address
						if any(substr in address for substr in nsw_list):
							state = "NSW"
						if any(substr in address for substr in qld_list):
							state = "QLD"
						if any(substr in address for substr in sa_list):
							state = "SA"
						if any(substr in address for substr in vic_list):
							state = "VIC"
						if any(substr in address for substr in tas_list):
							state = "TAS"
						if any(substr in address for substr in nt_list):
							state = "NT"
						if any(substr in address for substr in wa_list):
							state = "WA"
					else:
						state = "None"

					state_list.append(state)
				else:
					state = state_list[state_count]

				state_count += 1

				query = """INSERT OR IGNORE INTO CLIMATE(obs_year, rcp, season, latitude, longitude, state, temp_type, temp_value) VALUES (?,?,?,?,?,?,?,?)"""
				payload = (obs_year, rcp, season, latitude, longitude, state, temp_type, temp_value)
				dbexecute(cxn, query, payload)	

		file_count += 1


if __name__ == '__main__':

	# set up the locations for data retrieval and storage, connect to db and create tables
	DATABASE = "dataset.db"
	cxn = sqlite3.connect(DATABASE)
	create_tables(cxn)

	climate_data_path = 'Climate data/'

	files = os.path.join(climate_data_path,"*")

	csv_to_sql(files)

	cxn.commit()
	cxn.close()
	



