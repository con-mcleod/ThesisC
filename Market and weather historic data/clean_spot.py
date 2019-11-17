import os, glob, csv, sqlite3, sys

"""
	SQLITE3 to csv for analysis
	.headers on
	.mode csv
	.output state_spot.csv
	select * from SPOT_DATA where (obs_date like "2018%" and state="NSW");
"""

def create_tables(cxn):
	"""
	Function to create tables in sqlite3
	:param cxn: the connection to the sqlite3 database
	:return:
	"""
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS SPOT_DATA")
	cursor.execute("""CREATE TABLE IF NOT EXISTS SPOT_DATA(
		state varchar(3),
		obs_date varchar(10),
		hour int,
		minute int,
		demand float,
		spot_price float,
		unique(state, obs_date, hour, minute)
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

	for file in glob.glob(files):
		print (file)

		# read each csv file by transcribing rows to columns for simpler extraction
		with open(file,'r', encoding='utf-8') as enc_in:
			reader = csv.reader(enc_in)
			# skip header row
			next(reader, None)
			for row in reader:
				# if page didn't exist
				if '<html' in row[0]:
					print (file + " - Page did not exist")
					break

				state = row[0][0:3]
				if state == "SA1":
					state = state[0:2]
				state = state.lower()
					
				year = row[1][0:4]
				month = row[1][5:7]
				day = row[1][8:10]
				obs_date = str(year) + "/" + str(month) + "/" + str(day)
				hour = row[1][11:13]
				minute = row[1][14:16]
				demand = row[2]
				spot_price = row[3]

				query = """INSERT OR IGNORE INTO SPOT_DATA(state,obs_date,hour,minute,demand,spot_price)
				VALUES (?,?,?,?,?,?)"""
				payload = (state,obs_date,hour,minute,demand,spot_price)
				dbexecute(cxn, query, payload)



if __name__ == '__main__':

	# set up the locations for data retrieval and storage, connect to db and create tables
	DATABASE = "dataset.db"
	cxn = sqlite3.connect(DATABASE)
	create_tables(cxn)

	spot_data_path = 'AEMO Spot data/'

	files = os.path.join(spot_data_path,"*")

	csv_to_sql(files)

	cxn.commit()
	cxn.close()
	



