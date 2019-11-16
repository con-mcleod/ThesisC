import requests
import os.path

states = ["QLD","NSW","VIC","SA","TAS"]
years = ["2010","2011","2012","2013","2014","2015","2016","2017","2018","2019"]
months = ["01","02","03","04","05","06","07","08","09","10","11","12"]

total_downloads = len(states) * len(years) * len(months)
print (total_downloads)
download_count = 0

for state in states:
	for year in years:
		for month in months:
			url = "https://aemo.com.au/aemo/data/nem/priceanddemand/PRICE_AND_DEMAND_" + year + month + "_" + state + "1.csv"
			response = requests.get(url)
			filename = year + month + "_" + state + "1.csv"

			# if filename already exists, skip
			if os.path.isfile(filename):
				print ("File already exists")
				download_count += 1
				print (str(total_downloads - download_count) + " downloads remaining")
				continue

			with open(filename, "wb") as handle:
				for data in response.iter_content():
					handle.write(data)
			download_count += 1

			remaining_downloads = total_downloads - download_count
			if (remaining_downloads == 0):
				print ("\nDownload complete!")
			else:
				print ("\n" + year + month + "_" + state + "1.csv --- download complete")
				print (str(remaining_downloads) + " downloads remaining")
