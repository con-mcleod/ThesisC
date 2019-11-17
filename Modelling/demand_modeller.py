"""
Author: Connor McLeod
zID: z5058240
About: Script for developing demand forecasts
"""

import csv, sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets,linear_model, preprocessing
from sklearn import neighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from math import sqrt

def csv_to_df(data_file):
	"""
	Create a Pandas DataFrame from a CSV file
	:param data_file: csv file location
	"""
	df = pd.read_csv(data_file)
	return df


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


def create_tables(cxn):
	"""
	Function to create tables in sqlite3
	:param cxn: the connection to the sqlite3 database
	:return:
	"""
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS DEMAND_FORECAST")
	cursor.execute("""CREATE TABLE IF NOT EXISTS DEMAND_FORECAST(
		rfr_prediction float,
		knn_prediction float
		)""")
	cursor.close()


def plot_correlation(df):
	"""
	Plot a correlation heatmap for the provided dataframe
	:param df: Pandas dataframe
	"""
	corr = df.corr()
	mask = np.zeros_like(corr, dtype=np.bool)
	mask[np.triu_indices_from(mask)] = True
	f, ax = plt.subplots()
	cmap = sns.diverging_palette(220, 10, as_cmap=True)
	sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.8, center=0, 
				square=True, linewidths=.5, cbar_kws={"shrink": .5})
	plt.show()


def plot_pairplot(df, spot_price, features):
	"""
	Compare the provided data
	:param df: Pandas dataframe
	:param spot_price: The target variable
	:param features: Column headings to be compared
	"""
	pairplot_vars = spot_price + features
	df = df[pairplot_vars]
	sns.pairplot(data=df)
	plt.rcParams['figure.figsize']=(15,8)
	plt.show()


def performance_metrics(test_y,pred_y):
	"""
	Function to determine the R^2 value of the model prediction
	:param test_y: actual values
	:param pred_y: predicted values
	"""
	r2 = r2_score(test_y,pred_y)
	print ("R2-score: %.2f" % r2)

	print ("Demand actual average: {0:.2f}".format(sum(test_y)/len(test_y)))
	predicted_avg = pred_y['prediction'].mean()
	print ("Demand predicted average: ",round(predicted_avg,2))


def plot_prediction(test_y, y_predictions, model):
	"""
	"""
	plt.plot(test_y, 'b-', label='actual demand')
	plt.plot(y_predictions['prediction'],'ro', label=model, markersize=2)
	plt.legend()
	plt.xlabel('30-minute demand observation')
	plt.ylabel('Demand (MW)')
	plt.title(model)
	plt.show()


def plot_model_fit(train_x, train_y, test_x, test_y, predicted_y):
	"""
	"""
	plt.scatter(train_x[:,0],train_y)
	plt.scatter(test_x[:,0],test_y, color="red")
	plt.plot(test_x[:,0],predicted_y,color="green")
	plt.show()


def linear_regr(train_x, test_x, train_y, test_y):
	"""
	"""
	print ("\n# Linear regression model #")
	lin_reg = linear_model.LinearRegression()
	lin_reg.fit(train_x, train_y)
	lr_predict = lin_reg.predict(test_x)
	print ('Coefficients: ', lin_reg.coef_)
	print ('Y-int: ', lin_reg.intercept_)
	# plot_model_fit(train_x, train_y, test_x, test_y, lr_predict)
	lr_predictions = pd.DataFrame(data = {'prediction': lr_predict})
	plot_prediction(test_y, lr_predictions, "Linear regression")
	performance_metrics(test_y,lr_predictions)

	return lin_reg


def random_forest(train_x, test_x, train_y, test_y, features):
	"""
	"""
	print ("\n# Random forest regressor model #")

	rfr = RandomForestRegressor(n_estimators=10, max_features=num_features)
	rfr.fit(train_x, train_y)
	rfr_predict = rfr.predict(test_x)
	# plot_model_fit(train_x, train_y, test_x, test_y, rfr_predict)
	rfr_predictions = pd.DataFrame(data = {'prediction': rfr_predict})
	plot_prediction(test_y, rfr_predictions, "Random forest")
	performance_metrics(test_y,rfr_predictions)

	return rfr


def knn_model(train_x, test_x, train_y, test_y):
	"""
	"""
	print ("\n# k-NN model #")
	knn = neighbors.KNeighborsRegressor(n_neighbors = 3)
	knn.fit(train_x, train_y)
	knn_predict = knn.predict(test_x)
	# plot_model_fit(train_x, train_y, test_x, test_y, knn_predict)
	knn_predictions = pd.DataFrame(data = {'prediction': knn_predict})
	plot_prediction(test_y, knn_predictions, "K-nearest neighbour")
	performance_metrics(test_y,knn_predictions)

	return knn



if __name__ == '__main__':


	# create 2018 df from 2018 csv
	data_file = "datasets/tas_history.csv"
	df_train = csv_to_df(data_file)

	# create forecast df from forecast.db
	cxn = sqlite3.connect('datasets/tas_report.db')
	cursor = cxn.cursor()
	cursor.execute("DROP TABLE IF EXISTS DEMAND_FORECAST")
	cursor.close()

	df_test = pd.read_sql_query("SELECT * FROM forecast", cxn)
	# df_test = df_test.loc[df_test.RCP == RCP_value]
	df_test["temp_forecast"] = pd.to_numeric(df_test["temp_forecast"])
	df_test.rename(columns={'temp_forecast':'curr_temp'}, inplace=True)

	# add a lag term for demand
	df_train['demand_lag'] = df_train['demand'].shift(1)
	df_train = df_train.dropna()
	
	# define dependent and independent variables
	target = ['demand']
	features = ['curr_temp','hour']
	num_features = len(features)
	print ('\nFeatures used: ', features)

	# Plot variables in a pairplot
	# plot_pairplot(df_train, target, features)

	# Define training data (2018 data)
	data_x = df_train[features].values
	data_y = df_train['demand'].values
	train_x, test_x, train_y, test_y = train_test_split(data_x, data_y, test_size=0.05)

	forecast = df_test[features].values

	# linear regresssion
	lin_reg = linear_regr(train_x, test_x, train_y, test_y)
	LR_forecast = lin_reg.predict(forecast)
	LR_forecasts = pd.DataFrame(data = {'LR_demand': LR_forecast})

	# Random forest
	rfr = random_forest(train_x, test_x, train_y, test_y, features)
	rfr_forecast = rfr.predict(forecast)
	rfr_forecasts = pd.DataFrame(data = {'rfr_demand': rfr_forecast})

	# k-Nearest Neighbour
	knn = knn_model(train_x, test_x, train_y, test_y)
	knn_forecast = knn.predict(forecast)
	knn_forecasts = pd.DataFrame(data = {'knn_demand': knn_forecast})

	forecasts = pd.merge(rfr_forecasts, knn_forecasts, left_index=True, right_index=True)
	forecasts = pd.merge(df_test, forecasts, left_index=True, right_index=True)
	forecasts.to_sql(name='DEMAND_FORECAST', con=cxn)

	cxn.commit()
	cxn.close()



