import pandas as pd
from sklearn.cluster import KMeans

def load_data():
   
		df = pd.read_csv('data/olympics.csv', index_col=0, skiprows=1)
		
		for col in df.columns:
			if col[:2]=='01':
				df.rename(columns={col:'Gold'+col[4:]}, inplace=True)
			if col[:2]=='02':
				df.rename(columns={col:'Silver'+col[4:]}, inplace=True)
			if col[:2]=='03':
				df.rename(columns={col:'Bronze'+col[4:]}, inplace=True)
			
		country_ids = df.index.str.split('(') 

		df.index = country_ids.str[0] 
		df['C_id'] = country_ids.str[1].str[:3]

		df = df.drop('Totals')
		
		return df		
    


def first_country(df):
	first_country = df.ix[0]
	"""print(first_country)"""
	return first_country
    


def gold_medal(df):
	gold_max = df['Gold'].argmax()
	
	return gold_max
	

def biggest_difference_in_gold_medal(df):
	bd = (df['Gold']-df['Gold.1']).argmax()
	return bd


def get_points(df):
	df['Points'] = df['Gold']*3 + df['Gold.1']*3 + df['Gold.2']*3 + df['Silver']*2 + df['Silver.1']*2 + df['Silver.2']*2 + df['Bronze']*1 + df['Bronze.1']*2 + df['Bronze.2']*1
	return df['Points']
	
"""Optimal k can be found using elbow method
	def k_means(df):
	
	# Number of clusters
	kmeans = KMeans(n_clusters=k)
	# Fitting the input data
	kmeans = kmeans.fit(X)
	# Getting the cluster labels
	labels = kmeans.predict(X)
	# Centroid values
	centroids = kmeans.cluster_centers_
	
	return k, centroids
		
"""
d = load_data()
first_country(d)
gold_medal(d)
biggest_difference_in_gold_medal(d)
get_points(d)
