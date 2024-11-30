import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data = pd.read_csv("alien_galaxy.csv")
threshold = 0.4 * len(data)
data = data.dropna(thresh=threshold, axis=1)

for col in data.select_dtypes(include=['float64', 'int64']).columns:
    data[col].fillna(data[col].median(), inplace=True)
for col in data.select_dtypes(include=['object']).columns:
    data[col].fillna(data[col].mode()[0], inplace=True)

data['Discovery_Date'] = pd.to_datetime(data['Discovery_Date'], errors='coerce')
data['Discovery_Year'] = data['Discovery_Date'].dt.year
data.drop(columns=['Discovery_Date'], inplace=True)

numerical_features = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
categorical_features = data.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ]
)

kmeans_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=5, random_state=42))
])

data['Cluster'] = kmeans_pipeline.fit_predict(data)

pca = PCA(n_components=2)
data_pca = pca.fit_transform(preprocessor.fit_transform(data))

data['PCA1'] = data_pca[:, 0]
data['PCA2'] = data_pca[:, 1]

plt.figure(figsize=(10, 7))
colors = ['red', 'blue', 'green', 'purple', 'orange']

for cluster in data['Cluster'].unique():
    cluster_data = data[data['Cluster'] == cluster]
    plt.scatter(cluster_data['PCA1'], cluster_data['PCA2'], 
                color=colors[cluster], label=f'Cluster {cluster}', alpha=0.6)

plt.title("Clustering of Alien-Colonized Planets")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(title='Cluster')
plt.grid(True)
plt.show()
