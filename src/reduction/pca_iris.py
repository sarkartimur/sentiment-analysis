import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# https://archive.ics.uci.edu/dataset/53/iris
iris = fetch_ucirepo(id=53)
data = iris.data.original

print(data.head())

X = data.drop('class', axis=1).copy()
y = data['class'].copy()
X = scale(X)

"""
Build and run pca
"""
pca = PCA()
pca.fit(X) # find principal components
pca_data = pca.transform(X) # project original features into the new coordinate system

"""
Draw a scree plot
"""
per_var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var) + 1)]
plt.bar(x=range(1, len(per_var) + 1), height=per_var, tick_label=labels)
plt.xlabel('Principal component')
plt.ylabel('Percentage of explained variance')
plt.show()

"""
Draw a scatterplot of PC1 and PC2
"""
pca_df = pd.DataFrame(pca_data, index=y.index, columns=labels)
print(pca_df.head())
unique_classes = y.unique()
for class_label in unique_classes:
    class_data = pca_df[y == class_label]
    plt.scatter(
        class_data['PC1'],
        class_data['PC2'],
        label=class_label,
        alpha=0.8
    )
plt.xlabel(f'PC1 - {per_var[0]}%')
plt.ylabel(f'PC2 - {per_var[1]}%')
plt.title('PCA of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()

"""
Output loading scores for each feature
"""
loading_scores = pd.DataFrame(
    pca.components_.T,
    columns=[f'PC{i+1}' for i in range(len(pca.components_))],
    index=data.columns[:-1]
)
print(loading_scores)