import pandas as pd
import umap
from sklearn.preprocessing import scale, LabelEncoder
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo


RANDOM_SEED = 42


iris = fetch_ucirepo(id=53)
data = iris.data.original

X = data.drop('class', axis=1).copy()
y = data['class'].copy()

X_scaled = scale(X)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

umap = umap.UMAP(
    n_components=2,
    random_state=RANDOM_SEED,
    n_neighbors=15,
    min_dist=0.1,
    metric='euclidean'
)

# Fit and transform with supervision (using class labels)
umap_data = umap.fit_transform(X_scaled, y=y_encoded)

umap_df = pd.DataFrame(umap_data, columns=['UMAP1', 'UMAP2'])
umap_df['class'] = y.values
plt.figure(figsize=(10, 6))
for class_label in le.classes_:
    class_data = umap_df[umap_df['class'] == class_label]
    plt.scatter(
        class_data['UMAP1'],
        class_data['UMAP2'],
        label=class_label,
        alpha=0.8,
        s=50
    )
plt.xlabel('UMAP Dimension 1')
plt.ylabel('UMAP Dimension 2')
plt.title('Supervised UMAP Projection of Iris Dataset')
plt.legend()
plt.grid(True)
plt.show()