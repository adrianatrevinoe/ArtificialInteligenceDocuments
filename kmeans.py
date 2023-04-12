"""
test-k-means
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly as py
import numpy as np
from sklearn.cluster import KMeans


DATA_FILE = "./data/Mall_Customers.csv"
RADOM_STATE = 13


def generate_image_name(name):
    return f"./graphs/{name}.png"


dataset = pd.read_csv(DATA_FILE)

print(dataset.head())
# Print information
print(dataset.info())
# Print statistics
print(dataset.describe())


pairplot = sns.pairplot(dataset,
                        vars=[
                            "Age", "Annual Income (k$)", "Spending Score (1-100)"],
                        hue="Gender"
                        )

X_COL = "Age"
Y_COL = "Spending Score (1-100)"
HUE = "Gender"


# fig = pairplot._figure
# fig.savefig(generate_image_name("pair_plot"))

# plt.figure(num=2, figsize=(10, 7))
# # plt.scatter(x="Age", y="Spending Score (1-100)", data=dataset, s=50)
# scatter_fig = sns.scatterplot(
# dataset, x=X_COL, y=Y_COL, hue=HUE).get_figure()
# scatter_fig.savefig(generate_image_name("scatter"))


def apply_kmeans(n_clusters, fig_number):

    # Apply K-Means
    kmeans_model = KMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        max_iter=500,
        tol=0.0001,
        random_state=RADOM_STATE,
        algorithm="elkan",
    )
    X = dataset[[X_COL, Y_COL]].to_numpy()
    kmeans_model.fit(X)
    labels = kmeans_model.labels_
    centroids = kmeans_model.cluster_centers_

    delta = 0.05

    xmin, xmax = X[:, 0].min() - 1, X[:, 0].max() + 1
    ymin, ymax = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(xmin, xmax, delta),
                         np.arange(ymin, ymax, delta))
    Z = kmeans_model.predict(np.c_[xx.ravel(), yy.ravel()])

    fig = plt.figure(num=fig_number, figsize=(10, 7))
    Z = Z.reshape(xx.shape)
    plt.imshow(Z,
               interpolation="nearest",
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Pastel2, aspect='auto', origin='lower')
    plt.scatter(x=X_COL, y=Y_COL,
                data=dataset, c=labels, s=100)
    plt.scatter(x=centroids[:, 0], y=centroids[:, 1],
                s=100, c="red", alpha=0.5)
    plt.title(f"N = {n_clusters}")
    plt.xlabel(X_COL)
    plt.ylabel(Y_COL)
    return fig


n_clusters = np.arange(2, 10, 1)

# for n in n_clusters:
# print(n)
# fig = apply_kmeans(n, n)
# fig.savefig(generate_image_name(f"n_means_{n}"))

# n = dataset.shape[0]
# fig = apply_kmeans(n, n)
# fig.savefig(generate_image_name(f"n_means_{n}"))
