from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score, silhouette_score, silhouette_samples
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from kneed import KneeLocator

import time

title_fontsize = 24
fontsize = 24
legend_fontsize = 18
default_figure_size = (15, 8)

km_arguements = {"init": 'random',
                 "n_init": 10,
                 "max_iter": 500,
                 "tol": 1e-04,
                 "random_state": 42}

em_arguements = {"covariance_type": 'full',
                 "random_state": 0}

def compute_kmeans_scores(X, n):
    davies_bouldin_scores = []
    distortions = []
    silhouette_scores = []
    calinski_harabasz_scores = []
    times = []

    for i in range(2, n+1):
        start_time = time.time()
        km = KMeans(n_clusters=i, **km_arguements)
        km.fit(X)
        time_taken = time.time() - start_time
        times.append(time_taken)
        distortions.append(km.inertia_)
        davies_bouldin_scores.append(davies_bouldin_score(X, km.labels_))
        silhouette_avg = silhouette_score(X, km.labels_)
        silhouette_scores.append(silhouette_avg)
        calinski_harabasz_scores.append(calinski_harabasz_score(X, km.labels_))
        print("For n_clusters = {} average silhouette_score: {} time taken: {}s".format(i, silhouette_avg, time_taken))
    return distortions, davies_bouldin_scores, silhouette_scores, calinski_harabasz_scores, times


def plot_kmeans_all(distortions, davies_bouldin_scores, silhouette_scores, calinski_harabasz_scores, times, DATASET, tick_spacing=2):
    n = len(distortions)+2
    db_copy = davies_bouldin_scores.copy()
    silhouette_copy = silhouette_scores.copy()
    distortions_copy = distortions.copy()
    ch_copy = calinski_harabasz_scores.copy()

    db_copy = db_copy / max(db_copy)
    silhouette_copy = silhouette_copy / max(silhouette_copy)
    distortions_copy = np.array(distortions_copy) / max(distortions_copy)
    ch_copy = ch_copy / max(ch_copy)

    # https://matplotlib.org/2.2.2/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig = plt.figure(figsize=default_figure_size)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=fontsize)
    host = fig.add_subplot(111)
    host.set_xticks(range(2, n, tick_spacing))
    host.xaxis.grid(True)

    par1 = host.twinx()

    p1, = host.plot(range(2, n), db_copy, color="blue", marker="o", label="Davies Bouldin Index")
    p2, = host.plot(range(2, n), silhouette_copy, color="red", marker="o",  label="Silhouette Coefficient")
    p3, = host.plot(range(2, n), distortions_copy, color="orange", marker="o",  label="Distortion Score")
    p4, = host.plot(range(2, n), ch_copy, color="darkviolet", marker="o",  label="Calinski Harabasz Index")

    p5, = par1.plot(range(2, n), times, color="yellowgreen", label="Time (s)")

    host.grid(False)
    par1.grid(False)
    host.xaxis.grid(True)

    host.set_xlabel("Number of Clusters (k)", fontsize=fontsize, fontweight='bold')
    host.set_ylabel("Normalized Scores", fontsize=fontsize, fontweight='bold')
    par1.set_ylabel("Time (s)", fontsize=fontsize, fontweight='bold')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p5.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]

    host.legend(lines, [l.get_label() for l in lines], fontsize=legend_fontsize)

    plt.title("{} Dataset KMeans Cluster Size Comparison".format(DATASET), fontsize=title_fontsize, fontweight='bold')
    plt.show()


def plot_kmeans_selection(davies_bouldin_scores, silhouette_scores, best_k, DATASET, tick_spacing=4):
    n = len(davies_bouldin_scores)+2
    # https://matplotlib.org/2.2.2/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig = plt.figure(figsize=default_figure_size)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    host = fig.add_subplot(111)
    host.set_xticks(range(2, n, tick_spacing))
    host.xaxis.grid(True)

    par1 = host.twinx()

    p3 = plt.axvline(x=best_k, color='k', linestyle='--', label="k = {}".format(best_k))

    p1, = host.plot(range(2, n), silhouette_scores, color="red", marker="o", label="Silhouette Coefficient")
    p2, = par1.plot(range(2, n), davies_bouldin_scores, color="blue", marker="o", label="Davies Bouldin Index")

    host.grid(False)
    par1.grid(False)
    host.xaxis.grid(True)

    host.set_xlabel("Number of Clusters (k)", fontsize=fontsize, fontweight='bold')
    host.set_ylabel("Silhouette Coefficient", fontsize=fontsize, fontweight='bold')
    par1.set_ylabel("Davies Bouldin Index", fontsize=fontsize, fontweight='bold')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3]

    host.legend(lines, [l.get_label() for l in lines], fontsize=legend_fontsize)

    plt.title("{} Dataset KMeans Cluster Selection".format(DATASET), fontsize=title_fontsize, fontweight='bold')
    plt.show()


def plot_silhouette_kmeans(X, best_k, DATASET, xlim):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    n_clusters = best_k

    fig = plt.figure(figsize=default_figure_size)
    ax1 = fig.add_subplot(111)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_xlim(xlim)
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    start_time = time.time()
    clusterer = KMeans(n_clusters=n_clusters, **km_arguements)
    cluster_labels = clusterer.fit_predict(X)
    time_taken = time.time() - start_time

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters = {} average silhouette_score: {} time taken: {}s".format(best_k, silhouette_avg, time_taken))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("{} Dataset KMeans Silhouette Plot K = {}".format(DATASET, best_k), fontsize=title_fontsize,
                  fontweight='bold')
    ax1.set_xlabel("Silhouette Coefficient Values", fontsize=fontsize)
    ax1.set_ylabel("Cluster Label", fontsize=fontsize)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=fontsize)
    plt.show()

def compute_em_scores(X, n):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
    davies_bouldin_scores = []
    bic_scores = []
    aic_scores = []
    silhouette_scores = []
    times = []

    for i in range(2, n+1):
        start_time = time.time()
        em = GaussianMixture(i, **em_arguements)
        em.fit(X)
        time_taken = time.time() - start_time
        times.append(time_taken)
        davies_bouldin_scores.append(davies_bouldin_score(X, em.predict(X)))
        bic_scores.append(em.bic(X))
        aic_scores.append(em.aic(X))
        silhouette_avg = silhouette_score(X, em.predict(X))
        silhouette_scores.append(silhouette_avg)
        print("For n_clusters = {} average silhouette_score: {} time taken: {}s".format(i, silhouette_avg, time_taken))
    return davies_bouldin_scores, silhouette_scores, bic_scores, aic_scores, times

def plot_em_all(davies_bouldin_scores, silhouette_scores, bic_scores, aic_scores, times, DATASET, tick_spacing=1):
    n = len(davies_bouldin_scores)+2
    db_copy = davies_bouldin_scores.copy()
    silhouette_copy = silhouette_scores.copy()
    bic_scores_copy = bic_scores.copy()
    aic_scores_copy = aic_scores.copy()

    db_copy = db_copy / max(db_copy)
    silhouette_copy = silhouette_copy / max(silhouette_copy)
    bic_scores_copy = np.array(bic_scores_copy) / max(bic_scores_copy)
    aic_scores_copy = aic_scores_copy / max(aic_scores_copy)

    # https://matplotlib.org/2.2.2/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig = plt.figure(figsize=default_figure_size)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=fontsize)
    host = fig.add_subplot(111)
    host.set_xticks(range(2, n, tick_spacing))
    host.xaxis.grid(True)

    par1 = host.twinx()

    p1, = host.plot(range(2, n), db_copy, color="blue", marker="o", label="Davies Bouldin Index")
    p2, = host.plot(range(2, n), silhouette_copy, color="red", marker="o",  label="Silhouette Coefficient")
    p3, = host.plot(range(2, n), bic_scores_copy, color="orange", marker="o",  label="Bayesian information criterion")
    p4, = host.plot(range(2, n), aic_scores_copy, color="darkviolet", marker="o",  label="Akaike information criterion")

    p5, = par1.plot(range(2, n), times, color="yellowgreen", label="Time (s)")

    host.grid(False)
    par1.grid(False)
    host.xaxis.grid(True)

    host.set_xlabel("Number of Clusters (k)", fontsize=fontsize, fontweight='bold')
    host.set_ylabel("Normalized Scores", fontsize=fontsize, fontweight='bold')
    par1.set_ylabel("Time (s)", fontsize=fontsize, fontweight='bold')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p5.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p5.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]

    host.legend(lines, [l.get_label() for l in lines], fontsize=legend_fontsize)

    plt.title("{} Dataset Expectation Maximization Cluster Size Comparison".format(DATASET), fontsize=title_fontsize, fontweight='bold')
    plt.show()


# https://matplotlib.org/2.2.2/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
def make_patch_spines_invisible(ax):
    ax.set_frame_on(True)
    ax.patch.set_visible(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

def plot_em_selection(davies_bouldin_scores_em, silhouette_scores_em, bic_scores, aic_scores, best_k, DATASET, tick_spacing=1):
    n = len(davies_bouldin_scores_em)+2
    n_components = np.arange(2, n)
    # https://matplotlib.org/2.2.2/gallery/ticks_and_spines/multiple_yaxis_with_spines.html
    fig = plt.figure(figsize=default_figure_size)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    host = fig.add_subplot(111)
    plt.yticks(fontsize=fontsize)
    par1 = host.twinx()
    plt.yticks(fontsize=fontsize)
    par2 = host.twinx()
    plt.yticks(fontsize=fontsize)

    # Offset the right spine of par2.  The ticks and label have already been
    # placed on the right by twinx above.
    par2.spines["right"].set_position(("axes", 1.1))
    # Having been created by twinx, par2 has its frame off, so the line of its
    # detached spine is invisible.  First, activate the frame but make the patch
    # and spines invisible.
    make_patch_spines_invisible(par2)
    # Second, show the right spine.
    par2.spines["right"].set_visible(True)

    p3 = plt.axvline(x=best_k, color='k', linestyle='--', label="k = {}".format(best_k))

    p1, = host.plot(n_components, silhouette_scores_em, color="red", marker="o", label="Silhouette Score")
    p2, = par1.plot(n_components, davies_bouldin_scores_em, color="blue", marker="o", label="Davies Bouldin Score")
    p4, = par2.plot(n_components, bic_scores, color="green", marker="o", label="BIC Score")
    p5, = par2.plot(n_components, aic_scores, color="limegreen", marker="o", label="AIC Score")

    host.grid(False)
    par1.grid(False)
    par2.grid(False)
    host.xaxis.grid(True)

    host.set_xlabel("Number of Clusters (k)", fontsize=fontsize, fontweight='bold')
    host.set_ylabel("Silhouette Score", fontsize=fontsize, fontweight='bold')
    par1.set_ylabel("Davies Bouldin Score", fontsize=fontsize, fontweight='bold')
    par2.set_ylabel("BIC / AIC Scores", fontsize=fontsize, fontweight='bold')

    host.yaxis.label.set_color(p1.get_color())
    par1.yaxis.label.set_color(p2.get_color())

    tkw = dict(size=4, width=1.5)
    host.tick_params(axis='y', colors=p1.get_color(), **tkw)
    par1.tick_params(axis='y', colors=p2.get_color(), **tkw)
    host.tick_params(axis='x', **tkw)

    lines = [p1, p2, p3, p4, p5]

    host.set_xticks(range(2, n, tick_spacing))

    plt.title("{} Dataset Expectation Maximization Cluster Selection".format(DATASET), fontsize=title_fontsize,
              fontweight='bold')
    host.legend(lines, [l.get_label() for l in lines], fontsize=legend_fontsize, loc="center right")
    plt.show()

def plot_silhouette_em(X, best_k, DATASET, xlim):
    # https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
    n_clusters = best_k

    fig = plt.figure(figsize=default_figure_size)
    ax1 = fig.add_subplot(111)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1 but in this example all
    ax1.set_xlim(xlim)
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 10 for reproducibility.
    start_time = time.time()
    em_model = GaussianMixture(best_k, **em_arguements)
    em_model.fit(X)
    cluster_labels = em_model.predict(X)
    time_taken = time.time() - start_time

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(X, cluster_labels)
    print("For n_clusters = {} average silhouette_score: {} time taken: {}s".format(best_k, silhouette_avg, time_taken))

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(n_clusters):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / n_clusters)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("{} Dataset Expectation Maximization Silhouette Plot K = {}".format(DATASET, best_k), fontsize=title_fontsize,
                  fontweight='bold')
    ax1.set_xlabel("Silhouette Coefficient Values", fontsize=fontsize)
    ax1.set_ylabel("Cluster Label", fontsize=fontsize)

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    # ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.show()

def plot_pca_component_selection(X, num_features, DATASET, explained_variance_threshold=0.9, x_tick_spacing=20, ax2_y_tick_spacing=0.005):
    fontsize = 16
    pca_reconstruction_error = []
    pca = PCA(n_components=num_features)
    pca.fit(X)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=default_figure_size, sharey=False)
    fig.suptitle("{} Dataset PCA Component Selection".format(DATASET), fontsize=fontsize)
    x = [i for i in range(pca.n_components)]

    for n in x:
        tmp_pca = PCA(n_components=n)
        X_pca = tmp_pca.fit_transform(X)

        # https://intellipaat.com/community/22811/pca-projection-and-reconstruction-in-scikit-learn
        X_projected = tmp_pca.inverse_transform(X_pca)
        loss = ((X - X_projected) ** 2).mean()
        pca_reconstruction_error.append(np.sum(loss))

    if pca_reconstruction_error is not None:
        par1 = ax1.twinx()
        p3, = par1.plot(x, pca_reconstruction_error, color="orange", label="Reconstruction Error")
        par1.set_ylabel("Reconstruction Error", fontsize=16)

    # kneedle = KneeLocator(x, np.cumsum(pca.explained_variance_ratio_), S=1.0, curve="concave", direction="increasing")
    # ax1.axvline(x=kneedle.knee, color='k', linestyle='--', label="Maximum Curvature = {}".format(kneedle.knee))

    num_components_explained_varaince_threshold = np.argmax(np.cumsum(pca.explained_variance_ratio_) > explained_variance_threshold)
    p2 = ax1.axvline(x=np.argmax(np.cumsum(pca.explained_variance_ratio_) > explained_variance_threshold), color='b', linestyle='--',
                label="{:.0f}% Explained Variance".format(explained_variance_threshold*100, num_components_explained_varaince_threshold))

    p1, = ax1.plot(np.cumsum(pca.explained_variance_ratio_), label="Explained Variance")
    ax1.set_xlabel('Number of components', fontsize=fontsize)
    ax1.set_ylabel('Explained Variance Ratio', fontsize=fontsize)
    ax1.tick_params(axis='y', size=20)
    ax1.xaxis.set_ticks(x[::x_tick_spacing])
    ax1.tick_params(labelsize=16)
    ax1.yaxis.set_ticks([i for i in np.arange(0, 1.1, 0.1)])
    ax1.grid(True)
    ax1.set_title("Cumulative Explained Variance Vs Number of Components", fontsize=14)

    lines = [p1, p2, p3]
    ax1.legend(lines, [l.get_label() for l in lines], loc='center right', fontsize=14)

    ax2.bar(x=x, height=pca.explained_variance_ratio_)
    ax2.set_xlabel('Number of components', fontsize=fontsize)
    ax2.set_ylabel('Explained Variance Ratio', fontsize=fontsize)
    ax2.xaxis.set_ticks(x[::x_tick_spacing])
    ax2.set_yticks([i for i in np.arange(0, max(pca.explained_variance_ratio_), ax2_y_tick_spacing)])
    ax2.tick_params(labelsize=16)
    ax2.grid(True)
    ax2.set_title("Explained Variance per Component", fontsize=14)

    plt.tight_layout()
    plt.show()

    print("Explained Variance = {:.0f}% Num Componenets: {} ".format(explained_variance_threshold*100, num_components_explained_varaince_threshold))
    print("Reconstruction Error for Num Components = {} = {}".format(num_components_explained_varaince_threshold, pca_reconstruction_error[num_components_explained_varaince_threshold]))

def plot_ica_selection(x, kurtosis, best_n_components, best_kurtosis, DATASET, ica_reconstruction_error=None, x_tick_spacing=20, ax2_y_ticks=20):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=default_figure_size, sharey=False)
    fig.suptitle("{} Dataset ICA Component Selection".format(DATASET), fontsize=fontsize)

    p1 = ax1.axvline(x=best_n_components, color='k', linestyle='--', label="k = {}".format(best_n_components))

    ax1.set_xlabel("Independent Components", fontsize=fontsize)
    ax1.set_ylabel("Average Kurtosis", fontsize=fontsize)
    p2, = ax1.plot(x, kurtosis, 'bo-', label="Avg Kurtosis")
    ax1.xaxis.set_ticks([i for i in range(min(x), max(x)+x_tick_spacing, x_tick_spacing)])

    ax1.tick_params(axis='y', size=20)
    ax1.tick_params(labelsize=16)
    ax1.grid(True)
    ax1.set_title("Average Kurtosis Vs Number of Components", fontsize=16)

    if ica_reconstruction_error is not None:
        par1 = ax1.twinx()
        p3, = par1.plot(x, ica_reconstruction_error, color="orange", label="Reconstruction Error")
        par1.set_ylabel("Reconstruction Error", fontsize=16)
        lines = [p1, p2, p3]
    else:
        lines = [p1, p2]

    ax1.legend(lines, [l.get_label() for l in lines], loc='best', fontsize=16)

    ax2.bar(x=[i for i in range(1, best_n_components+1)], height=best_kurtosis)
    ax2.set_xlabel("Independent Components", fontsize=fontsize)
    ax2.set_ylabel("Kurtosis", fontsize=fontsize)

    ax2.set_yticks([i for i in range(0, int(max(best_kurtosis)+ax2_y_ticks), ax2_y_ticks)])
    ax2.set_xticks([i for i in range(1, best_n_components+x_tick_spacing, x_tick_spacing)])
    ax2.tick_params(labelsize=16)
    ax2.grid(True)
    ax2.set_title("Kurtosis per Component for {} Components".format(best_n_components), fontsize=16)

    plt.tight_layout()
    plt.show()
