from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import sklearn
import numpy as np
from sklearn.cluster import DBSCAN
import numpy as np
import math



def diff_metric(t1, t2):
    '''
    differences Metric of two (equal/unequal size) tranjes.
    :param t1: first tranjes.
    :param t2: second tranjes.
    :return: distance value.
    '''
    a = Proper_Trajectories.unflatten(t1)
    b = Proper_Trajectories.unflatten(t2)
    d = 0
    for coords1, coords2 in zip(a, b):
        d += math.sqrt(abs(coords1[0] - coords2[0]) + abs(coords1[1] - coords2[1]))
    return d


class Proper_Trajectories:
    def __init__(self, raw_trajectories ):
        self.raw_trajectories = raw_trajectories

    @staticmethod
    def pad_traj(traj, length):
        # padd traj with zeros, not in use!!!.
        if len(traj)>= length:
            return traj
        traj = traj + [[0,0] for i in range(length-len(traj))]
        return traj

    @staticmethod
    def mark_k_points(traj, k):
        return [traj[index] for index in list(range(0, len(traj), round(len(traj)/k)))]

    @staticmethod
    def flat_trajes(xx):
        def flatten(i):
            ll = []
            [[ll.append(d) for d in t] for t in i]
            return ll

        return np.array([flatten(i) for i in xx])

    @staticmethod
    def preprocess_trajs(trajes):
        '''
        build input for dbscan from row data (list of unequal size trajes).
        :param trajes: is a list of trajes
        :return: input to DBScan.
        '''
        shortest_traj = min(map(lambda x: len(x), trajes))
        representative_points_traj = [Proper_Trajectories.mark_k_points(trj, shortest_traj) for trj in trajes]
        input_dbscan = Proper_Trajectories.flat_trajes(representative_points_traj)
        return input_dbscan, np.array(representative_points_traj)

    @staticmethod
    def unflatten(yy):
        ll = []
        for i in range(0, len(yy), 2):
            ll.append([yy[i], yy[i+1]])
        return ll

    @staticmethod
    def opp_transform(xx):
        '''
        opposite input_db_scan tranjes to raw tranjes.
        :param xx:
        :return:
        '''
        return np.array([Proper_Trajectories.unflatten(i) for i in xx])

    @staticmethod
    def draw(row_trajes, db):
        # region draw clusters
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_

        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        print('Estimated number of clusters: %d' % n_clusters_)
        print('Estimated number of noise points: %d' % n_noise_)
        print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
        print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
        print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
        print("Adjusted Rand Index: %0.3f"
              % metrics.adjusted_rand_score(labels_true, labels))
        print("Adjusted Mutual Information: %0.3f"
              % metrics.adjusted_mutual_info_score(labels_true, labels))

        import matplotlib.pyplot as plt

        # Black removed and is used for noise instead.
        unique_labels = set(labels)
        colors = [plt.cm.Spectral(each)
                  for each in np.linspace(0, 1, len(unique_labels))]
        colors = ['b','r','g']
        for k, col in zip(unique_labels, colors):
            if k == -1:
                # Black used for noise.
                col = [0, 0, 0, 1]

            class_member_mask = (labels == k)

            xy = row_trajes[class_member_mask & core_samples_mask]
            for traj in xy:
                num_of_pts = len(traj)
                x = [x[0] for x in traj]
                y = [x[1] for x in traj]

                plt.plot(x, y, 'g', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

            xy = row_trajes[class_member_mask & ~core_samples_mask]
            for traj in xy:
                num_of_pts = len(traj)
                x = [x[0] for x in traj]
                y = [x[1] for x in traj]

                plt.plot(x, y, 'b', markerfacecolor=tuple(col),
                         markeredgecolor='k', markersize=6)

        plt.title('Estimated number of clusters: %d' % n_clusters_)
        # plt.savefig('/home/work/plot.jpg')
        plt.show()


row_trajes = list([[[1,1], [2,2], [3,3], [4,4], [5,5], [6,6]],
                    [[6.3,6.3], [5.3,5.3], [4.5,4.5], [3.5,3.5]],
                    [[9,9], [8,8], [7,7]]
                    ])

input_dbscan, shorten_row_trajes = Proper_Trajectories.preprocess_trajs(row_trajes)
zz = Proper_Trajectories.opp_transform(input_dbscan)
labels_true = np.array([0, 1, 1])
# diff_metric(uu[0], uu[1])

db = DBSCAN(eps=3, metric=diff_metric, min_samples=5).fit(input_dbscan)

Proper_Trajectories.draw(shorten_row_trajes, db)




# # region draw clusters
# core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
# core_samples_mask[db.core_sample_indices_] = True
# labels = db.labels_
#
# # Number of clusters in labels, ignoring noise if present.
# n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
# n_noise_ = list(labels).count(-1)
#
# print('Estimated number of clusters: %d' % n_clusters_)
# print('Estimated number of noise points: %d' % n_noise_)
# print("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
# print("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
# print("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
# print("Adjusted Rand Index: %0.3f"
#       % metrics.adjusted_rand_score(labels_true, labels))
# print("Adjusted Mutual Information: %0.3f"
#       % metrics.adjusted_mutual_info_score(labels_true, labels))
# # print("Silhouette Coefficient: %0.3f"
# #       % metrics.silhouette_score(X, labels))
#
# # #############################################################################
# # Plot result
# import matplotlib.pyplot as plt
#
# # Black removed and is used for noise instead.
# unique_labels = set(labels)
# colors = [plt.cm.Spectral(each)
#           for each in np.linspace(0, 1, len(unique_labels))]
# for k, col in zip(unique_labels, colors):
#     if k == -1:
#         # Black used for noise.
#         col = [0, 0, 0, 1]
#
#     class_member_mask = (labels == k)
#
#     xy = shorten_row_trajes[class_member_mask & core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=14)
#
#     xy = shorten_row_trajes[class_member_mask & ~core_samples_mask]
#     plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
#              markeredgecolor='k', markersize=6)
#
# plt.title('Estimated number of clusters: %d' % n_clusters_)
# plt.show()
#
# # endregion draw clusters


# X = np.array([[1, 1], [1, 2], [1, 3],
#            [1, 30], [1, 31], [1, 32], [1,20]])
# labels_true = np.array([0,0,0, 1,1,1,  2])
# db = DBSCAN(eps=3, min_samples=2).fit(X)

# DBSCAN(eps=3, min_samples=2)
# def func(x, y):
#     # dd = np.array([[i[1]-j[1] for i in x] for j in x])
#     dd = abs(x[1]-y[1])
#     return dd

# X = np.array([[1, 1], [1, 2], [1, 3],
#            [1, 30], [1, 31], [1, 32], [1,20]])
# labels_true = np.array([0,0,0, 1,1,1,  2])

# X = np.array([[1, 1 ,1,1], [2,10, 2], [3,10,3,3]])
