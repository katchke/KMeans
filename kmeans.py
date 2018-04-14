#!/usr/bin/python

import os
import sys

import random
import math

"""
Compute KMeans centroids
    - Add space-separated values in a file with a new point on each line
    - Run "python kmeans.py <file_name> <no_of_clusters_(k)>"
    - Position of centroids will be stored in clusters.txt
"""


class KMeans(object):
    def __init__(self, pts, k):
        """
        Initialize KMeans
        :param pts: Input points (list of lists)
        :param k: No. of clusters (int)
        """
        self.pts = pts
        self.k = k

        n_pts = len(pts)
        n_dim = len(pts[0])

        for pt in pts:
            if len(pt) != n_dim:
                raise ValueError('Dimensions of all points need to be the same')

        # Randomly assign k number of points as centroids
        self.centroids = [pts[i] for i in random.sample(range(0, n_pts - 1), k)]

    def compute(self):
        """
        Compute K-Means
        :return: k Centroids (list of lists)
        """
        itr = 0

        while True:
            distance_matrix = self._compute_distance_matrix(self.pts, self.centroids)

            clusters = [row.index(min(row)) for row in distance_matrix]  # Clusters the points belong to

            new_centroids = self._compute_centroids(self.pts, clusters)

            itr += 1

            # Check stopping criterion every 50 iterations
            if itr % 50 == 0:
                stop = True

                # Stop iterations if position of centroids do not change much (less than 0.01)
                for i in range(self.k):
                    if self._compute_euclidean_dist(new_centroids[i], self.centroids[i]) > 0.01:
                        stop = False

                if stop:
                    return self.centroids

            self.centroids = new_centroids

    @staticmethod
    def _compute_euclidean_dist(a, b):
        """
        Compute Euclidean distance between two points
        :param a: Point 1 (list)
        :param b: Point 2 (list)
        :return: Euclidean Distance (float)
        """

        assert len(a) == len(b)

        return math.sqrt(sum(map(lambda x, y: (x - y) ** 2, a, b)))

    def _compute_centroids(self, pts, cls):
        """
        Compute centroid of each cluster
        :param pts: Input points (list of lists)
        :param cls: Clusters the points belong to (same dimension as pts) (list)
        :return: Centroids for each cluster (list of lists)
        """

        assert len(pts) == len(cls)

        cts = []

        for m in range(self.k):
            cl_points = [pts[n] for n, cl in enumerate(cls) if cl == m]  # List of points belonging to cluster m
            cts.append(map(lambda *args: sum(args) / len(args), *cl_points))  # Average all points in cluster

        return cts

    def _compute_distance_matrix(self, pts, cts):
        """
        Compute distances between centroids and input points
        :param pts: Input points (list of lists)
        :param cts: Centroids for each cluster (list of lists)
        :return: Distance matrix of dimension n_points x n_centroids (list of lists)
        """

        return [[self._compute_euclidean_dist(c, p) for c in cts] for p in pts]


def to_float(string):
    """
    Convert string to float. Raise error if not possible
    :param string: String to convert (str)
    :return: Floating point number (float)
    """
    try:
        num = float(string)
    except ValueError:
        raise ValueError('Make sure all data points are numbers')

    return num


# Handle inputs

if not len(sys.argv) > 2:
    raise IOError('Provide name of file and number of clusters (k) as arguments')

fname = sys.argv[1]

try:
    K = int(sys.argv[2])  # No. of clusters
except ValueError:
    raise ValueError('Number of clusters needs to be an integer')

if not os.path.exists(fname):
    raise IOError('Input file does not exist')

f = open(fname, 'rb')
lines = f.readlines()
f.close()

if not len(lines):
    raise ValueError('File empty')

# Process inputs

points = [map(to_float, line.split(' ')) for line in lines]

# Compute centroids

kmeans = KMeans(points, K)
centroids = kmeans.compute()

# Handle output

f = open('clusters.txt', 'wb')

for centroid in centroids:
    f.write(' '.join(map(str, centroid)) + '\n')

f.close()
