# K-Means Clustering

This repository contains a manual implementation of the K-Means clustering algorithm. K-Means is an unsupervised machine learning algorithm used for partitioning a dataset into K distinct, non-overlapping subsets (clusters).

## Table of Contents

- [Overview](#overview)
- [Usage](#usage)
- [Implementation](#implementation)
- [License](#license)

## Overview

K-Means clustering is a popular algorithm for data clustering. The main idea is to partition a set of data points into K clusters, where each point belongs to the cluster with the nearest mean. The algorithm iteratively refines the cluster assignments until convergence.

## Usage

To use this implementation, follow these steps:

1. Clone the repository:

   ```bash
   git clone https://github.com/yonmiru/kmeans-cluster.git
   ```

2. Navigate to the project directory:

   ```bash
   cd kmeans-cluster
   ```

3. Run the K-Means clustering algorithm on your dataset:

   ```bash
   python App.py
   ```

   Ensure you have Python installed on your system.

## Implementation

The `Kmeans.py` file contains the manual implementation of the K-Means clustering algorithm. The script includes functions for initializing centroids, assigning points to clusters, updating centroids, and running the main K-Means algorithm.
