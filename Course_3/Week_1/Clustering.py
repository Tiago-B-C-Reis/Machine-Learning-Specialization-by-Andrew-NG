import numpy as np
import data_set


# Random initialization - It's a method that randomly associates K centroids 'coordinates' to some X(i) in order
# for the algorithm to start from somewhere.
def kMeans_init_centroids(X, K):
    """
    This function initializes K centroids that are to be
    used in K-Means on the dataset X
    Args:
        X (ndarray): Data points
        K (int):     number of centroids/clusters
    Returns:
        centroids (ndarray): Initialized centroids
    """

    # Randomly reorder the indices of examples. Creates a list with the X.shape[0] dimension and transforms
    # the X(i) values in its index and them creates a np.array with index that are randomly positioned.
    randidx = np.random.permutation(X.shape[0])

    # Uses the first K index from the randidx array and picks those from the X array, resulting in
    # taking K X(i) examples from X array.
    centroids = X[randidx[:K]]

    return centroids
# -------------------------------------------------------------------------------------------------------------


# This function calculates C(i), in other words it calculates de distance between X(i) and the centroid(i) and
# relates every X(i) with an index that belongs to its closest centroid.
# This function is basically the calculation of Cost function 'J()'.
def find_closest_centroids(X, centroids):
    """
    Computes the centroid memberships for every example
    Args:
        X (ndarray): (m, n) Input values
        centroids (ndarray): k centroids
    Returns:
        idx (array_like): (m,) closest centroids
    """

    # Set K
    K = centroids.shape[0]

    # You need to return the following variables correctly
    idx = np.zeros(X.shape[0], dtype=int)

    for i in range(X.shape[0]):
        distance = []
        for j in range(centroids.shape[0]):
            norm_ij = np.linalg.norm(X[i] - centroids[j])
            distance.append(norm_ij)
        idx[i] = np.argmin(distance)

    return idx
# -------------------------------------------------------------------------------------------------------------


# This function calculates the mean for each C(i) using its related X(i) and relocates the centroid to be more in
# the center of its cluster.
def compute_centroids(X, idx, K):
    """
    Returns the new centroids by computing the means of the
    data points assigned to each centroid.
    Args:
        X (ndarray):   (m, n) Data points
        idx (ndarray): (m,) Array containing index of the closest centroid for each
                       example in X. Concretely, idx[i] contains the index of
                       the centroid closest to example i
        K (int):       number of centroids
    Returns:
        centroids (ndarray): (K, n) New centroids computed
    """

    # Useful variables
    m, n = X.shape

    # You need to return the following variables correctly
    centroids = np.zeros((K, n))

    for k in range(centroids.shape[0]):
        points = X[idx == k]
        centroids[k] = np.mean(points, axis=0)

    return centroids
# -------------------------------------------------------------------------------------------------------------


# This function uses the two functions above to relocate the centroids x times, depending on the nº of iterations
# we want. We calculate here the Cost function of K-means for a defined number of iterations and find the
# clusters centroid coordinates with the lowest J() found in the number of iterations performed.
def run_kMeans(X, initial_centroids, max_iters=50):
    """
    Runs the K-Means algorithm on data matrix X, where each row of X
    is a single example
    """

    # Initialize values
    m, n = X.shape
    K = initial_centroids.shape[0]
    centroids = initial_centroids
    previous_centroids = centroids
    idx = np.zeros(m)

    # Run K-Means
    for i in range(max_iters):

        # Output progress
        print(f'K-Means iteration {i}/{max_iters-1}')

        # For each example in X, assign it to the closest centroid
        idx = find_closest_centroids(X, centroids)

        # Given the memberships, compute new centroids
        centroids = compute_centroids(X, idx, K)

    return centroids, idx
# -------------------------------------------------------------------------------------------------------------


# Load an example dataset
X = data_set.X

# Set initial centroids
K = 3
initial_centroids = kMeans_init_centroids(X, K)

# Number of iterations
max_iters = 100

centroids, idx = run_kMeans(X, initial_centroids, max_iters)
print("The centroids are:\n", centroids)
