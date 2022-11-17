import numpy as np
import tensorflow as tf
from tensorflow import keras

movieList_df = '../small_movie_list.txt'


# --------------------------------------------------------------------------------------------------------------------
# GRADED FUNCTION: cofi_cost_func
# UNQ_C1
def cofi_cost_func(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    nm, nu = Y.shape
    J = 0

    for j in range(nu):
        w_j = W[j]
        b_j = b[0, j]
        for i in range(nm):
            x_i = X[i]
            y_i_j = Y[i, j]
            r_i_j = R[i, j]

            J += np.square(r_i_j * ((np.dot(w_j, x_i) + b_j) - y_i_j))

    J = J / 2
    J += (lambda_/2) * (np.sum(np.square(W)) + np.sum(np.square(X)))

    return J
# --------------------------------------------------------------------------------------------------------------------


# Vectorized Implementation
# It is important to create a vectorized implementation to compute  𝐽 , since it will later be called many times during
# optimization. The linear algebra utilized is not the focus of this series, so the implementation is provided.
# If you are an expert in linear algebra, feel free to create your version without referencing the code below.
# Run the code below and verify that it produces the same results as the non-vectorized version.
def cofi_cost_func_v(X, W, b, Y, R, lambda_):
    """
    Returns the cost for the content-based filtering
    Vectorized for speed. Uses tensorflow operations to be compatible with custom training loop.
    Args:
      X (ndarray (num_movies,num_features)): matrix of item features
      W (ndarray (num_users,num_features)) : matrix of user parameters
      b (ndarray (1, num_users)            : vector of user parameters
      Y (ndarray (num_movies,num_users)    : matrix of user ratings of movies
      R (ndarray (num_movies,num_users)    : matrix, where R(i, j) = 1 if the i-th movies was rated by the j-th user
      lambda_ (float): regularization parameter
    Returns:
      J (float) : Cost
    """
    j = (tf.linalg.matmul(X, tf.transpose(W)) + b - Y)*R
    J = 0.5 * tf.reduce_sum(j**2) + (lambda_/2) * (tf.reduce_sum(X**2) + tf.reduce_sum(W**2))
    return J
# --------------------------------------------------------------------------------------------------------------------


# Reload ratings
Y, R = load_ratings_small()
# Add new user ratings to Y
Y = np.c_[my_ratings, Y]
# Add new user indicator matrix to R
R = np.c_[(my_ratings != 0).astype(int), R]
# Normalize the Dataset
Ynorm, Ymean = normalizeRatings(Y, R)
#  Useful Values
num_movies, num_users = Y.shape
num_features = 100
# Set Initial Parameters (W, X), use tf.Variable to track these variables
tf.random.set_seed(1234) # for consistent results
W = tf.Variable(tf.random.normal((num_users,  num_features),dtype=tf.float64),  name='W')
X = tf.Variable(tf.random.normal((num_movies, num_features),dtype=tf.float64),  name='X')
b = tf.Variable(tf.random.normal((1,          num_users),   dtype=tf.float64),  name='b')

# Instantiate an optimizer.
optimizer = keras.optimizers.Adam(learning_rate=1e-1)
iterations = 200
for iter in range(iterations):
    # Use TensorFlow's GrandientTape
    # to record the operations used to compute the cost.
    with tf.GradientTape() as tape:
        # Compute the cost (forward pass is included in cost)
        cost_value = cofi_cost_func_v(X, W, b, Y, R, nu, nm, lambda_)
    # Use the gradient tape to automatically retrieve
    # the gradient of the trainable variables with
    # respect to the loss.
    grads = tape.gradient(cost_value, [X, W, b])

    # Run one step of gradient decent by updating
    # the value of the variables to minimize the loss.
    optimizer.apply_gradients(zip(grads, [X, W, b]))

    # Log periodically.
    if iter % 10 == 0:
        print(f"Training loss at iteration {iter}: {cost_value:0.1f}")


# Make a prediction using trained weights and biases
p = np.matmul(X.numpy(), np.transpose(W.numpy())) + b.numpy()

# restore the mean
pm = p + Ymean

my_predictions = pm[:, 0]

# sort predictions
ix = tf.argsort(my_predictions, direction='DESCENDING')

