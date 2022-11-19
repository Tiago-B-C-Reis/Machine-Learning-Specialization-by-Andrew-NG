# We will use familiar packages, NumPy, TensorFlow and helpful routines from scikit-learn.
# We will also use tabulate to neatly print tables and Pandas to organize tabular data.

import numpy as np
import numpy.ma as ma
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
# import tabulate
pd.set_option("display.precision", 1)

top10_df = pd.read_csv("./Content_based_filtering_data/movies.csv")

# Preparing the training data ----------------------------------------------------------------------------------------
# scale training data
item_train_unscaled = item_train
user_train_unscaled = user_train
y_train_unscaled = y_train

scalerItem = StandardScaler()
scalerItem.fit(item_train)
item_train = scalerItem.transform(item_train)

scalerUser = StandardScaler()
scalerUser.fit(user_train)
user_train = scalerUser.transform(user_train)

scalerTarget = MinMaxScaler((-1, 1))
scalerTarget.fit(y_train.reshape(-1, 1))
y_train = scalerTarget.transform(y_train.reshape(-1, 1))
# ynorm_test = scalerTarget.transform(y_test.reshape(-1, 1))

print(np.allclose(item_train_unscaled, scalerItem.inverse_transform(item_train)))
print(np.allclose(user_train_unscaled, scalerUser.inverse_transform(user_train)))
# --------------------------------------------------------------------------------------------------------------------

# Train_Set, Test_set split ------------------------------------------------------------------------------------------
item_train, item_test = train_test_split(item_train, train_size=0.80, shuffle=True, random_state=1)
user_train, user_test = train_test_split(user_train, train_size=0.80, shuffle=True, random_state=1)
y_train, y_test = train_test_split(y_train,    train_size=0.80, shuffle=True, random_state=1)
print(f"movie/item training data shape: {item_train.shape}")
print(f"movie/item test data shape: {item_test.shape}")
# --------------------------------------------------------------------------------------------------------------------

# Neural Network for content-based filtering -------------------------------------------------------------------------
# UNQ_C1
num_outputs = 32
tf.random.set_seed(1)
user_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)])

item_NN = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(32)])

# create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=num_user_features)
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=num_item_features)
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# compute the dot product of the two vectors vu and vm
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# specify the inputs and output of the model
model = tf.keras.Model([input_user, input_item], output)

model.summary()

# We will use a mean squared error loss and an Adam optimizer.
tf.random.set_seed(1)
cost_fn = tf.keras.losses.MeanSquaredError()
opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt, loss=cost_fn)
tf.random.set_seed(1)
model.fit([user_train[:, u_s:], item_train[:, i_s:]], y_train, epochs=30)

# Evaluate the model to determine loss on the test data.
model.evaluate([user_test[:, u_s:], item_test[:, i_s:]], y_test)
# --------------------------------------------------------------------------------------------------------------------

# Using the model to do 3 types of predictions:
# 1. Predictions for a new user.
# 2. Predictions for an existing user.
# 3. Finding Similar Items.

# 1. -----------------------------------------------------------------------------------------------------------------
new_user_id = 5000
new_rating_ave = 0.0
new_action = 5.0
new_adventure = 4.0
new_animation = 5.0
new_childrens = 0.0
new_comedy = 4.0
new_crime = 0.0
new_documentary = 4.0
new_drama = 0.0
new_fantasy = 5.0
new_horror = 0.0
new_mystery = 0.0
new_romance = 0.0
new_scifi = 4.0
new_thriller = 0.0
new_rating_count = 3

user_vec = np.array([[new_user_id, new_rating_count, new_rating_ave,
                      new_action, new_adventure, new_animation, new_childrens,
                      new_comedy, new_crime, new_documentary,
                      new_drama, new_fantasy, new_horror, new_mystery,
                      new_romance, new_scifi, new_thriller]])

# generate and replicate the user vector to match the number movies in the data set.
user_vecs = gen_user_vecs(user_vec,len(item_vecs))

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  # negate to get largest rating first
sorted_ypu = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display

print_pred_movies(sorted_ypu, sorted_items, movie_dict, maxcount = 50)


# 2. -----------------------------------------------------------------------------------------------------------------
uid = 7
# form a set of user vectors. This is the same vector, transformed and repeated.
user_vecs, y_vecs = get_user_vecs(uid, user_train_unscaled, item_vecs, user_to_genre)

# scale our user and item vectors
suser_vecs = scalerUser.transform(user_vecs)
sitem_vecs = scalerItem.transform(item_vecs)

# make a prediction
y_p = model.predict([suser_vecs[:, u_s:], sitem_vecs[:, i_s:]])

# unscale y prediction
y_pu = scalerTarget.inverse_transform(y_p)

# sort the results, highest prediction first
sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  # negate to get largest rating first
sorted_ypu = y_pu[sorted_index]
sorted_items = item_vecs[sorted_index]  # using unscaled vectors for display
sorted_user = user_vecs[sorted_index]
sorted_y = y_vecs[sorted_index]

# print sorted predictions for movies rated by the user
print_existing_user(sorted_ypu, sorted_y.reshape(-1,1), sorted_user, sorted_items, ivs, uvs, movie_dict, maxcount=50)


# 3. -----------------------------------------------------------------------------------------------------------------
# GRADED_FUNCTION: sq_dist
def sq_dist(a, b):
    """
    Returns the squared distance between two vectors
    Args:
      a (ndarray (n,)): vector with n features
      b (ndarray (n,)): vector with n features
    Returns:
      d (float) : distance
    """

    n = a.shape[0]
    d = 0
    for i in range(n):
        d += np.square(a[i] - b[i])
    # or:
    d1 = np.sum(np.square(a - b))

    return d
# ---------------------------------------------------------


# input layer
input_item_m = tf.keras.layers.Input(shape=num_item_features)
# use the trained item_NN
vm_m = item_NN(input_item_m)
# incorporate normalization as was done in the original model
vm_m = tf.linalg.l2_normalize(vm_m, axis=1)
model_m = tf.keras.Model(input_item_m, vm_m)
model_m.summary()

scaled_item_vecs = scalerItem.transform(item_vecs)
vms = model_m.predict(scaled_item_vecs[:,i_s:])
print(f"size of all predicted movie feature vectors: {vms.shape}")


count = 50  # number of movies to display
dim = len(vms)
dist = np.zeros((dim, dim))

for i in range(dim):
    for j in range(dim):
        dist[i, j] = sq_dist(vms[i, :], vms[j, :])

m_dist = ma.masked_array(dist, mask=np.identity(dist.shape[0]))  # mask the diagonal

disp = [["movie1", "genres", "movie2", "genres"]]
for i in range(count):
    min_idx = np.argmin(m_dist[i])
    movie1_id = int(item_vecs[i, 0])
    movie2_id = int(item_vecs[min_idx, 0])
    disp.append([movie_dict[movie1_id]['title'], movie_dict[movie1_id]['genres'],
                 movie_dict[movie2_id]['title'], movie_dict[movie1_id]['genres']]
                )
table = tabulate.tabulate(disp, tablefmt='html', headers="firstrow")

