from __future__ import print_function

import glob
import math
import os
import urllib2
import json
import pprint as pp

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = "{:.1f}".format

keys = ["C", "C_Sharp", "D", "D_Sharp", "E", "F", "F_Sharp", "G", "G_Sharp", "A", "A_Sharp", "B"]
time_signatures = ["sig0", "sig1", "sig2", "sig3", "sig4", "sig5"]
modes = ["minor", "major"]

def get_track_ids_url(track_ids):
    """ Build the GET request to get the information for a list of tracks

      Args:
        track_ids: A list of track ids
      Returns:
        A url string
    """
    url = "https://api.spotify.com/v1/audio-features?ids="
    for id in track_ids:
        if id:
            url += id + ","
    return url[:-1]

def add_one_hot_encodings(track):
    for key in keys:
        track[key] = 0
    track[keys[track["key"]]] = 1
    for time_signature in time_signatures:
        track[time_signature] = 0
    track[time_signatures[track["time_signature"]]] = 1
    for mode in modes:
        track[mode] = 0
    track[modes[track["mode"]]] = 1

def getData(playlist_url, top_40):
    """ Build the pandas dataset of track information by gathering dta from the
        Spotify API

      Returns:
        A dataset of tracks with their information
    """
    # key is going to change frequently because it expires -- hopefully will find a workaround
    request_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer BQBei5fVo7JTEojGqppgfpGkCD63R92h-AF3BWpIPq3ovArhJ80cpGooSiFNd_YwK7n-rJxQHxSFoq-Q6ZGprcVYAHLPUvbEcyw6UU4ZItngG_gU9XzUW2pERUFYdzsu7ZZQyl-udTt-"
    }
    playlist_request = urllib2.Request(playlist_url, headers=request_headers)
    playlist_contents = urllib2.urlopen(playlist_request)
    playlist_contents_json = json.load(playlist_contents)
    track_id_list = []
    for item in playlist_contents_json["tracks"]["items"]:
        track_id_list.append(item["track"]["id"])
    tracks_request = urllib2.Request(get_track_ids_url(track_id_list), headers=request_headers)
    tracks_contents = urllib2.urlopen(tracks_request)
    tracks_contents_json = json.load(tracks_contents)
    for track in tracks_contents_json["audio_features"]:
        track["top_40"] = 1 if top_40 else 0
        add_one_hot_encodings(track)
    tracks_dataframe = pd.DataFrame.from_records(tracks_contents_json["audio_features"])
    next = playlist_contents_json["tracks"]["next"]
    while next != None:
        playlist_request = urllib2.Request(next, headers=request_headers)
        playlist_contents = urllib2.urlopen(playlist_request)
        playlist_contents_json = json.load(playlist_contents)
        track_id_list = []
        for item in playlist_contents_json["items"]:
            track_id_list.append(item["track"]["id"])
        tracks_request = urllib2.Request(get_track_ids_url(track_id_list), headers=request_headers)
        tracks_contents = urllib2.urlopen(tracks_request)
        tracks_contents_json = json.load(tracks_contents)
        for track in tracks_contents_json["audio_features"]:
            track["top_40"] = 1 if top_40 else 0
            add_one_hot_encodings(track)
        temp_tracks_dataframe = pd.DataFrame.from_records(tracks_contents_json["audio_features"])
        tracks_dataframe = pd.concat([tracks_dataframe, temp_tracks_dataframe])
        print(len(tracks_dataframe))
        next = playlist_contents_json["next"]
    return tracks_dataframe

def preprocess_features(tracks_dataframe):
    track_features = tracks_dataframe[
        [
        "acousticness",
        "danceability",
        "duration_ms",
        "energy",
        "instrumentalness",
        "liveness",
        "loudness",
        "speechiness",
        "tempo",
        "valence",
        "C",
        "C_Sharp",
        "D",
        "D_Sharp",
        "E",
        "F",
        "F_Sharp",
        "G",
        "G_Sharp",
        "A",
        "A_Sharp",
        "B",
        "sig0",
        "sig1",
        "sig2",
        "sig3",
        "sig4",
        "sig5",
        "minor",
        "major"]]
    return track_features

def preprocess_targets(tracks_dataframe):
    track_targets = pd.DataFrame()
    track_targets["top_40"] = tracks_dataframe["top_40"]
    return track_targets

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

# key, mode, and time signature need one hot encoding
def construct_feature_columns():
    key = tf.feature_column.numeric_column("key")
    mode = tf.feature_column.numeric_column("mode")
    time_signature = tf.feature_column.numeric_column("time_signature")

def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a neural network model.

    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """

    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}

    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)

    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)

    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])

def train_nn_regression_model(
    my_optimizer,
    steps,
    batch_size,
    hidden_units,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a neural network regression model.

  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.

  Args:
    my_optimizer: An instance of `tf.train.Optimizer`, the optimizer to use.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    batch_size: A non-zero `int`, the batch size.
    hidden_units: A `list` of int values, specifying the number of neurons in each layer.
    training_examples: A `DataFrame` containing one or more columns from
      `tracks_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `tracks_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `tracks_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `tracks_dataframe` to use as target for validation.

  Returns:
    A tuple `(estimator, training_losses, validation_losses)`:
      estimator: the trained `DNNRegressor` object.
      training_losses: a `list` containing the training loss values taken during training.
      validation_losses: a `list` containing the validation loss values taken during training.
  """

  periods = 10
  steps_per_period = steps / periods

  # Create a DNNRegressor object.
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  dnn_regressor = tf.estimator.DNNRegressor(
      feature_columns=construct_feature_columns(training_examples),
      hidden_units=hidden_units,
      optimizer=my_optimizer
  )

  # Create input functions.
  training_input_fn = lambda: my_input_fn(training_examples,
                                          training_targets["top_40"],
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples,
                                                  training_targets["top_40"],
                                                  num_epochs=1,
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples,
                                                    validation_targets["top_40"],
                                                    num_epochs=1,
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    dnn_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = dnn_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])

    validation_predictions = dnn_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])

    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  print("Final RMSE (on training data):   %0.2f" % training_root_mean_squared_error)
  print("Final RMSE (on validation data): %0.2f" % validation_root_mean_squared_error)

  return dnn_regressor, training_rmse, validation_rmse

try:
    tracks_dataframe = pd.read_pickle("./tracks.pkl")
except IOError:
    tracks_dataframe = getData("https://api.spotify.com/v1/users/mitchell.mason58/playlists/3pQrIDZCas8kSGCugEaNOm", True)
    tracks_dataframe = pd.concat([
                                  tracks_dataframe,
                                  getData("https://api.spotify.com/v1/users/mitchell.mason58/playlists/1pPNuunLNQD2b7CTBtB63e", True),
                                  getData("https://api.spotify.com/v1/users/qaz23/playlists/6MNWLIltmajGap6sxHEdMz", False),
                                  getData("https://api.spotify.com/v1/users/cboyd501/playlists/5zQ7aCDPOhhZbQfeUB8uUz", False),
                                  getData("https://api.spotify.com/v1/users/qaz23/playlists/11PERGmbJOO8moiVaFbOWy", False)])
    tracks_dataframe.to_pickle("./tracks.pkl") # write to file so I don"t bother spotify every time I run the code

tracks_dataframe = tracks_dataframe.drop_duplicates("id", keep=False)
# normalize data
tracks_dataframe["tempo"] = linear_scale(tracks_dataframe["tempo"])
tracks_dataframe["duration_ms"] = linear_scale(tracks_dataframe["duration_ms"])
tracks_dataframe["loudness"] = linear_scale(tracks_dataframe["loudness"])

# get test DataFrame. every 6th row so it is the same each time
test_data = tracks_dataframe.iloc[::6]
tracks_dataframe = pd.concat([test_data, tracks_dataframe], ignore_index=True)
tracks_dataframe = tracks_dataframe.drop_duplicates("id", keep=False)
tracks_dataframe = tracks_dataframe.reindex(
    np.random.permutation(tracks_dataframe.index))

# choose 2800 tracks for training
training_examples = preprocess_features(tracks_dataframe.head(2800))
training_targets = preprocess_targets(tracks_dataframe.head(2800))

# choose remaining for validation
validation_examples = preprocess_features(tracks_dataframe.tail(len(tracks_dataframe) - 2800))
validation_targets = preprocess_targets(tracks_dataframe.tail(len(tracks_dataframe) - 2800))

_ = train_nn_regression_model(
    my_optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.0007),
    steps=5000,
    batch_size=70,
    hidden_units=[10, 10],
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)
