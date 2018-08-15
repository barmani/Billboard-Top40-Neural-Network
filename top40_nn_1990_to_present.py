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
pd.options.display.float_format = '{:.1f}'.format

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
    "Authorization": "Bearer BQADSjEWAsXEvr_Zfhzh2uZfzMdYVSuDYAIq8oblv0wW63W_sdmALBAv5PTX_6SOEh4UHGlNU3UqFflNFwXskbVhkytlGD9e051Tc7n-2UZXBt8KyAGTTTf2mSTD3pKEgkOQOi5ygpvV"
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
        track["top_40"] = top_40
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
            track["top_40"] = top_40
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
        "key",
        "liveness",
        "loudness",
        "mode",
        "speechiness",
        "tempo",
        "time_signature",
        "valence"]]
    return track_features

def preprocess_targets(tracks_dataframe):
    track_targets = pd.DataFrame()
    track_targets["top_40"] = tracks_dataframe["top_40"]

def linear_scale(series):
  min_val = series.min()
  max_val = series.max()
  scale = (max_val - min_val) / 2.0
  return series.apply(lambda x:((x - min_val) / scale) - 1.0)

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
    tracks_dataframe.to_pickle("./tracks.pkl") # write to file so I don't bother spotify every time I run the code

# get test DataFrame. every 6th row so it is the same each time
test_data = tracks_dataframe.iloc[::6]
tracks_dataframe = tracks_dataframe[~tracks_dataframe.id.isin(test_data.id)]
tracks_dataframe = tracks_dataframe.reindex(
    np.random.permutation(tracks_dataframe.index))

# choose 2800 tracks for training
training_examples = preprocess_features(track_dataframe.head(2800))
training_targets = preprocess_targets(track_dataframe.head(2800))

# choose remaining for validation
validation_examples = preprocess_features(track_dataframe.tail(len(track_dataframe) - 2800))
validation_targets = preprocess_targets(track_dataframe.tail(len(track_dataframe) - 2800))
