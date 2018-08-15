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
        url += id + ","
    return url[:-1]

def getData():
    """ Build the pandas dataset of track information by gathering dta from the
        Spotify API

      Returns:
        A dataset of tracks with their information
    """
    # key is going to change frequently because it expires -- hopefully will find a workaround
    request_headers = {
    "Accept": "application/json",
    "Content-Type": "application/json",
    "Authorization": "Bearer BQBddbt6YM2VMnnK6J5GbpFlF-hjF0sXxBn-ldjamrNKoYOZS3oLjMV_axjT1h0ooT5KRFUB6_Eiv5Id5YNgRRjn5bgpD_OiIomG2MwlCyBwVjrv9SeF5MTOdfhYc5xBX3m-_GmCRpHR"
    }
    playlist_request = urllib2.Request("https://api.spotify.com/v1/users/mitchell.mason58/playlists/3pQrIDZCas8kSGCugEaNOm", headers=request_headers)
    playlist_contents = urllib2.urlopen(playlist_request)
    playlist_contents_json = json.load(playlist_contents)
    track_id_list = []
    for item in playlist_contents_json["tracks"]["items"]:
        track_id_list.append(item["track"]["id"])
    tracks_request = urllib2.Request(get_track_ids_url(track_id_list), headers=request_headers)
    tracks_contents = urllib2.urlopen(tracks_request)
    tracks_contents_json = json.load(tracks_contents)
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
        pp.pprint(tracks_contents_json["audio_features"])
        temp_tracks_dataframe = pd.DataFrame.from_records(tracks_contents_json["audio_features"])
        tracks_dataframe = pd.concat([tracks_dataframe, temp_tracks_dataframe])
        print(len(tracks_dataframe))
        next = playlist_contents_json["next"]

getData()
