"""Retrieve data about nontrending and trending videos from YouTube api and write to numpy files"""
import requests
import json
import urllib
import time
from tqdm import tqdm
import pandas as pd
import numpy as np

# define number of video ids to be grabbed in api request at a time
group_size = 50

def get_api_key(filename):
    """Get api key from file"""
    api_key_file = open(filename, 'r')
    return api_key_file.read().rstrip()

def get_list_group_ids(filename):
    """
    Get video ids from csv and put them in groups of 50 so we can request data
    for 50 videos from the YouTube api per api request
    """
    # load dataframe from csv
    df = pd.read_csv(filename, error_bad_lines=False)
    # drop null values
    df = df.dropna(axis=0, how='any')
    # purely get 1st column
    video_ids = df.iloc[:, 1]
    # convert to 1d array
    video_ids = video_ids.ravel()

    individual_str = ""
    list_groups_of_ids = []
    i = 0
    for video_id in video_ids:
        if (i == 0):
            individual_str += video_id
            i += 1
        else:
            if (i < group_size):
                i += 1
                individual_str += ',' + video_id
            else:
                list_groups_of_ids.append(individual_str)
                individual_str = ""
                i = 0
    return list_groups_of_ids

class VideoDetails:
    """
    VideoDetails class used to accumulate data retrieved from api. Defined class
    so we can create accumulate data separately for trending and nontrending videos
    """
    def __init__(self):
        self.video_id = []
        self.video_title = []
        self.channel_title = []
        self.view_count = []
        self.like_count = []
        self.dislike_count = []
        self.favorite_count = []
        self.comment_count = []
        self.published_at = []
        self.category_id = []
        self.content_duration = []
        self.tags = []

def get_video_details(video_id, details_class, api_key):
    """Hit api and accumulate data for given class"""
    try:
        searchUrl="https://www.googleapis.com/youtube/v3/videos?id="+video_id+"&key="+api_key+"&part=statistics,snippet,content_details"
        response = urllib.request.urlopen(searchUrl).read()
        data = json.loads(response)
        for i in range(0, group_size):
            details_class.video_id.append(data['items'][i]['id'])
            details_class.video_title.append(data['items'][i]['snippet']['title'])
            details_class.channel_title.append(data['items'][i]['snippet']['channelTitle'])
            details_class.view_count.append(data['items'][i]['statistics']['viewCount'])
            details_class.like_count.append(data['items'][i]['statistics']['likeCount'])
            details_class.dislike_count.append(data['items'][i]['statistics']['dislikeCount'])
            details_class.favorite_count.append(data['items'][i]['statistics']['favoriteCount'])
            details_class.comment_count.append(data['items'][i]['statistics']['commentCount'])
            details_class.content_duration.append(data['items'][i]['contentDetails']['duration'])
            details_class.published_at.append(data['items'][i]['snippet']['publishedAt'])
            details_class.category_id.append(data['items'][i]['snippet']['categoryId'])
            try:
                details_class.tags.append(data['items'][i]['snippet']['tags'])
            except (IndexError, KeyError):
                details_class.tags.append([])
    except (IndexError, KeyError):
        return

def details_object_to_array(detailsObject):
    """Return a list of rows of data we care about"""
    return list(zip(detailsObject.view_count, detailsObject.like_count,
        detailsObject.dislike_count, detailsObject.comment_count))

def main():
    """Get data from nontrending and trending videos and write to numpy files"""
    start_time = time.time()
    api_key = get_api_key('api_key.txt')

    list_groups_of_trending_ids = get_list_group_ids('trending-yt.csv')
    list_groups_of_nontrending_ids = get_list_group_ids('nontrending-yt.csv')
    # initialize trending and nontrending details objects
    nontrending_details = VideoDetails()
    trending_details = VideoDetails()
    # modify 2nd parameter of slice to specify number of groups of 50 to query
    slice_trending = slice(0, len(list_groups_of_trending_ids))
    slice_nontrending = slice(0, len(list_groups_of_nontrending_ids))

    # get video stats for each list of videos
    print("Getting nontrending video data...")
    for group in tqdm(list_groups_of_nontrending_ids[slice_nontrending]):
        get_video_details(group, nontrending_details, api_key)
    print("Getting trending video data...")
    for group in tqdm(list_groups_of_trending_ids[slice_trending]):
        get_video_details(group, trending_details, api_key)

    # convert class objects to tuples
    nontrending_details = details_object_to_array(nontrending_details)
    trending_details = details_object_to_array(trending_details)

    print("\n\nNontrending video details \n ------------------------- \n")
    for details in nontrending_details:
        print(details)

    print("\n\nTrending video details \n ------------------------- \n")
    for details in trending_details:
        print(details)

    nontrending_stats_mat = np.array(nontrending_details).astype(np.float)
    trending_stats_mat = np.array(trending_details).astype(np.float)
    np.save('nontrending_stats', nontrending_stats_mat)
    np.save('trending_stats', trending_stats_mat)
    print("Took " + str(time.time() - start_time) + " seconds.")



main()
