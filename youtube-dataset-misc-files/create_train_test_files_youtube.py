"""
Creates youtube.train and youtube.test files from YouTube api data
stored in numpy arrays
(Input Attributes: view count, like count, dislike count, comment count)
(Output Attributes: 0 or 1 to represent trending or nontrending)
"""
import numpy as np

def normalize(d):
    """Normalize matrix values by dividing by max per column"""
    d /= np.max(d, axis=0)
    return d

def standardize(a):
    """Make data 0 mean with standard deviation equal to 1"""
    d = np.array(a)
    d -= np.mean(d, axis=0)
    d /= np.std(d, axis=0)
    return np.array(d)

def main():
    """
    Take data from .npy files and format to be fit for .train and .test used in
    neural network training and testing programs.
    """
    nontrending_stats_non = np.load('nontrending_stats.npy')
    trending_stats_non = np.load('trending_stats.npy')
    # Combine trending and nontrending data, then standardize them
    all_data = np.concatenate((trending_stats_non[0:950,:], nontrending_stats_non[0:950]), axis=0)
    standardized_data = standardize(all_data)
    # Split em up
    trending_stats = standardized_data[0:950,:]
    nontrending_stats = standardized_data[951:,:]
    # Curate sets of data necessary for neural net
    train_data = np.concatenate((trending_stats[0:400,:], nontrending_stats[0:400,:]), axis = 0)
    test_data = np.concatenate((trending_stats[401:801,:], nontrending_stats[401:801,:]), axis = 0)
    train_labels = np.concatenate((np.ones((400,1)), np.zeros((400,1))), axis = 0)
    test_labels = np.concatenate((np.ones((400,1)), np.zeros((400,1))), axis = 0)

    training_set = np.append(train_data, train_labels, axis=1)
    testing_set = np.append(test_data, test_labels, axis=1)

    np.random.shuffle(training_set)
    np.random.shuffle(testing_set)

    file = open("youtube.test", "w+")
    file.write("800 4 1\n")
    for row in testing_set:
        np.savetxt(file, row[:4], newline=" ", fmt="%1.3f")
        np.savetxt(file, row[4:], fmt="%1i")
    file.close()

    file = open("youtube.train", "w+")
    file.write("800 4 1\n")
    for row in training_set:
        np.savetxt(file, row[:4], newline=" ", fmt="%1.3f")
        np.savetxt(file, row[4:], fmt="%1i")
    file.close()



main()
