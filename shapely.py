import numpy as np
import pandas as pd

class ShapelyAttributer(object):

    def __init__(self, journeys, values):
        """
        Compute the Shapely channel value for each of the channels {0, ..., n}.

        Parameters
        ----------
        journeys :  dataframe. Each column represents a sequence number; each
                    row is a user. The values should be integers corresponding
                    to the channel touched at this point in the sequence.
        values :    vector of values corresponding to the users in journeys.
        """
        self.journeys = journeys
        self.values = values
        (self.n_users, self.max_journey) = self.journeys.shape
        self.n_channels = int(np.max(np.max(self.journeys)))

        # Compute a weight column
        self.journeys['weight'] = 1 / (self.journeys.count(axis=1) + 1)
        self.journeys['values'] = self.values

        if not len(self.values) == self.n_users:
            raise Exception("journeys and values must be the same length.")

    def fit(self):
        """
        Compute Shapely values for each channel.
        Note that this method is order agnostic and also ignores double touches.
        """
        self.shapely_vals = np.zeros(self.n_channels + 1)
        for j in range(self.n_channels + 1):
            # Find all journeys / coalitions containing channel j
            coalition_idx = (self.journeys == j).any(axis=1)
            self.shapely_vals[j] = sum(
                self.journeys['values'][coalition_idx] / self.journeys['weight'][coalition_idx])
        self.shapely_proportions = self.shapely_vals / sum(self.shapely_vals)

    def fit_ordered(self):
        """
        Compute Shapely values for each channel/time segment combo.
        Note that double touches are double counted.
        """
        self.ordered_shapely_vals = np.zeros((self.max_journey, self.n_channels + 1))
        # Iterate through channels
        for j in range(self.n_channels + 1):
            # Iterate through touchpoints
            for i in range(self.max_journey):
                time_column = "t" + str(i)
                coalition_idx = self.journeys[time_column] == j
                self.ordered_shapely_vals[i, j] = sum(
                    self.journeys['values'][coalition_idx] / self.journeys['weight'][coalition_idx])
        self.ordered_shapely_proportions = self.ordered_shapely_vals / self.ordered_shapely_vals.sum()

    def score_user(self, user_journey, ordered=True):
        """
        user_journey : array-like, sequence of the user's channel touches
        """
        if ordered:
            score_matrix = self.ordered_shapely_proportions
            self.score = 0
            # ordered_shapely_proportions measure the effectiveness of a time/
            # channel combo. Sum these values up for the user's journey.
            for time, channel in enumerate(user_journey):
                self.score += score_matrix[time, channel]
        else:
            score_matrix = self.shapely_proportions
            # If the user touches all channels, this will just be 1
            self.score = sum(set(SA.shapely_proportions[user_journey]))
        return self.score


# journeys = pd.read_csv('data/journeys.csv', header=None)
# journeys.columns = ['t' + str(c) for c in journeys.columns]
# values = np.random.rand(journeys.shape[0], 1)
