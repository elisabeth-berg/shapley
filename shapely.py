import numpy as np
import pandas as pd

class ShapelyAttributer(object):

    def __init__(self, journeys, values):
        """
        Compute the Shapely channel value for each of the channels {0, ..., n}.

        Future work
            -- incorporate repeated visits
            -- incorporate touchpoint ordering
            -- sort journeys by win/not
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
        Compute shapely values for each channel.
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
        Compute shapely values for each channel/time segment combo.
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

    def score_user(self, user_journey):
        """
        For now, assume we care about the order (i.e., use ordered_shapely_...)
        user_journey : array-like, sequence of the user's channel touches
        """
        score_matrix = self.ordered_shapely_proportions
        self.score = 0
        for time, channel in enumerate(user_journey):
            self.score += score_matrix[time, channel]
        return self.score


#journeys = pd.read_csv('data/journeys.csv', header=None)
#journeys.columns = ['t' + str(c) for c in journeys.columns]
#values = np.random.rand(journeys.shape[0], 1)
