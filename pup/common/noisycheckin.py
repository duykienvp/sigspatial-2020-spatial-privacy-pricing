import numpy
from scipy.stats import rv_continuous

from pup.algorithms import util
from pup.common.checkin import Checkin
from pup.common.grid import Grid
from pup.common.rectangle import Rectangle


class NoisyCheckin(Checkin):
    """ Noisy version of a check-in. The exact location information is removed.

    The exact location information include: lat, lon, location_id, x, y

    Attributes
    ----------
    noise_level: float
        noise level used for this noisy check-in
    rv_x: rv_continuous
        continuous random variable representing the distribution of this noisy data over x dimension
    rv_y: rv_continuous
        continuous random variable representing the distribution of this noisy data over y dimension
    """

    def __init__(self, c: Checkin, noise_level: float, rv_x: rv_continuous, rv_y: rv_continuous):
        """ Initialize a checkin with given values from datasets.

        Parameters
        ----------
        c: Checkin
            the check-in to inherit data from
        noise_level: float
            noise level used for this noisy check-in
        rv_x: rv_continuous
            continuous random variable representing the distribution of this noisy data over x dimension
        rv_y: rv_continuous
            continuous random variable representing the distribution of this noisy data over y dimension
        """
        super().__init__(c.c_id, c.user_id, c.timestamp, c.datetime, c.lat, c.lon, c.location_id)

        # exact location information is removed
        self.lat = None
        self.lon = None
        self.location_id = None
        self.x = None
        self.y = None

        # noise information
        self.rv_x = rv_x
        self.rv_y = rv_y
        self.noise_level = noise_level

    def __str__(self):
        return "Checkin(user_id={user_id}, timestamp={timestamp}, datetime={datetime}, " \
               "lat={lat}, lon={lon}, location_id={location_id}, x={x}, y={y}, " \
               "rv_x={rv_x}, rv_y={rv_y}, noise_level={noise_level})".format(**vars(self))

    def cal_prob_inside_rect(self, rect: Rectangle) -> float:
        """ Calculate the probability that the original check-in of this noisy check-in is inside a rectangle

        Parameters
        ----------
        rect
            rectangle of interest

        Returns
        -------
        float
            the probability that the original check-in of this noisy check-in is inside a rectangle
        """
        return util.cal_prob_inside_rect(rect, self.rv_x, self.rv_y)

    def cal_prob_grid(self, grid: Grid) -> numpy.ndarray:
        """ Calculate probability of being inside each of cell of the grid

        Parameters
        ----------
        grid
            the grid

        Returns
        -------
        ndarray
            the array of probabilities for each grid cell
        """
        max_x_idx, max_y_idx = grid.get_shape()

        # calculate cdf for each line in x dimension
        x_cdf = list()
        for x in range(max_x_idx + 1):
            cell_x = x * grid.cell_len_x + grid.min_x
            x_cdf.append(self.rv_x.cdf(cell_x))

        # calculate cdf for each line in y dimension
        y_cdf = list()
        for y in range(max_y_idx + 1):
            cell_y = y * grid.cell_len_y + grid.min_y
            y_cdf.append(self.rv_y.cdf(cell_y))

        # calculate the probability for each cell
        probs = numpy.zeros((max_x_idx, max_y_idx))

        # prob_inside_domain = self.cal_prob_inside_rect(grid)
        for x in range(max_x_idx):
            for y in range(max_y_idx):
                prob_x = x_cdf[x] - x_cdf[x + 1]
                prob_y = y_cdf[y] - y_cdf[y + 1]

                prob = prob_x * prob_y

                # probs[x, y] = prob / prob_inside_domain
                probs[x, y] = prob

        return probs
