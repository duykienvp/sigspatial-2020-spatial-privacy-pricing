class Checkin(object):
    """ Object to store and manipulate a checkin

    Note about time: Datetime field is converted directly from raw time string.
    If there is no timestamp available in raw data, timestamp field is extracted from datetime field.
    If there is timestamp available in raw data, timestamp field is converted directly from raw data.
    Thus, timestamp field may not be the same with timestamp extracted from datetime field due to timezone difference

    Attributes
    ----------
    user_id: int, or str
        user id
    c_id: int
        id of check-in
    timestamp: int
        timestamp
    datetime: datetime
        datetime
    lat: float
        latitude
    lon: float
        longitude
    location_id: int, or str
        id of the check-in location
    x: float
        x in (x, y) coordinates
    y: float
        y in (x, y) coordinates
    """
    c_id = None
    user_id = None
    timestamp = None
    datetime = None
    lat = None
    lon = None
    location_id = None
    x = None  # converted to Oxy coordinate
    y = None  # converted to Oxy coordinate

    user_privacy_value = None  # privacy level of the user of this data point
    sensitivity = None  # sensitivity of this data point to the user
    combined_privacy_value = None  # combined privacy value

    def __init__(self, c_id, user_id, timestamp, datetime, lat, lon, location_id):
        """
        Initialize a checkin with given values from datasets

        Parameters
        ----------
        c_id: int, or str
            id of check-in
        user_id: int, or str
            user id
        timestamp: int
            timestamp
        datetime: datetime
            datetime
        lat: float
            latitude
        lon: float
            longitude
        location_id: int, or str
            id of the check-in location
        """
        self.c_id = c_id
        self.user_id = user_id
        self.timestamp = timestamp
        self.datetime = datetime
        self.lat = lat
        self.lon = lon
        self.location_id = location_id

    def __str__(self) -> str:
        return "Checkin(c_id={c_id}, user_id={user_id}, timestamp={timestamp}, datetime={datetime}, " \
               "lat={lat}, lon={lon}, location_id={location_id}, x={x}, y={y}, " \
               "user_privacy_value={user_privacy_value}, sensitivity={sensitivity}, " \
               "combined_privacy_value={combined_privacy_value})".format(**vars(self))





