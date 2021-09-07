from typing import Dict, Union, List

from pup.common.checkin import Checkin

UserId = Union[int, str]     # User ID can be `int` or `str`
CheckinId = Union[int, str]  # Checkin ID can be `int` or `str`

CheckinList = List[Checkin]
CheckinDataset = Dict[UserId, Dict[CheckinId, Checkin]]

