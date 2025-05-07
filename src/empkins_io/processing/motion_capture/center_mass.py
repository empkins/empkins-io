from copy import deepcopy
from typing import Any, Dict, Optional, Sequence

from empkins_io.processing.motion_capture._base import _BaseMotionCaptureProcessor
from empkins_io.sensors.motion_capture.motion_capture_formats._base_format import _BaseMotionCaptureDataFormat
from empkins_io.sensors.motion_capture.motion_capture_formats.center_mass import CenterOfMassData


class CenterOfMassProcessor(_BaseMotionCaptureProcessor):
    def __init__(self, data: CenterOfMassData):
        assert isinstance(data, CenterOfMassData)
        super().__init__(data)

    def filter_position_drift(
        self, key: str, filter_params: Optional[Dict[str, Any]] = None
    ) -> _BaseMotionCaptureDataFormat:
        """Filter positional displacement drift in center-of-mass data.

        Parameters
        ----------
        Wn : float, optional
            Wn parameter of filter passed to :func:`scipy.signals.butter`.

        Returns
        -------
        :class:`~empkins_io.sensors.motion_capture.motion_capture_formats.center_mass.CenterOfMassData`
            ``CenterOfMassData`` instance with data corrected for positional displacement drift

        """
        com_data = deepcopy(self.data_dict[key])
        data = com_data.data

        data_filt = self._filter_position_drift(data, filter_params.get("Wn", 0.01))
        # data_filt = data_filt.add(data.iloc[0, :])

        com_data.data = data_filt
        return com_data

    def filter_rotation_drift(
        self, key: str, filter_params: Optional[Sequence[Dict[str, Any]]] = None
    ) -> _BaseMotionCaptureDataFormat:
        raise NotImplementedError("Rotation drift filtering not applicable for CenterOfMassData!")
