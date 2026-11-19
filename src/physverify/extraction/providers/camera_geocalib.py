"""GeoCalib: focal length and gravity direction estimated from the clip's frames."""

from physverify.extraction.providers._geocalib import GeoCalibCamera

PLUGINS = (GeoCalibCamera,)
