"""Open-vocabulary boxes from the clip's object queries with Grounding DINO tiny."""

from physverify.extraction.providers._grounding_dino import GroundingDinoDetector

PLUGINS = (GroundingDinoDetector,)
