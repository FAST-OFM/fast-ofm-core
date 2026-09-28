"""Neutral stitching-job orchestration across a replaceable worker boundary."""

from .service import StitchingServiceError, run_stitching_payload

__all__ = ["StitchingServiceError", "run_stitching_payload"]
