import numpy as np
from abc import ABC, abstractmethod


class GeometryRepresentation(ABC):
    domain = "geometry"

    @abstractmethod
    def geometry_representation_block(self) -> dict:
        pass

    @abstractmethod
    def to_dict(self) -> dict:
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, data: dict):
        pass


class Profile2DGeometry(GeometryRepresentation):
    representation = "profile"
    dimensionality = "2d"
    is_mesh = False

    def __init__(
        self,
        *,
        x1: np.ndarray,
        x2: np.ndarray,
        bounds: dict,
        sampling_metadata: dict,
        extrusion_thickness: float,
    ):
        if x1.shape != x2.shape:
            raise ValueError("x1 and x2 arrays must have the same shape")

        self.x1 = x1
        self.x2 = x2
        self.bounds = bounds
        self.sampling_metadata = sampling_metadata
        self.extrusion_thickness = extrusion_thickness

    def geometry_representation_block(self) -> dict:
        return {
            "domain": "geometry",
            "representation": "profile",
            "dimensionality": "2d",
            "is_mesh": False,
            "description": (
                "Frozen 2D profile intended for extrusion; "
                "this file does not contain a 3D mesh."
            ),
        }

class EmptyGeometry(GeometryRepresentation):
    domain = "geometry"

    def geometry_representation_block(self) -> dict:
        return {
            "domain": "geometry",
            "representation": "empty",
            "dimensionality": "",
            "is_mesh": False,
            "description": (
                "An empty geometry. "
                "The smartwedge geometry is fully described by the wedge parameters."
            ),
        }

    def to_dict(self) -> dict:
        return {"type": "empty",}

    @classmethod
    def from_dict(cls, data: dict):
        return cls()