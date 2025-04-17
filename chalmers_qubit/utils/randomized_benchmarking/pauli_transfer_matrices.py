import numpy as np
from typing import Literal

__all__ = [
    "I",
    "X",
    "Y",
    "Z",
    "S",
    "S2",
    "H",
    "CZ",
    "X_theta",
    "Y_theta",
]

I = np.eye(4)

# Pauli group
X = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]], dtype=int)
Y = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=int)
Z = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]], dtype=int)

# Exchange group
S = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]], dtype=int)
S2 = np.dot(S, S)

# Hadamard group
H = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0], [0, 1, 0, 0]], dtype=int)

# CZ
CZ = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=int,
)

def X_theta(theta: float, unit: Literal["deg", "rad"] = "deg") -> np.ndarray:
    """
    Return the Pauli Transfer Matrix (PTM) of a rotation of angle theta
    around the X-axis.

    Args:
        theta (float): Rotation angle.
        unit (str): Unit of the angle, either "deg" for degrees or "rad" for radians.

    Returns:
        np.ndarray: The 4x4 PTM matrix corresponding to the X rotation.
    """
    if unit == "deg":
        theta = np.deg2rad(theta)
    elif unit != "rad":
        raise ValueError(f"Unsupported unit '{unit}'. Use 'deg' or 'rad'.")
    
    cos = np.cos(theta)
    sin = np.sin(theta)

    X = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, cos, -sin],
            [0, 0, sin, cos],
        ],
        dtype=np.float64,
    )
    return X


def Y_theta(theta: float, unit: Literal["deg", "rad"] = "deg") -> np.ndarray:
    """
    Return the Pauli Transfer Matrix (PTM) of a rotation of angle theta
    around the Y-axis.

    Args:
        theta (float): Rotation angle.
        unit (str): Unit of the angle, either "deg" for degrees or "rad" for radians.

    Returns:
        np.ndarray: The 4x4 PTM matrix corresponding to the Y rotation.
    """
    if unit == "deg":
        theta = np.deg2rad(theta)
    elif unit != "rad":
        raise ValueError(f"Unsupported unit '{unit}'. Use 'deg' or 'rad'.")

    cos = np.cos(theta)
    sin = np.sin(theta)

    Y = np.array(
        [
            [1, 0,    0,   0],
            [0, cos,  0,  sin],
            [0, 0,    1,   0],
            [0, -sin, 0,  cos],
        ],
        dtype=np.float64,
    )
    return Y