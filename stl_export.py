import struct

import numpy as np


def _rectangle_perimeter(x0, x1, y0, y1, subdivisions):
    """Return evenly subdivided rectangle edges in counter-clockwise order."""
    points = []

    for index in range(subdivisions):
        fraction = index / subdivisions
        points.append((x0 + (x1 - x0) * fraction, y0))

    for index in range(subdivisions):
        fraction = index / subdivisions
        points.append((x1, y0 + (y1 - y0) * fraction))

    for index in range(subdivisions):
        fraction = index / subdivisions
        points.append((x1 - (x1 - x0) * fraction, y1))

    for index in range(subdivisions):
        fraction = index / subdivisions
        points.append((x0, y1 - (y1 - y0) * fraction))

    return np.asarray(points, dtype=float)


def _validate_holes(hole_positions, plate_size, radius):
    half_width = plate_size[0] / 2.0
    half_height = plate_size[1] / 2.0

    outside = (
        np.abs(hole_positions[:, 0]) + radius > half_width + 1e-9
    ) | (
        np.abs(hole_positions[:, 1]) + radius > half_height + 1e-9
    )

    if np.any(outside):
        raise ValueError(
            f"{int(np.count_nonzero(outside))} hole(s) extend beyond the plate"
        )

    # A spatial hash finds overlapping holes without allocating an NxN matrix.
    diameter = radius * 2.0
    occupied_cells = {}

    for index, position in enumerate(hole_positions):
        cell = tuple(np.floor(position / diameter).astype(int))

        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                neighbour = (cell[0] + offset_x, cell[1] + offset_y)

                for other_index in occupied_cells.get(neighbour, ()):
                    separation = np.linalg.norm(
                        position - hole_positions[other_index]
                    )

                    if separation < diameter - 1e-9:
                        raise ValueError(
                            "Hole diameter causes overlapping holes; "
                            "reduce the diameter or increase the spacing"
                        )

        occupied_cells.setdefault(cell, []).append(index)


def _write_binary_triangle(file_handle, vertices):
    vertices = np.asarray(vertices, dtype=np.float32)
    normal = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    length = np.linalg.norm(normal)

    if length > 0:
        normal /= length

    file_handle.write(struct.pack(
        "<12fH",
        *normal,
        *vertices.reshape(-1),
        0,
    ))


def export_pinhole_plate_stl(
        file_path,
        hole_positions_mm,
        plate_size_mm,
        hole_diameter_mm,
        thickness_mm=5.0,
        circle_segments=48,
):
    """Write a watertight binary STL of a rectangular perforated plate.

    The XY origin is at the center of the plate and the plate is centered on
    Z=0. All dimensions are interpreted as millimetres.
    """
    holes = np.asarray(hole_positions_mm, dtype=float)
    plate_size = np.asarray(plate_size_mm, dtype=float)

    if holes.ndim != 2 or holes.shape[1] != 2 or len(holes) == 0:
        raise ValueError("hole_positions_mm must have shape (N, 2) with N > 0")

    if plate_size.shape != (2,) or np.any(plate_size <= 0):
        raise ValueError("plate_size_mm must contain two positive dimensions")

    if hole_diameter_mm <= 0:
        raise ValueError("hole_diameter_mm must be positive")

    if thickness_mm <= 0:
        raise ValueError("thickness_mm must be positive")

    if circle_segments < 12 or circle_segments % 4 != 0:
        raise ValueError("circle_segments must be a multiple of 4 and at least 12")

    # Projection math can leave insignificant floating-point differences among
    # coordinates that belong to the same mask row or column.
    holes = np.round(holes, decimals=9)
    radius = hole_diameter_mm / 2.0
    _validate_holes(holes, plate_size, radius)

    x_coordinates = np.unique(holes[:, 0])
    y_coordinates = np.unique(holes[:, 1])
    half_width, half_height = plate_size / 2.0

    x_boundaries = np.concatenate((
        [-half_width],
        (x_coordinates[:-1] + x_coordinates[1:]) / 2.0,
        [half_width],
    ))
    y_boundaries = np.concatenate((
        [-half_height],
        (y_coordinates[:-1] + y_coordinates[1:]) / 2.0,
        [half_height],
    ))

    x_indices = {coordinate: index for index, coordinate in enumerate(x_coordinates)}
    y_indices = {coordinate: index for index, coordinate in enumerate(y_coordinates)}

    for x_position, y_position in holes:
        x_index = x_indices[x_position]
        y_index = y_indices[y_position]
        clearance = min(
            x_position - x_boundaries[x_index],
            x_boundaries[x_index + 1] - x_position,
            y_position - y_boundaries[y_index],
            y_boundaries[y_index + 1] - y_position,
        )

        if radius >= clearance - 1e-9:
            raise ValueError(
                "Hole diameter is too large for the spacing between mask "
                "rows or columns, or a hole is tangent to the plate edge"
            )

    hole_lookup = {tuple(position) for position in holes}
    subdivisions = circle_segments // 4
    bottom_z = -thickness_mm / 2.0
    top_z = thickness_mm / 2.0
    triangle_count = 0

    with open(file_path, "wb") as stl_file:
        header = b"Pinhole plate; dimensions in millimetres"
        stl_file.write(header.ljust(80, b"\0"))
        stl_file.write(struct.pack("<I", 0))

        def add_triangle(point_a, point_b, point_c):
            nonlocal triangle_count
            _write_binary_triangle(stl_file, (point_a, point_b, point_c))
            triangle_count += 1

        for y_index, y_position in enumerate(y_coordinates):
            y0 = y_boundaries[y_index]
            y1 = y_boundaries[y_index + 1]

            for x_index, x_position in enumerate(x_coordinates):
                x0 = x_boundaries[x_index]
                x1 = x_boundaries[x_index + 1]
                perimeter = _rectangle_perimeter(
                    x0, x1, y0, y1, subdivisions
                )
                has_hole = (x_position, y_position) in hole_lookup

                if has_hole:
                    directions = perimeter - (x_position, y_position)
                    directions /= np.linalg.norm(
                        directions,
                        axis=1,
                        keepdims=True,
                    )
                    inner = (x_position, y_position) + radius * directions

                for point_index in range(circle_segments):
                    next_index = (point_index + 1) % circle_segments
                    outer_a = perimeter[point_index]
                    outer_b = perimeter[next_index]

                    if has_hole:
                        inner_a = inner[point_index]
                        inner_b = inner[next_index]

                        add_triangle(
                            (*outer_a, top_z),
                            (*outer_b, top_z),
                            (*inner_b, top_z),
                        )
                        add_triangle(
                            (*outer_a, top_z),
                            (*inner_b, top_z),
                            (*inner_a, top_z),
                        )
                        add_triangle(
                            (*outer_a, bottom_z),
                            (*inner_b, bottom_z),
                            (*outer_b, bottom_z),
                        )
                        add_triangle(
                            (*outer_a, bottom_z),
                            (*inner_a, bottom_z),
                            (*inner_b, bottom_z),
                        )
                        add_triangle(
                            (*inner_a, bottom_z),
                            (*inner_b, top_z),
                            (*inner_b, bottom_z),
                        )
                        add_triangle(
                            (*inner_a, bottom_z),
                            (*inner_a, top_z),
                            (*inner_b, top_z),
                        )
                    else:
                        center = (x_position, y_position)
                        add_triangle(
                            (*center, top_z),
                            (*outer_a, top_z),
                            (*outer_b, top_z),
                        )
                        add_triangle(
                            (*center, bottom_z),
                            (*outer_b, bottom_z),
                            (*outer_a, bottom_z),
                        )

                    edge_group = point_index // subdivisions
                    on_outer_edge = (
                        (y_index == 0 and edge_group == 0)
                        or (x_index == len(x_coordinates) - 1 and edge_group == 1)
                        or (y_index == len(y_coordinates) - 1 and edge_group == 2)
                        or (x_index == 0 and edge_group == 3)
                    )

                    if on_outer_edge:
                        add_triangle(
                            (*outer_a, bottom_z),
                            (*outer_b, bottom_z),
                            (*outer_b, top_z),
                        )
                        add_triangle(
                            (*outer_a, bottom_z),
                            (*outer_b, top_z),
                            (*outer_a, top_z),
                        )

        stl_file.seek(80)
        stl_file.write(struct.pack("<I", triangle_count))

    return triangle_count
