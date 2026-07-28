import math

import numpy as np


POINTS_PER_MM = 72.0 / 25.4


def _format_number(value):
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _pdf_object(object_number, contents):
    return (
        f"{object_number} 0 obj\n".encode("ascii")
        + contents
        + b"\nendobj\n"
    )


def export_pinhole_pattern_pdf(
        file_path,
        hole_positions_mm,
        canvas_size_mm,
        hole_diameter_mm,
):
    """Write a one-page, physically scaled PDF containing black hole circles.

    Hole positions use a center-origin XY coordinate system in millimetres.
    The PDF page is exactly ``canvas_size_mm`` and contains no margins or
    other marks.
    """
    holes = np.asarray(hole_positions_mm, dtype=float)
    canvas_size = np.asarray(canvas_size_mm, dtype=float)

    if holes.ndim != 2 or holes.shape[1] != 2 or len(holes) == 0:
        raise ValueError("hole_positions_mm must have shape (N, 2) with N > 0")

    if not np.all(np.isfinite(holes)):
        raise ValueError("hole_positions_mm must contain only finite values")

    if (
            canvas_size.shape != (2,)
            or not np.all(np.isfinite(canvas_size))
            or np.any(canvas_size <= 0)
    ):
        raise ValueError("canvas_size_mm must contain two positive dimensions")

    if not math.isfinite(hole_diameter_mm) or hole_diameter_mm <= 0:
        raise ValueError("hole_diameter_mm must be positive")

    radius_mm = hole_diameter_mm / 2.0
    half_size = canvas_size / 2.0
    outside = np.any(np.abs(holes) + radius_mm > half_size + 1e-9, axis=1)

    if np.any(outside):
        raise ValueError(
            f"{int(np.count_nonzero(outside))} hole(s) extend beyond the canvas"
        )

    width_points, height_points = canvas_size * POINTS_PER_MM
    radius_points = radius_mm * POINTS_PER_MM
    # Four cubic Bezier curves are the standard PDF representation of a circle.
    control_offset = radius_points * 0.5522847498307936
    commands = ["0 g"]

    for x_mm, y_mm in holes:
        center_x = (x_mm + half_size[0]) * POINTS_PER_MM
        center_y = (y_mm + half_size[1]) * POINTS_PER_MM
        x0 = center_x - radius_points
        x1 = center_x + radius_points
        y0 = center_y - radius_points
        y1 = center_y + radius_points
        c = control_offset
        f = _format_number

        commands.extend((
            f"{f(center_x)} {f(y1)} m",
            f"{f(center_x + c)} {f(y1)} {f(x1)} {f(center_y + c)} "
            f"{f(x1)} {f(center_y)} c",
            f"{f(x1)} {f(center_y - c)} {f(center_x + c)} {f(y0)} "
            f"{f(center_x)} {f(y0)} c",
            f"{f(center_x - c)} {f(y0)} {f(x0)} {f(center_y - c)} "
            f"{f(x0)} {f(center_y)} c",
            f"{f(x0)} {f(center_y + c)} {f(center_x - c)} {f(y1)} "
            f"{f(center_x)} {f(y1)} c",
            "f",
        ))

    stream = ("\n".join(commands) + "\n").encode("ascii")
    media_box = (
        f"[0 0 {_format_number(width_points)} "
        f"{_format_number(height_points)}]"
    )
    objects = (
        _pdf_object(1, b"<< /Type /Catalog /Pages 2 0 R >>"),
        _pdf_object(
            2,
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        ),
        _pdf_object(
            3,
            (
                f"<< /Type /Page /Parent 2 0 R /MediaBox {media_box} "
                "/Resources << >> /Contents 4 0 R >>"
            ).encode("ascii"),
        ),
        _pdf_object(
            4,
            f"<< /Length {len(stream)} >>\nstream\n".encode("ascii")
            + stream
            + b"endstream",
        ),
    )

    pdf = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]

    for pdf_object in objects:
        offsets.append(len(pdf))
        pdf.extend(pdf_object)

    xref_offset = len(pdf)
    pdf.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")

    for offset in offsets[1:]:
        pdf.extend(f"{offset:010d} 00000 n \n".encode("ascii"))

    pdf.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode("ascii")
    )

    with open(file_path, "wb") as pdf_file:
        pdf_file.write(pdf)

    return len(holes)
