import numpy as np

from text_mask import create_text_mask

# -----------------------------
# Geometry
# -----------------------------

def normalize(v):
    return v / np.linalg.norm(v)

def plane_basis(normal, preferred_up=np.array([0.0, 1.0, 0.0])):
    """
    Construct stable horizontal and vertical unit vectors in a plane.
    """
    normal = normalize(np.asarray(normal, dtype=float))
    preferred_up = normalize(np.asarray(preferred_up, dtype=float))

    if abs(np.dot(normal, preferred_up)) > 0.99:
        preferred_up = np.array([0.0, 0.0, 1.0])

    axis_x = normalize(np.cross(preferred_up, normal))
    axis_y = normalize(np.cross(normal, axis_x))

    return axis_x, axis_y

def intersect_plane(
        ray_origin,
        ray_direction,
        plane_origin,
        plane_normal
):

    denom = np.dot(
        ray_direction,
        plane_normal
    )

    if abs(denom) < 1e-8:
        return None

    t = np.dot(
        plane_origin - ray_origin,
        plane_normal
    ) / denom

    if t < 0:
        return None

    return ray_origin + t * ray_direction



def create_crescent_kernel(
        sun_diameter_mm,
        phase,
        output_px_per_mm,
        supersample=4
):
    """
    Create an ideal eclipsed-Sun image before aperture blur.

    phase:
        1.0 -> unobscured Sun
        near 0.0 -> thin visible crescent
    """

    phase = float(np.clip(phase, 0.0, 1.0))

    diameter_px = max(
        3,
        int(np.ceil(
            sun_diameter_mm * output_px_per_mm
        ))
    )

    # Odd dimensions give the kernel a well-defined central pixel.
    if diameter_px % 2 == 0:
        diameter_px += 1

    sample_size = diameter_px * supersample

    coordinates = (
        np.arange(sample_size) + 0.5
    ) / supersample - diameter_px / 2

    y, x = np.meshgrid(
        coordinates,
        coordinates,
        indexing="ij"
    )

    radius_px = (
        sun_diameter_mm
        * output_px_per_mm
        / 2.0
    )

    sun = x*x + y*y <= radius_px*radius_px

    # Retains the phase convention used by the existing simulator.
    moon_offset_px = (
        2.0 * phase * radius_px
    )

    moon = (
        (x - moon_offset_px) ** 2 + y*y
        <= radius_px*radius_px
    )

    crescent = (
        sun & ~moon
    ).astype(np.float32)

    crescent = crescent.reshape(
        diameter_px,
        supersample,
        diameter_px,
        supersample
    ).mean(axis=(1, 3))

    return crescent


def fft_convolve_full(image, kernel):
    """
    Full 2D convolution using NumPy FFT.

    The returned array includes the complete expanded result, so a
    large projected aperture cannot be clipped to the original solar
    kernel's rectangular bounds.
    """
    output_shape = (
        image.shape[0] + kernel.shape[0] - 1,
        image.shape[1] + kernel.shape[1] - 1,
    )

    image_fft = np.fft.rfftn(image, output_shape)
    kernel_fft = np.fft.rfftn(kernel, output_shape)

    result = np.fft.irfftn(
        image_fft * kernel_fft,
        output_shape,
    )

    # Numerical FFT noise can produce tiny negative values.
    return np.maximum(result, 0.0).astype(np.float32)


def projector_to_background_matrix(
        projector_x,
        projector_y,
        light_direction,
        background_origin,
        background_normal,
        screen_x,
        screen_y
):
    """
    Return the 2×2 linear mapping from millimetres on the projector
    plate to millimetres on the background plane.

    This accounts for an oblique background surface, so a circular
    hole may project as an ellipse.
    """

    projector_center = np.zeros(3)

    center_hit = intersect_plane(
        projector_center,
        light_direction,
        background_origin,
        background_normal
    )

    if center_hit is None:
        raise ValueError(
            "The central light ray does not intersect the background."
        )

    def project_offset(offset):
        hit = intersect_plane(
            offset,
            light_direction,
            background_origin,
            background_normal
        )

        if hit is None:
            raise ValueError(
                "A projector-plane basis ray does not intersect "
                "the background."
            )

        delta = hit - center_hit

        return np.array([
            np.dot(delta, screen_x),
            np.dot(delta, screen_y)
        ])

    mapped_x = project_offset(projector_x)
    mapped_y = project_offset(projector_y)

    return np.column_stack((mapped_x, mapped_y))


def create_projected_hole_kernel(
        hole_size_mm,
        projector_to_screen,
        output_px_per_mm,
        supersample=4
):
    """
    Rasterize the projected image of a circular hole.

    The physical hole is circular on the projector plate. On an
    oblique background, its parallel projection becomes an ellipse.
    """

    if hole_size_mm <= 0:
        raise ValueError("hole_size_mm must be positive")

    radius_mm = hole_size_mm / 2.0

    # Extents of the transformed circular aperture along screen X/Y.
    extent_x_mm = radius_mm * np.linalg.norm(
        projector_to_screen[0, :]
    )
    extent_y_mm = radius_mm * np.linalg.norm(
        projector_to_screen[1, :]
    )

    half_width_px = max(
        1,
        int(np.ceil(extent_x_mm * output_px_per_mm)) + 1
    )
    half_height_px = max(
        1,
        int(np.ceil(extent_y_mm * output_px_per_mm)) + 1
    )

    width_px = 2 * half_width_px + 1
    height_px = 2 * half_height_px + 1

    sample_width = width_px * supersample
    sample_height = height_px * supersample

    x_px = (
        np.arange(sample_width) + 0.5
    ) / supersample - width_px / 2

    y_px = (
        np.arange(sample_height) + 0.5
    ) / supersample - height_px / 2

    x_mm = x_px / output_px_per_mm
    y_mm = y_px / output_px_per_mm

    screen_y_grid, screen_x_grid = np.meshgrid(
        y_mm,
        x_mm,
        indexing="ij"
    )

    screen_points = np.stack(
        [screen_x_grid, screen_y_grid],
        axis=-1
    )

    inverse_mapping = np.linalg.inv(
        projector_to_screen
    )

    # Map each screen sample back onto the projector plate.
    projector_points = (
        screen_points @ inverse_mapping.T
    )

    inside = (
        projector_points[..., 0] ** 2
        + projector_points[..., 1] ** 2
        <= radius_mm ** 2
    )

    # Downsample to antialiased output pixels.
    kernel = inside.astype(np.float32).reshape(
        height_px,
        supersample,
        width_px,
        supersample
    ).mean(axis=(1, 3))

    # Preserve relative transmitted light. Do not normalize to a peak
    # of 1 here; larger holes should transmit more total light.
    return kernel


# -----------------------------
# Simulation
# -----------------------------

def simulate(
        text,
        text_size_mm,

        sun_direction,
        background_normal,
        background_size_mm,

        hole_spacing_x_mm,
        hole_spacing_y_mm,

        projection_distance_mm,

        phase,
        hole_size_mm,

        output_px_per_mm,

        clip_brightness=1.0,
        letter_spacing_percent=0.0,
        line_spacing_percent=20.0,
):

    sun_direction = normalize(
        np.array(sun_direction)
    )

    background_normal = normalize(
        np.array(background_normal)
    )


    #
    # Projector coordinate system
    #

    projector_normal = sun_direction
    projector_x, projector_y = plane_basis(projector_normal)

    pattern = create_text_mask(
        text,
        text_size_mm[0],
        text_size_mm[1],
        hole_spacing_x_mm,
        hole_spacing_y_mm,
        letter_spacing_percent,
        line_spacing_percent,
    )

    holes = []
    h, w = pattern.shape

    for y in range(h):
        for x in range(w):
            if pattern[y, x]:
                holes.append(
                    projector_x
                    * ((x - (w - 1) / 2) * hole_spacing_x_mm)
                    + projector_y
                    * (((h - 1) / 2 - y) * hole_spacing_y_mm)
                )

    if not holes:
        raise ValueError("The generated mask contains no pinholes.")

    holes = np.asarray(holes, dtype=float)

    background_origin = (
            -sun_direction * projection_distance_mm
    )

    #
    # Background coordinate system
    #
    # Project the projector's own X and Y axes onto the background.
    # This preserves the visual orientation of the pinhole pattern.
    #

    screen_x = (
            projector_x
            - np.dot(projector_x, background_normal) * background_normal
    )

    if np.linalg.norm(screen_x) < 1e-8:
        raise ValueError(
            "Cannot define background horizontal axis: "
            "projector X is perpendicular to the background plane."
        )

    screen_x = normalize(screen_x)

    screen_y = (
            projector_y
            - np.dot(projector_y, background_normal) * background_normal
    )

    # Remove any component parallel to screen_x.
    screen_y = screen_y - np.dot(screen_y, screen_x) * screen_x

    if np.linalg.norm(screen_y) < 1e-8:
        raise ValueError(
            "Cannot define background vertical axis: "
            "projector Y is degenerate on the background plane."
        )

    screen_y = normalize(screen_y)

    # Explicitly preserve the projector-axis directions.
    if np.dot(screen_x, projector_x) < 0:
        screen_x = -screen_x

    if np.dot(screen_y, projector_y) < 0:
        screen_y = -screen_y

    light_direction = -sun_direction

    projector_to_screen = projector_to_background_matrix(
        projector_x=projector_x,
        projector_y=projector_y,
        light_direction=light_direction,
        background_origin=background_origin,
        background_normal=background_normal,
        screen_x=screen_x,
        screen_y=screen_y
    )

    #
    # Project holes
    #

    projections=[]

    light_direction = -sun_direction


    for hole in holes:

        hit = intersect_plane(
            hole,
            light_direction,
            background_origin,
            background_normal
        )

        if hit is None:
            continue


        px = np.dot(
            hit-background_origin,
            screen_x
        )

        py = np.dot(
            hit-background_origin,
            screen_y
        )

        projections.append(
            [px, py]
        )


    projections=np.array(projections)


    #
    # Render full background surface
    #

    bg_width, bg_height = background_size_mm


    width_px = int(
        bg_width *
        output_px_per_mm
    )

    height_px = int(
        bg_height *
        output_px_per_mm
    )


    image = np.zeros(
        (
            height_px,
            width_px
        ),
        dtype=np.float32
    )


    #
    # Eclipse image size
    #

    # Angular diameter of the Sun is approximately 0.53 degrees.
    sun_diameter_mm = (
            projection_distance_mm
            * np.tan(np.deg2rad(0.53))
    )

    ideal_crescent = create_crescent_kernel(
        sun_diameter_mm=sun_diameter_mm,
        phase=phase,
        output_px_per_mm=output_px_per_mm
    )

    projected_hole = create_projected_hole_kernel(
        hole_size_mm=hole_size_mm,
        projector_to_screen=projector_to_screen,
        output_px_per_mm=output_px_per_mm
    )

    # The finite aperture blurs the ideal solar image.
    optical_kernel = fft_convolve_full(
        ideal_crescent,
        projected_hole
    )

    kernel_height, kernel_width = optical_kernel.shape


    #
    # Stamp eclipse images
    #

    for x, y in projections:

        ix = int(round(
            (x + bg_width / 2)
            * output_px_per_mm
        ))

        iy = int(round(
            (bg_height / 2 - y)
            * output_px_per_mm
        ))

        x0 = ix - kernel_width // 2
        y0 = iy - kernel_height // 2

        source_x0 = max(0, -x0)
        source_y0 = max(0, -y0)

        source_x1 = min(
            kernel_width,
            width_px - x0
        )

        source_y1 = min(
            kernel_height,
            height_px - y0
        )

        if (
                source_x1 <= source_x0
                or source_y1 <= source_y0
        ):
            continue

        destination_x0 = x0 + source_x0
        destination_y0 = y0 + source_y0

        destination_x1 = x0 + source_x1
        destination_y1 = y0 + source_y1

        image[
            destination_y0:destination_y1,
            destination_x0:destination_x1
        ] += optical_kernel[
            source_y0:source_y1,
            source_x0:source_x1
        ]

    if clip_brightness <= 0:
        clip_brightness = image.max()

    if clip_brightness > 0:
        image /= clip_brightness


    projector_size_mm = (
        w * hole_spacing_x_mm,
        h * hole_spacing_y_mm
    )


    return (
        image,
        holes,
        projections,
        projector_size_mm
    )
