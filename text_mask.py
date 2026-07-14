import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_text_mask(
        text,
        text_width_mm,
        text_height_mm,
        hole_spacing_x_mm,
        hole_spacing_y_mm,
        supersample=8
):
    """
    Create the pinhole mask.

    text_width_mm and text_height_mm define the exact physical
    bounding box of the resulting pinhole pattern.

    The rendered glyphs are cropped to their actual bounds and then
    scaled independently in X and Y to fill that physical bounding box.
    """

    if not text:
        raise ValueError("text must not be empty")

    if text_width_mm <= 0 or text_height_mm <= 0:
        raise ValueError("text dimensions must be positive")

    if hole_spacing_x_mm <= 0 or hole_spacing_y_mm <= 0:
        raise ValueError("hole spacing must be positive")

    if supersample < 1:
        raise ValueError("supersample must be at least 1")

    cols = max(
        1,
        int(round(text_width_mm / hole_spacing_x_mm))
    )

    rows = max(
        1,
        int(round(text_height_mm / hole_spacing_y_mm))
    )

    target_width_px = cols * supersample
    target_height_px = rows * supersample

    font_path = (
        "/usr/share/fonts/truetype/dejavu/"
        "DejaVuSans-Bold.ttf"
    )

    #
    # Determine a safe source render size from the text itself.
    # This canvas is created from the measured glyph bounds, so the
    # text cannot be clipped by an arbitrary fixed canvas.
    #

    source_font_size = max(64, target_height_px)

    measurement_image = Image.new("L", (1, 1), 0)
    measurement_draw = ImageDraw.Draw(measurement_image)

    source_font = ImageFont.truetype(
        font_path,
        source_font_size
    )

    left, top, right, bottom = measurement_draw.textbbox(
        (0, 0),
        text,
        font=source_font
    )

    glyph_width = max(1, right - left)
    glyph_height = max(1, bottom - top)

    #
    # Render onto a canvas derived exactly from the measured bounds.
    #

    source_image = Image.new(
        "L",
        (glyph_width, glyph_height),
        0
    )

    source_draw = ImageDraw.Draw(source_image)

    source_draw.text(
        (-left, -top),
        text,
        font=source_font,
        fill=255
    )

    #
    # Resize independently in X and Y.
    #
    # This deliberately stretches the glyphs so their final physical
    # bounding box is exactly text_width_mm × text_height_mm.
    #

    stretched_image = source_image.resize(
        (target_width_px, target_height_px),
        Image.Resampling.LANCZOS
    )

    stretched = (
        np.asarray(stretched_image, dtype=np.float32) / 255.0
    )

    #
    # Downsample each supersampled cell to one possible hole location.
    #

    coverage = stretched.reshape(
        rows,
        supersample,
        cols,
        supersample
    ).mean(axis=(1, 3))

    return coverage > 0.35