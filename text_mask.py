import numpy as np
import math

from PIL import Image, ImageDraw, ImageFont

def create_text_mask(
        text,
        text_width_mm,
        text_height_mm,
        hole_spacing_x_mm,
        hole_spacing_y_mm,
        letter_spacing_percent=0.0,
        line_spacing_percent=20.0,
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

    if not np.isfinite(letter_spacing_percent):
        raise ValueError("letter spacing must be finite")

    if not np.isfinite(line_spacing_percent):
        raise ValueError("line spacing must be finite")

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

    line_spacing = source_font_size * line_spacing_percent / 100.0
    letter_spacing = source_font_size * letter_spacing_percent / 100.0
    lines = text.split("\n")
    ascent, descent = source_font.getmetrics()
    line_advance = ascent + descent + line_spacing
    line_widths = [
        measurement_draw.textlength(line, font=source_font)
        + max(0, len(line) - 1) * letter_spacing
        for line in lines
    ]
    block_width = max(line_widths, default=0.0)
    glyphs = []
    bounds = []

    for line_index, (line, line_width) in enumerate(zip(lines, line_widths)):
        line_x = (block_width - line_width) / 2.0
        line_y = line_index * line_advance

        for character_index, character in enumerate(line):
            prefix_width = measurement_draw.textlength(
                line[:character_index + 1],
                font=source_font,
            ) - measurement_draw.textlength(
                character,
                font=source_font,
            )
            character_x = (
                line_x
                + prefix_width
                + character_index * letter_spacing
            )
            position = (character_x, line_y)
            glyphs.append((position, character))
            bounds.append(measurement_draw.textbbox(
                position,
                character,
                font=source_font,
            ))

    if not bounds:
        raise ValueError("text must contain at least one visible character")

    left = min(bound[0] for bound in bounds)
    top = min(bound[1] for bound in bounds)
    right = max(bound[2] for bound in bounds)
    bottom = max(bound[3] for bound in bounds)
    left = math.floor(left)
    top = math.floor(top)
    right = math.ceil(right)
    bottom = math.ceil(bottom)

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

    for (character_x, character_y), character in glyphs:
        source_draw.text(
            (character_x - left, character_y - top),
            character,
            font=source_font,
            fill=255,
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
