import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, TextBox
from matplotlib.ticker import MultipleLocator

from eclipse_projector import plane_basis, simulate
from stl_export import export_pinhole_plate_stl


# -----------------------------
# Initial parameters
# -----------------------------

params = {
    "text": "IZ A TEST?",

    # Physical pinhole pattern
    "text_width": 800.0,
    "text_height": 180.0,

    # White background sheet/wall
    "background_width": 900.0,
    "background_height": 300.0,

    # Sun and wall geometry
    "sun_azimuth": 0.0,
    "sun_elevation": 0.0,
    "wall_tilt": 24.5,

    # Pinhole grid
    "hole_spacing_x": 8.0,
    "hole_spacing_y": 14.0,
    "hole_size": 2.0,

    # Projection
    "projection_distance": 3000.0,
    "phase": 0.15,

    # Rendering
    "resolution": 2.0,
    "clip_brightness": 25.0
}


# -----------------------------
# Direction helpers
# -----------------------------

def direction_from_angles(azimuth_deg, elevation_deg):
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)

    return np.array([
        np.cos(el) * np.sin(az),
        np.sin(el),
        np.cos(el) * np.cos(az),
    ])


def wall_normal_from_tilt(tilt_deg):
    tilt = np.deg2rad(tilt_deg)

    return np.array([
        0.0,
        np.sin(tilt),
        -np.cos(tilt),
    ])


def queue_simulation():
    global pending_simulation_args

    pending_simulation_args = {
        "text": params["text"],

        "text_size_mm": (
            params["text_width"],
            params["text_height"],
        ),

        "sun_direction": direction_from_angles(
            params["sun_azimuth"],
            params["sun_elevation"],
        ),

        "background_normal": wall_normal_from_tilt(
            params["wall_tilt"],
        ),

        "background_size_mm": (
            params["background_width"],
            params["background_height"],
        ),

        "hole_spacing_x_mm": params["hole_spacing_x"],
        "hole_spacing_y_mm": params["hole_spacing_y"],

        "projection_distance_mm": params["projection_distance"],

        "phase": params["phase"],
        "hole_size_mm": params["hole_size"],

        "output_px_per_mm": params["resolution"],
        "clip_brightness": params["clip_brightness"],
    }

    update_timer.stop()
    update_timer.start()

# -----------------------------
# Plot setup
# -----------------------------

fig, ax = plt.subplots(figsize=(14, 8))

pending_simulation_args = None

update_timer = fig.canvas.new_timer(interval=150)
update_timer.single_shot = True

ax.set_facecolor("white")

ax.xaxis.set_major_locator(MultipleLocator(100))
ax.yaxis.set_major_locator(MultipleLocator(100))

ax.xaxis.set_minor_locator(MultipleLocator(20))
ax.yaxis.set_minor_locator(MultipleLocator(20))

ax.grid(
    which="major",
    linewidth=0.8,
    alpha=0.55,
)

ax.grid(
    which="minor",
    linewidth=0.35,
    alpha=0.25,
)

ax.set_xlabel("Horizontal position (mm)")
ax.set_ylabel("Vertical position (mm)")

ax.tick_params(
    axis="both",
    which="both",
    labelsize=8,
)

ax.set_axisbelow(True)

# Reserve the right side for controls.
plt.subplots_adjust(
    left=0.04,
    right=0.6,
    top=0.90,
    bottom=0.06,
)

status_text = fig.text(
    0.04,
    0.015,
    "",
    ha="left",
    va="bottom",
    fontsize=9,
)

# Current displayed image and physical background size
image_handle = None
current_background_size = None
current_hole_positions = None
current_hole_diameter = None


# -----------------------------
# Simulation update
# -----------------------------

update_num = 1
def update(_=None):
    global image_handle
    global current_background_size
    global pending_simulation_args
    global update_num
    global current_hole_positions
    global current_hole_diameter

    if pending_simulation_args is None:
        return

    simulation_args = pending_simulation_args
    pending_simulation_args = None

    try:
        print(f"Generating projected image... [#{update_num}]")
        update_num += 1
        img, holes, projections, projector_size_mm = simulate(
            **simulation_args
        )

    except (ValueError, FloatingPointError) as error:
        status_text.set_text(
            f"Simulation error: {error}"
        )
        fig.canvas.draw_idle()
        return

    projector_x, projector_y = plane_basis(
        simulation_args["sun_direction"]
    )
    current_hole_positions = np.column_stack((
        holes @ projector_x,
        holes @ projector_y,
    ))
    current_hole_diameter = simulation_args["hole_size_mm"]

    bg_width = params["background_width"]
    bg_height = params["background_height"]

    new_background_size = (
        bg_width,
        bg_height,
    )

    background_size_changed = (
        current_background_size != new_background_size
    )

    current_background_size = new_background_size

    extent = (
        -bg_width / 2,
        bg_width / 2,
        -bg_height / 2,
        bg_height / 2,
    )

    if image_handle is None:
        ax.set_facecolor("white")

        image_handle = ax.imshow(
            img,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            extent=extent,
            origin="upper",
            aspect="equal",
            interpolation="nearest",
            zorder=2,
        )

        ax.xaxis.set_major_locator(
            MultipleLocator(100)
        )

        ax.yaxis.set_major_locator(
            MultipleLocator(100)
        )

        ax.xaxis.set_minor_locator(
            MultipleLocator(20)
        )

        ax.yaxis.set_minor_locator(
            MultipleLocator(20)
        )

        ax.grid(
            which="major",
            linewidth=0.8,
            alpha=0.55,
            zorder=0,
        )

        ax.grid(
            which="minor",
            linewidth=0.35,
            alpha=0.25,
            zorder=0,
        )

        ax.set_xlabel("Horizontal position (mm)")
        ax.set_ylabel("Vertical position (mm)")

        ax.tick_params(
            axis="both",
            which="both",
            labelsize=8,
        )

        ax.set_axisbelow(True)

        ax.set_xlim(
            -bg_width / 2,
            bg_width / 2,
        )

        ax.set_ylim(
            -bg_height / 2,
            bg_height / 2,
        )

    else:
        image_handle.set_data(img)
        image_handle.set_extent(extent)

        if background_size_changed:
            ax.set_xlim(
                -bg_width / 2,
                bg_width / 2,
            )

            ax.set_ylim(
                -bg_height / 2,
                bg_height / 2,
            )

    ax.set_title(
        f"{params['text']}\n"
        f"Distance: {params['projection_distance'] / 1000:.2f} m   "
        f"Visible phase: {params['phase']:.3f}   "
        f"Hole diameter: {params['hole_size']:.2f} mm"
    )

    status_text.set_text(
        f"Pinhole plate: "
        f"{projector_size_mm[0]:.0f} × "
        f"{projector_size_mm[1]:.0f} mm   |   "
        f"Background: "
        f"{bg_width:.0f} × "
        f"{bg_height:.0f} mm   |   "
        f"Holes: {len(holes)}"
    )

    fig.canvas.draw_idle()

update_timer.add_callback(update)

# -----------------------------
# Text input
# -----------------------------

text_box_ax = fig.add_axes([
    0.77,
    0.925,
    0.20,
    0.035,
])

text_box = TextBox(
    text_box_ax,
    "Text ",
    initial=params["text"]
)

# Work around a Matplotlib bug where TextBox._resize is wrapped as a
# mouse-only callback and tries to access ResizeEvent.inaxes.
resize_callbacks = fig.canvas.callbacks.callbacks.get(
    "resize_event",
    {}
)

for callback_id, callback_ref in list(resize_callbacks.items()):
    callback = callback_ref()

    if (
        callback is not None
        and getattr(callback, "__self__", None) is text_box
        and getattr(callback, "__name__", "") == "_resize"
    ):
        fig.canvas.mpl_disconnect(callback_id)


def update_text(value):
    cleaned = value.strip()

    if cleaned:
        params["text"] = cleaned
        queue_simulation()


text_box.on_submit(update_text)


# -----------------------------
# Sliders
# -----------------------------

slider_specs = [
    (
        "text_width",
        "Text width (mm)",
        100.0,
        1000.0,
        params["text_width"],
        1.0,
    ),
    (
        "text_height",
        "Text height (mm)",
        50.0,
        500.0,
        params["text_height"],
        1.0,
    ),
    (
        "background_width",
        "Background width (mm)",
        200.0,
        3000.0,
        params["background_width"],
        10.0,
    ),
    (
        "background_height",
        "Background height (mm)",
        200.0,
        2000.0,
        params["background_height"],
        10.0,
    ),
    (
        "sun_azimuth",
        "Sun azimuth (°)",
        -180.0,
        180.0,
        params["sun_azimuth"],
        1.0,
    ),
    (
        "sun_elevation",
        "Sun elevation (°)",
        1.0,
        90.0,
        params["sun_elevation"],
        0.5,
    ),
    (
        "wall_tilt",
        "Wall tilt (°)",
        -80.0,
        80.0,
        params["wall_tilt"],
        0.5,
    ),
    (
        "hole_spacing_x",
        "Hole spacing X (mm)",
        2.0,
        40.0,
        params["hole_spacing_x"],
        0.25,
    ),
    (
        "hole_spacing_y",
        "Hole spacing Y (mm)",
        2.0,
        40.0,
        params["hole_spacing_y"],
        0.25,
    ),
    (
        "projection_distance",
        "Projection distance (mm)",
        500.0,
        5000.0,
        params["projection_distance"],
        10.0,
    ),

    # Eclipse phase control
    (
        "phase",
        "Visible Sun phase",
        0.01,
        1.0,
        params["phase"],
        0.01,
    ),

    # Hole size control
    (
        "hole_size",
        "Hole diameter (mm)",
        0.25,
        15.0,
        params["hole_size"],
        0.05,
    ),

    (
        "resolution",
        "Output pixels/mm",
        0.25,
        5.0,
        params["resolution"],
        0.25,
    ),

    (
        "clip_brightness",
        "Output max brightness",
        0.0,
        250.0,
        params["clip_brightness"],
        1,
    ),
]


sliders = {}

panel_left = 0.77
panel_width = 0.20

slider_top = 0.865
slider_step = 0.061
slider_height = 0.025


def make_callback(name):
    def callback(value):
        params[name] = float(value)
        queue_simulation()

    return callback


for index, (
    key,
    label,
    low,
    high,
    initial,
    step,
) in enumerate(slider_specs):

    y = slider_top - index * slider_step

    slider_ax = fig.add_axes([
        panel_left,
        y,
        panel_width,
        slider_height,
    ])

    slider = Slider(
        ax=slider_ax,
        label=label,
        valmin=low,
        valmax=high,
        valinit=initial,
        valstep=step,
    )

    slider.on_changed(make_callback(key))
    sliders[key] = slider


# -----------------------------
# Reset button
# -----------------------------

reset_ax = fig.add_axes([
    0.84,
    0.035,
    0.09,
    0.04,
])

reset_button = Button(
    reset_ax,
    "Reset",
)


def reset(_):
    text_box.set_val("ZA VEDNO?")

    for slider in sliders.values():
        slider.reset()


reset_button.on_clicked(reset)

reset_view_ax = fig.add_axes([
    0.73,
    0.035,
    0.09,
    0.04,
])

reset_view_button = Button(
    reset_view_ax,
    "Reset view",
)


def reset_view(_):
    if current_background_size is None:
        return

    bg_width, bg_height = current_background_size

    ax.set_xlim(
        -bg_width / 2,
        bg_width / 2,
    )

    ax.set_ylim(
        -bg_height / 2,
        bg_height / 2,
    )

    fig.canvas.draw_idle()


reset_view_button.on_clicked(reset_view)

# -----------------------------
# Pinhole plate export
# -----------------------------

export_ax = fig.add_axes([
    0.62,
    0.035,
    0.09,
    0.04,
])

export_button = Button(
    export_ax,
    "Export STL",
)


def export_holes(_):
    if current_hole_positions is None or current_hole_diameter is None:
        status_text.set_text("No pinhole plate is available to export.")
        fig.canvas.draw_idle()
        return

    # Tkinter supplies a native save dialog without tying this action to a
    # particular Matplotlib GUI backend.
    from tkinter import Tk, filedialog

    dialog_root = Tk()
    dialog_root.withdraw()

    try:
        file_path = filedialog.asksaveasfilename(
            parent=dialog_root,
            title="Export pinhole plate",
            defaultextension=".stl",
            filetypes=(("STL files", "*.stl"), ("All files", "*.*")),
            initialfile="pinhole_plate.stl",
        )
    finally:
        dialog_root.destroy()

    if not file_path:
        return

    try:
        triangle_count = export_pinhole_plate_stl(
            file_path=file_path,
            hole_positions_mm=current_hole_positions,
            plate_size_mm=current_background_size,
            hole_diameter_mm=current_hole_diameter,
            thickness_mm=5.0,
        )
    except (OSError, ValueError) as error:
        status_text.set_text(f"Export error: {error}")
    else:
        status_text.set_text(
            f"Exported {len(current_hole_positions)} holes "
            f"({triangle_count} triangles) to {file_path}"
        )

    fig.canvas.draw_idle()


export_button.on_clicked(export_holes)

# -----------------------------
# Start UI
# -----------------------------

queue_simulation()
plt.show()
