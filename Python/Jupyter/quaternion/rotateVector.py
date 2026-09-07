import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import quaternion

    return mo, np, quaternion


@app.cell(hide_code=True)
def _():
    import anywidget
    import traitlets

    class VectorViewer(anywidget.AnyWidget):
        vectors = traitlets.List([]).tag(sync=True)
        limit = traitlets.Float(10.0).tag(sync=True)

        _esm = """
        import Plotly from "https://esm.sh/plotly.js-dist-min@2";

        function makeData(vectors) {
          const traces = [];
          for (const item of vectors) {
            const v = item.vec;
            const color = item.color || "crimson";
            traces.push({
              type: "scatter3d", mode: "lines",
              x: [0, v[0]], y: [0, v[1]], z: [0, v[2]],
              line: { width: 8, color },
              name: item.name || "",
              showlegend: !!item.name,
            });
          }
          return traces;
        }

        function makeLayout(L) {
          return {
            uirevision: "keep",
            scene: {
              xaxis: { range: [-L, L] },
              yaxis: { range: [-L, L] },
              zaxis: { range: [-L, L] },
              aspectmode: "cube",
            },
            margin: { l: 0, r: 0, t: 0, b: 0 },
            legend: { x: 0, y: 1 },
          };
        }

        function render({ model, el }) {
          const div = document.createElement("div");
          div.style.height = "500px";
          el.appendChild(div);

          Plotly.newPlot(div, makeData(model.get("vectors")), makeLayout(model.get("limit")));

          model.on("change:vectors", () => {
            Plotly.react(div, makeData(model.get("vectors")), makeLayout(model.get("limit")));
          });
        }

        export default { render };
        """

    return (VectorViewer,)


@app.cell
def _(mo):
    original_vector_ui = mo.ui.matrix([1, 1, 1], step=1e-1, max_value=1, min_value=0)
    mo.md(f"Original vector: {original_vector_ui}")
    return (original_vector_ui,)


@app.cell
def _(mo):
    rotation_angle_get, rotation_angle_set = mo.state(90)
    return rotation_angle_get, rotation_angle_set


@app.cell
def _(rotation_angle_get):
    rotation_angle = rotation_angle_get()
    return (rotation_angle,)


@app.cell
def _(mo):
    # Kept in its own cell, independent of the angle state, so it never
    # gets recreated (and reset to its default) when the angle changes.
    rotation_ray_ui = mo.ui.matrix([1, 1, 1], step=1e-1, max_value=1, min_value=-1)
    return (rotation_ray_ui,)


@app.cell(hide_code=True)
def _(mo, rotation_angle_get, rotation_angle_set):
    rotation_angle_slider_ui = mo.ui.slider(start=0, stop=360, step=1, value=rotation_angle_get(), on_change=rotation_angle_set)
    return (rotation_angle_slider_ui,)


@app.cell
def _(mo, rotation_angle_get, rotation_angle_set):
    rotation_angle_number_ui = mo.ui.number(start=0, stop=360, step=1, value=rotation_angle_get(), on_change=rotation_angle_set)
    return (rotation_angle_number_ui,)


@app.cell
def _(mo, rotation_angle_number_ui, rotation_angle_slider_ui, rotation_ray_ui):
    mo.md(rf"""
    Rotation ray: {rotation_ray_ui}
    Rotation angle: {rotation_angle_slider_ui} {rotation_angle_number_ui} degree
    """)
    return


@app.cell(hide_code=True)
def _(np, rotation_angle, rotation_ray_ui):
    rotation_ray = np.array(rotation_ray_ui.value)
    rotation_vector = (rotation_ray * np.deg2rad(rotation_angle)) / (np.linalg.norm(rotation_ray))
    return rotation_ray, rotation_vector


@app.cell(hide_code=True)
def _(mo, quaternion, rotation_vector):
    rotation_quaternion = quaternion.from_rotation_vector(rotation_vector)
    mo.md(f"rotation {rotation_quaternion}")
    return (rotation_quaternion,)


@app.cell(hide_code=True)
def _(mo, np, original_vector_ui, quaternion, rotation_quaternion):
    original_vector = np.array(original_vector_ui.value)
    rotated_vector = quaternion.rotate_vectors(rotation_quaternion, original_vector)
    mo.md(f"Rotated {rotated_vector}")
    return original_vector, rotated_vector


@app.cell
def _(VectorViewer):
    # Created once: no dependency on the reactive vectors below, so this
    # cell never re-runs on slider changes and the 3D camera stays put.
    viewer = VectorViewer(vectors=[], limit=1.0)
    viewer
    return (viewer,)


@app.cell
def _(original_vector, rotated_vector, rotation_ray, viewer):
    # Re-runs on every slider/matrix change and mutates the existing
    # widget's data via Plotly.react, instead of recreating the widget.
    viewer.vectors = [
        {"vec": original_vector.tolist(), "name": "original", "color": "crimson"},
        {"vec": rotated_vector.tolist(), "name": "rotated", "color": "blue"},
        {"vec": rotation_ray.tolist(), "name": "rotation ray", "color": "green"},
    ]
    return


if __name__ == "__main__":
    app.run()
