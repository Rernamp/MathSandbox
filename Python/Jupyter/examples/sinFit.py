import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    import lmfit

    from rnmp.utils.signal import add_noise

    return add_noise, go, lmfit, mo, np


@app.cell
def _(np):
    rng = np.random.default_rng(42)
    return (rng,)


@app.cell
def _(mo):
    widgets = mo.ui.dictionary({
        "snr":mo.ui.number(label="SNR, dB", value=20),
        "sin_config": mo.ui.dictionary({
            "A":mo.ui.number(label="A", value=1),
            "f":mo.ui.number(label="f, Hz", value=1),
            "phi":mo.ui.number(label="Phi, rad", value=1),
            "offset":mo.ui.number(label="Offset", value=1)
        }),
        "init_offset": mo.ui.dictionary({
            "A":mo.ui.number(label="A", value=0),
            "f":mo.ui.number(label="f, Hz", value=0),
            "phi":mo.ui.number(label="Phi, rad", value=0),
            "offset":mo.ui.number(label="Offset", value=0)
        })  
    }, label="configuration")
    return (widgets,)


@app.cell
def _(mo):
    def render(el, title=None):
        if isinstance(el, mo.ui.dictionary):
            children = [render(v, k) for k, v in el.items()]
            block = mo.vstack(children, justify="start")
            return mo.vstack([mo.md(f"**{title}**"), block]) if title else mo.vstack(children)
        return el

    return (render,)


@app.cell
def _(render, widgets):
    render(widgets) 
    return


@app.cell
def _():
    duration_s = 1
    number_points = 100
    return duration_s, number_points


@app.cell
def _(duration_s, np, number_points, rng):
    time_points = np.sort(rng.uniform(size=number_points) * duration_s)
    return (time_points,)


@app.cell
def _(widgets):
    config = widgets.value
    snr = config["snr"]
    return config, snr


@app.cell
def _(config, sin_model, time_points):
    reference_signal = sin_model(time=time_points, **config["sin_config"])
    return (reference_signal,)


@app.cell
def _(add_noise, reference_signal, snr):
    noised_signal = add_noise(reference_signal, snr)
    return (noised_signal,)


@app.cell
def _(config, lmfit, noised_signal, np):
    _std_signal = np.std(noised_signal)
    _mean_signal = np.mean(noised_signal)

    _initial_offset = _mean_signal
    _initial_A = _std_signal * 2

    _initial_f = np.count_nonzero(np.diff(np.signbit(noised_signal-_mean_signal)))
    _initial_phi = np.arcsin((noised_signal[0] - _initial_offset) / _initial_A)

    params = lmfit.Parameters()
    params.add('A', value=_initial_A + config["init_offset"]["A"])
    params.add('offset', value=_initial_offset + config["init_offset"]["offset"])
    params.add('f', value=_initial_f + config["init_offset"]["f"])
    params.add('phi', value=_initial_phi + config["init_offset"]["f"])
    return (params,)


@app.cell
def _(np):
    def sin_model(A, f, phi, time, offset):
        return A * np.sin(2 * np.pi * time * f + phi) + offset

    return (sin_model,)


@app.cell
def _(sin_model):
    def fit_model(params, time_points, noised_signal):    
        result = sin_model(time=time_points, **params) - noised_signal
        return result

    return (fit_model,)


@app.cell
def _(np):
    def d_fit_model(params, time_points, noised_signal):
        A = params["A"].value
        f = params["f"].value
        phi = params["phi"].value
        return np.array([
            np.sin(2 * np.pi * f * time_points + phi),
            np.ones_like(time_points),
            A * 2 * np.pi * time_points * np.cos(2 * np.pi * f * time_points + phi),
            A * np.cos(2 * np.pi * f * time_points + phi)
        ])

    return (d_fit_model,)


@app.cell
def _():
    values = []
    return (values,)


@app.cell
def _(np, values):
    def iter_cb(params, iter, resid, *args, **kws):
        values.append(np.sum(resid**2))

    return (iter_cb,)


@app.cell
def _(
    d_fit_model,
    fit_model,
    iter_cb,
    lmfit,
    noised_signal,
    params,
    time_points,
):
    #min = lmfit.Minimizer(fit_model, params, fcn_args=(time_points, noised_signal))
    #out = min.leastsq(Dfun=d_fit_model, col_deriv=1)

    out = lmfit.minimize(fit_model, params, args=(time_points, noised_signal), Dfun=d_fit_model, col_deriv=1, iter_cb=iter_cb)
    # out = lmfit.minimize(fit_model, params, args=(time_points, noised_signal))
    return (out,)


@app.cell
def _(out, sin_model, time_points):
    fitted_signal = sin_model(time=time_points, **out.params)
    return (fitted_signal,)


@app.cell(hide_code=True)
def _(fitted_signal, go, noised_signal, reference_signal, time_points):
    _fig = go.Figure(
        data=[
            go.Scatter(x=time_points, y=reference_signal, name="reference_signal", mode="lines"),
            go.Scatter(x=time_points, y=noised_signal, name="noised_signal", mode="markers"),
            go.Scatter(x=time_points, y=fitted_signal, name="fitted_signal", mode="lines")
        ]
    )
    _fig
    return


@app.cell
def _(go, np, values):
    _fig = go.Figure(
        data=[
            go.Scatter(x=np.linspace(0, len(values) - 1, len(values)), y=values, name="errors", mode="lines")
        ]
    )
    _fig

    return


@app.cell
def _(out):
    out.params
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
