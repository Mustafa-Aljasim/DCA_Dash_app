# app.py
import base64
import io
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dash import Dash, dcc, html, Input, Output, State, exceptions
import plotly.graph_objects as go

# ----------------------------
# Arps model helpers
# ----------------------------
def arps_rate(t_years, qi, Di, b):
    t_years = np.asarray(t_years, dtype=float)
    qi = float(qi)
    Di = float(Di)
    b = float(b)

    if np.isclose(b, 0.0):
        return qi * np.exp(-Di * t_years)

    denom = np.clip((1.0 + b * Di * t_years), 1e-12, np.inf)
    return qi / (denom ** (1.0 / b))


def cumulative_trapz(dates, rates):
    """
    Cumulative production by trapezoidal integration over the (irregular) time grid.
    dates: pandas.DatetimeIndex or array-like of datetimes
    rates: array-like (same length), in rate-per-day units (e.g., STB/day)
    Returns cumulative in 'barrels' or same volumetric unit as rate-per-day * day.
    """
    if len(dates) < 2:
        return 0.0

    # Create a DataFrame to ensure dates and rates are sorted together correctly.
    df = pd.DataFrame({'dates': pd.to_datetime(dates), 'rates': rates})
    df = df.sort_values(by='dates').dropna()

    if len(df) < 2:
        return 0.0

    d_vals = df['dates'].values
    r_vals = df['rates'].values

    # time steps in days between points
    dt_days = np.diff(d_vals).astype("timedelta64[s]").astype(float) / 86400.0
    # trapezoid areas
    areas = 0.5 * (r_vals[:-1] + r_vals[1:]) * dt_days
    return float(np.nansum(areas))


def make_sample_data():
    start = pd.Timestamp("2020-01-01")
    n_days = int(3 * 365.25)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t_years = (dates - dates[0]).days / 365.25
    true_qi, true_Di, true_b = 1200.0, 0.85, 0.7
    q = arps_rate(t_years, true_qi, true_Di, true_b)
    np.random.seed(0)
    noise = np.random.normal(scale=0.03 * np.nanmax(q), size=q.shape)
    q_noisy = np.clip(q + noise, a_min=0, a_max=None)
    return pd.DataFrame({"Date": dates, "OIL": q_noisy})


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)
app.title = "Arps DCA – Plotly Dash"

DEFAULT_DF = make_sample_data()

def parse_contents(contents, filename):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next((c for c in df.columns if "date" in c), None)
    oil_col = next((c for c in df.columns if "oil" in c or "rate" in c), None)
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "OIL": pd.to_numeric(df[oil_col], errors="coerce")
    }).dropna().sort_values("Date")
    return out.reset_index(drop=True)


# Layout
app.layout = html.Div(style={"maxWidth": "1100px", "margin": "auto", "padding": "16px"}, children=[
    html.H2("Decline Curve Analysis (Arps)"),

    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap"}, children=[
        dcc.RadioItems(
            id="theme",
            options=[{"label": "Light", "value": "plotly_white"},
                     {"label": "Dark", "value": "plotly_dark"}],
            value="plotly_white", labelStyle={"marginRight": "12px"}
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
            style={"width": "220px", "lineHeight": "38px", "borderWidth": "1px", "borderStyle": "dashed",
                   "borderRadius": "8px", "textAlign": "center"},
            multiple=False
        ),
        html.Div(id="file-name", style={"fontSize": "12px", "opacity": 0.7, "alignSelf": "center"}),
        html.Button("Fit Model", id="fit-btn", n_clicks=0),
        html.Span(id="fit-status", style={"marginLeft": "8px", "fontSize": "12px", "opacity": 0.8}),
    ]),

    html.Hr(),

    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "12px"}, children=[
        html.Div([html.Label("qi"), dcc.Input(id="qi", type="number", value=1200.0, step="any", min=0, style={"width": "100%"})]),
        html.Div([html.Label("Di (1/yr)"), dcc.Input(id="Di", type="number", value=0.8, step="any", min=0, style={"width": "100%"})]),
        html.Div([html.Label("b"), dcc.Input(id="b", type="number", value=0.7, step=0.01, min=0, max=2, style={"width": "100%"})]),
        html.Div([html.Label("Forecast period (years)"), dcc.Slider(id="forecast-years", min=0, max=30, step=0.5, value=10,
                                                                     marks={0: "0", 5: "5", 10: "10", 20: "20", 30: "30"})]),
        html.Div([html.Label("Forecast step"), dcc.Dropdown(
            id="forecast-step",
            options=[{"label": "Daily", "value": "D"}, {"label": "Weekly", "value": "W"}, {"label": "Monthly", "value": "M"}],
            value="D", clearable=False
        )]),
        html.Div([html.Label("Y-axis scale"), dcc.RadioItems(
            id="y-axis-scale",
            options=[{"label": "Linear", "value": "linear"}, {"label": "Log", "value": "log"}],
            value="linear", labelStyle={"marginRight": "12px"}
        )]),
        html.Button("Download Forecast CSV", id="download-btn", n_clicks=0, style={"alignSelf": "end"}),
        dcc.Download(id="download-forecast")
    ]),

    html.Hr(),

    dcc.Graph(id="dca-graph", style={"height": "540px"}),

    html.Div(id="cum-box", style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                                  "gap": "12px", "marginTop": "8px", "padding": "12px",
                                  "border": "1px solid rgba(128,128,128,0.25)", "borderRadius": "8px"}),

    dcc.Store(id="data-store"),
    dcc.Store(id="fitted-params"),
    dcc.Store(id="forecast-data")
])

# Callbacks
@app.callback(
    Output("data-store", "data"),
    Output("file-name", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=False
)
def handle_upload(contents, filename):
    if contents is None:
        return DEFAULT_DF.to_json(date_format="iso", orient="split"), "Using sample data"
    try:
        df = parse_contents(contents, filename or "uploaded.csv")
        return df.to_json(date_format="iso", orient="split"), f"Loaded: {filename}"
    except Exception as e:
        return DEFAULT_DF.to_json(date_format="iso", orient="split"), f"Error reading file: {e}"

@app.callback(
    Output("fitted-params", "data"),
    Output("fit-status", "children"),
    Input("fit-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True
)
def fit_model(n_clicks, data_json):
    df = pd.read_json(data_json, orient="split")
    t_years = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25*24*3600)
    y = df["OIL"].values
    qi0 = max(y)
    mask = y > 0
    Di0 = 0.5
    b0 = 0.7
    bounds = ([1e-8, 1e-6, 0.0], [1e9, 5.0, 2.0])
    try:
        popt, _ = curve_fit(arps_rate, t_years[mask], y[mask], p0=[qi0, Di0, b0], bounds=bounds, maxfev=20000)
        return {"qi": float(popt[0]), "Di": float(popt[1]), "b": float(popt[2])}, "Fit complete."
    except Exception as e:
        return None, f"Fit failed: {e}"

@app.callback(
    Output("qi", "value"), Output("Di", "value"), Output("b", "value"),
    Input("fitted-params", "data"),
    State("qi", "value"), State("Di", "value"), State("b", "value"),
    prevent_initial_call=True
)
def update_inputs_from_fit(params, qi_val, Di_val, b_val):
    if not params:
        raise exceptions.PreventUpdate
    return params["qi"], params["Di"], params["b"]

@app.callback(
    Output("dca-graph", "figure"),
    Output("cum-box", "children"),
    Output("forecast-data", "data"),
    Input("data-store", "data"),
    Input("qi", "value"), Input("Di", "value"), Input("b", "value"),
    Input("forecast-years", "value"), Input("forecast-step", "value"),
    Input("theme", "value"), Input("y-axis-scale", "value"),
)
def update_graph_and_cum(data_json, qi, Di, b, forecast_years, step, theme, yaxis_scale):
    df = pd.read_json(data_json, orient="split").sort_values("Date")
    t_hist = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25*24*3600)
    q_hist_model = arps_rate(t_hist, qi, Di, b)

    last_date = df["Date"].iloc[-1]
    if forecast_years > 0:
        freq = {"D": "D", "W": "W", "M": "MS"}[step]
        periods = int(round(forecast_years * (365.25 / {"D": 1, "W": 7, "M": 30.4375}[step])))
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
        t_fore = (forecast_dates - df["Date"].iloc[0]).days / 365.25
        q_fore = arps_rate(t_fore, qi, Di, b)
    else:
        forecast_dates, q_fore = pd.DatetimeIndex([]), np.array([])

    cum_hist = cumulative_trapz(df["Date"], df["OIL"])
    cum_fore = cumulative_trapz(np.concatenate([[df["Date"].iloc[-1]], forecast_dates]),
                                np.concatenate([[q_hist_model[-1]], q_fore])) if len(q_fore) > 0 else 0.0
    cum_total = cum_hist + cum_fore

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["Date"], y=df["OIL"], mode="markers", name="Historical"))
    fig.add_trace(go.Scatter(x=df["Date"], y=q_hist_model, mode="lines", name="Model (hist)"))
    if len(forecast_dates) > 0:
        fig.add_trace(go.Scatter(x=forecast_dates, y=q_fore, mode="lines", name="Forecast"))

    fig.update_layout(template=theme, yaxis_type=yaxis_scale,
                      xaxis_title="Date", yaxis_title="Rate", hovermode="x unified")

    cum_children = [
        html.Div([html.Div("Cumulative (hist)"), html.Div(f"{cum_hist:,.0f}")]),
        html.Div([html.Div(f"Cumulative (forecast {forecast_years}y)"), html.Div(f"{cum_fore:,.0f}")]),
        html.Div([html.Div("Cumulative (total)"), html.Div(f"{cum_total:,.0f}")]),
    ]

    if len(forecast_dates) > 0:
        forecast_df = pd.DataFrame({"Date": forecast_dates, "Rate": q_fore})
        forecast_df["Cum_Model"] = cumulative_trapz(
            np.concatenate([df["Date"].values, forecast_dates.values]),
            np.concatenate([q_hist_model, q_fore])
        )
    else:
        forecast_df = pd.DataFrame(columns=["Date", "Rate", "Cum_Model"])

    return fig, cum_children, forecast_df.to_json(date_format="iso", orient="split")

@app.callback(
    Output("download-forecast", "data"),
    Input("download-btn", "n_clicks"),
    State("forecast-data", "data"),
    prevent_initial_call=True
)
def download_csv(n_clicks, forecast_json):
    if not forecast_json:
        raise exceptions.PreventUpdate
    df = pd.read_json(forecast_json, orient="split")
    return dcc.send_data_frame(df.to_csv, "forecast.csv", index=False)

if __name__ == "__main__":
    app.run(debug=True)
