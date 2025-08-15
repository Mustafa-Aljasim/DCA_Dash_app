# app.py
import base64
import io
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dash import Dash, dcc, html, Input, Output, State, exceptions, callback_context
import plotly.graph_objects as go

# ----------------------------
# Arps model helpers
# ----------------------------
def arps_rate(t_years, qi, Di, b):
    """
    Arps decline rate q(t).
    t_years : time since start in years (float or array)
    qi      : initial rate (same units as data, e.g., STB/day)
    Di      : initial decline (1/year)
    b       : hyperbolic exponent
    """
    t_years = np.asarray(t_years, dtype=float)
    qi = float(qi)
    Di = float(Di)
    b = float(b)

    # Exponential when b == 0
    if np.isclose(b, 0.0):
        return qi * np.exp(-Di * t_years)

    # Hyperbolic (b != 0)
    denom = (1.0 + b * Di * t_years)
    # Avoid negative/zero inside power due to extreme params
    denom = np.clip(denom, 1e-12, np.inf)
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
    """
    Generate a synthetic daily dataset (3 years) with hyperbolic decline plus noise.
    """
    start = pd.Timestamp("2020-01-01")
    n_days = int(3 * 365.25)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t_years = (dates - dates[0]).days / 365.25

    true_qi, true_Di, true_b = 1200.0, 0.85, 0.7
    q = arps_rate(t_years, true_qi, true_Di, true_b)
    np.random.seed(0)
    noise = np.random.normal(scale=0.03 * np.nanmax(q), size=q.shape)
    q_noisy = np.clip(q + noise, a_min=0, a_max=None)

    df = pd.DataFrame({"Date": dates, "OIL": q_noisy})
    return df


# ----------------------------
# Dash app
# ----------------------------
app = Dash(__name__)
app.title = "Arps DCA – Plotly Dash"

# Store a default dataset in memory
DEFAULT_DF = make_sample_data()

def parse_contents(contents, filename):
    """
    Decode uploaded CSV into DataFrame with Date, OIL columns.
    """
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        if filename.lower().endswith(".csv"):
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
        else:
            # Try reading as CSV anyway
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        # Fall back: try latin-1
        df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))

    # Normalize columns
    cols = {c: c.strip().lower() for c in df.columns}
    df.columns = [cols[c] for c in df.columns]

    # Try to find date and oil columns
    date_col = next((c for c in df.columns if c.lower() in ("date", "dates")), None)
    oil_col = next((c for c in df.columns if c.lower() in ("oil", "oil_rate", "rate", "q", "qo")), None)
    if date_col is None or oil_col is None:
        raise ValueError("CSV must include columns named 'Date' and 'OIL' (case-insensitive).")

    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "OIL": pd.to_numeric(df[oil_col], errors="coerce")
    }).dropna()

    out = out.sort_values("Date")
    return out.reset_index(drop=True)


# Layout
app.layout = html.Div(
    style={"margin": "0 auto", "maxWidth": "1100px", "padding": "16px"},
    children=[
        html.H2("Decline Curve Analysis (Arps)"),
        html.Div(
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"},
            children=[
                dcc.RadioItems(
                    id="theme",
                    options=[{"label": "Light", "value": "plotly_white"},
                             {"label": "Dark", "value": "plotly_dark"}],
                    value="plotly_white",
                    labelStyle={"marginRight": "12px"}
                ),
                html.Div(
                    [
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
                            style={
                                "width": "260px", "lineHeight": "38px",
                                "borderWidth": "1px", "borderStyle": "dashed",
                                "borderRadius": "8px", "textAlign": "center",
                                "padding": "0 8px", "cursor": "pointer",
                            },
                            multiple=False
                        ),
                        html.Div(id="file-name", style={"fontSize": "12px", "opacity": 0.7, "marginTop": "4px"})
                    ]
                ),
                html.Div(
                    [
                        html.Button("Fit Model (Nonlinear Regression)", id="fit-btn", n_clicks=0, style={"height": "40px"}),
                        html.Span(id="fit-status", style={"marginLeft": "8px", "fontSize": "12px", "opacity": 0.8})
                    ]
                ),
            ]
        ),

        html.Hr(),

        # Controls
        html.Div(
            style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))", "gap": "12px"},
            children=[
                html.Div([
                    html.Label("qi (initial rate, e.g., STB/day)"),
                    dcc.Input(id="qi", type="number", value=1200.0, step="any", min=0, style={"width": "100%"})
                ]),
                html.Div([
                    html.Label("Di (initial decline, 1/year)"),
                    dcc.Input(id="Di", type="number", value=0.8, step="any", min=0, style={"width": "100%"})
                ]),
                html.Div([
                    html.Label("b (hyperbolic exponent)"),
                    dcc.Input(id="b", type="number", value=0.7, step=0.01, min=0, max=2.0, style={"width": "100%"})
                ]),
                html.Div([
                    html.Label("Forecast period (years)"),
                    dcc.Slider(id="forecast-years", min=0, max=30, step=0.5, value=10,
                               marks={0: "0", 5: "5", 10: "10", 20: "20", 30: "30"})
                ]),
            ]
        ),

        html.Hr(),

        dcc.Graph(id="dca-graph", style={"height": "540px"}),

        html.Div(
            id="cum-box",
            style={
                "display": "grid",
                "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                "gap": "12px",
                "marginTop": "8px",
                "padding": "12px",
                "border": "1px solid rgba(128,128,128,0.25)",
                "borderRadius": "8px"
            }
        ),

        # Hidden stores
        dcc.Store(id="data-store"),
        dcc.Store(id="fitted-params"),
    ]
)

# ----------------------------
# Callbacks
# ----------------------------

# Initialize with default data at startup
@app.callback(
    Output("data-store", "data"),
    Output("file-name", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=False
)
def handle_upload(contents, filename):
    if contents is None:
        # Use default
        df = DEFAULT_DF.copy()
        return df.to_json(date_format="iso", orient="split"), "Using built-in sample data"
    try:
        df = parse_contents(contents, filename or "uploaded.csv")
        return df.to_json(date_format="iso", orient="split"), f"Loaded: {filename}"
    except Exception as e:
        # Fall back to default if parsing fails
        df = DEFAULT_DF.copy()
        return df.to_json(date_format="iso", orient="split"), f"Upload error; using sample data. ({e})"


@app.callback(
    Output("fitted-params", "data"),
    Output("fit-status", "children"),
    Input("fit-btn", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True
)
def fit_model(n_clicks, data_json):
    if not data_json:
        raise exceptions.PreventUpdate

    df = pd.read_json(data_json, orient="split")
    if df.empty:
        return None, "No data to fit."

    # Prepare t (years) and y
    df = df.sort_values("Date")
    t_years = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25 * 24 * 3600)
    y = df["OIL"].astype(float).values

    # Initial guesses
    qi0 = max(float(np.nanmax(y)), 1e-6)
    # Estimate Di0 via simple exponential fit on early data (rough)
    # Avoid zeros/negatives
    mask = (y > 0) & np.isfinite(y)
    if mask.sum() < 5:
        return None, "Not enough valid points for fitting."
    t_fit = t_years[mask]
    y_fit = y[mask]

    # crude Di guess from log-slope of first 10% segment
    k = max(3, int(0.1 * len(t_fit)))
    k = min(k, len(t_fit)-1)
    if k > 1 and np.all(y_fit[:k] > 0):
        slope, _ = np.polyfit(t_fit[:k], np.log(y_fit[:k]), 1)
        Di0 = max(1e-4, -slope)
    else:
        Di0 = 0.5

    b0 = 0.7

    # Bounds: qi>0, Di>0, b in [0, 2]
    bounds = ([1e-8, 1e-6, 0.0], [1e9, 5.0, 2.0])

    try:
        popt, _ = curve_fit(
            arps_rate, t_fit, y_fit,
            p0=[qi0, Di0, b0],
            bounds=bounds,
            maxfev=20000
        )
        qi_opt, Di_opt, b_opt = [float(x) for x in popt]
        return {"qi": qi_opt, "Di": Di_opt, "b": b_opt}, "Fit complete."
    except Exception as e:
        return None, f"Fit failed: {e}"


@app.callback(
    Output("qi", "value"),
    Output("Di", "value"),
    Output("b", "value"),
    Input("fitted-params", "data"),
    State("qi", "value"),
    State("Di", "value"),
    State("b", "value"),
    prevent_initial_call=True
)
def update_inputs_from_fit(params, qi_val, Di_val, b_val):
    if params is None:
        raise exceptions.PreventUpdate
    return params.get("qi", qi_val), params.get("Di", Di_val), params.get("b", b_val)


@app.callback(
    Output("dca-graph", "figure"),
    Output("cum-box", "children"),
    Input("data-store", "data"),
    Input("qi", "value"),
    Input("Di", "value"),
    Input("b", "value"),
    Input("forecast-years", "value"),
    Input("theme", "value"),
)
def update_graph_and_cum(data_json, qi, Di, b, forecast_years, theme):
    # Load data
    if not data_json:
        raise exceptions.PreventUpdate

    df = pd.read_json(data_json, orient="split")
    df = df.sort_values("Date")
    if df.empty:
        raise exceptions.PreventUpdate

    # Time base (years since first date)
    t_years_hist = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25 * 24 * 3600)

    # Model curve (historical window)
    q_hist_model = arps_rate(t_years_hist, qi, Di, b)

    # Forecast grid: daily steps for smoother curve and reasonable integration
    last_date = df["Date"].iloc[-1]
    total_days = int(round(float(forecast_years) * 365.25))
    if total_days > 0:
        forecast_dates = pd.date_range(last_date + timedelta(days=1), periods=total_days, freq="D")
        t_years_fore = (forecast_dates - df["Date"].iloc[0]).days / 365.25
        q_forecast = arps_rate(t_years_fore, qi, Di, b)
    else:
        forecast_dates = pd.DatetimeIndex([])
        q_forecast = np.array([])

    # Cumulative calculations
    cum_hist = cumulative_trapz(df["Date"], df["OIL"])
    # Use the model rates for forecast integration (starting from last historical point)
    cum_fore = cumulative_trapz(
        np.concatenate([df["Date"].values[-1:], forecast_dates.values]),
        np.concatenate([ [float(q_hist_model[-1])], q_forecast ]) if q_forecast.size > 0 else [float(q_hist_model[-1])]
    ) if q_forecast.size > 0 else 0.0

    cum_total = cum_hist + cum_fore

    # Build figure
    fig = go.Figure()

    # Historical data points
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["OIL"],
        mode="markers",
        name="Historical OIL rate",
        marker=dict(size=5),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Rate: %{y:.2f}<extra></extra>"
    ))

    # Model curve over historical dates
    fig.add_trace(go.Scatter(
        x=df["Date"], y=q_hist_model,
        mode="lines",
        name="Arps model (historical span)",
        line=dict(width=2, dash="solid"),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Model q: %{y:.2f}<extra></extra>"
    ))

    # Forecast curve
    if len(forecast_dates) > 0:
        fig.add_trace(go.Scatter(
            x=forecast_dates, y=q_forecast,
            mode="lines",
            name=f"Forecast (+{forecast_years} yr)",
            line=dict(width=2, dash="dash"),
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Forecast q: %{y:.2f}<extra></extra>"
        ))

    fig.update_layout(
        template=theme,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Oil rate (per day)", rangemode="tozero"),
        hovermode="x unified",
        transition_duration=200,
    )

    # Cumulative box
    cum_children = [
        html.Div([
            html.Div("Cumulative (historical)", style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(f"{cum_hist:,.0f}", style={"fontWeight": 600, "fontSize": "20px"})
        ]),
        html.Div([
            html.Div(f"Cumulative (forecast, {forecast_years} yr)", style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(f"{cum_fore:,.0f}", style={"fontWeight": 600, "fontSize": "20px"})
        ]),
        html.Div([
            html.Div("Cumulative (total)", style={"fontSize": "12px", "opacity": 0.8}),
            html.Div(f"{cum_total:,.0f}", style={"fontWeight": 700, "fontSize": "22px"})
        ]),
        html.Div([
            html.Div("Parameters", style={"fontSize": "12px", "opacity": 0.8, "marginBottom": "6px"}),
            html.Div([
                html.Div(f"qi = {float(qi):,.3f}"),
                html.Div(f"Di = {float(Di):.4f} 1/yr"),
                html.Div(f"b  = {float(b):.3f}"),
            ])
        ])
    ]

    return fig, cum_children


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)
