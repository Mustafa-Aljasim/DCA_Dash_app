# app.py
import base64
import io
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from dash import Dash, dcc, html, Input, Output, State, exceptions, ctx
import plotly.graph_objects as go

# =========================================================
# Arps helpers
# =========================================================
def arps_rate(t_years, qi, Di, b):
    """
    Arps decline: q(t) = qi / (1 + b*Di*t)^(1/b), exponential if b -> 0.
    t_years in years, Di in 1/yr.
    """
    t_years = np.asarray(t_years, dtype=float)
    qi = float(qi); Di = float(Di); b = float(b)
    if np.isclose(b, 0.0):
        return qi * np.exp(-Di * t_years)
    denom = np.clip(1.0 + b * Di * t_years, 1e-12, np.inf)
    return qi / (denom ** (1.0 / b))

def cumulative_trapz(dates, rates):
    """
    Trapezoidal integration for irregular time steps.
    dates: datetime-like array; rates: same length (per-day units).
    Returns cumulative in (rate_unit * days).
    """
    if len(dates) < 2:
        return 0.0
    d = pd.to_datetime(pd.Series(dates), utc=False).reset_index(drop=True)
    r = np.asarray(rates, dtype=float)
    cum = 0.0
    for i in range(1, len(d)):
        dt_days = (d.iloc[i] - d.iloc[i - 1]).total_seconds() / 86400.0
        cum += 0.5 * (r[i] + r[i - 1]) * dt_days
    return float(cum)

def running_cumulative(dates, rates):
    """Running cumulative aligned with dates using trapezoids."""
    if len(dates) == 0:
        return np.array([])
    d = pd.to_datetime(pd.Series(dates), utc=False).reset_index(drop=True)
    r = np.asarray(rates, dtype=float)
    out = np.zeros(len(d), dtype=float)
    for i in range(1, len(d)):
        dt_days = (d.iloc[i] - d.iloc[i - 1]).total_seconds() / 86400.0
        out[i] = out[i - 1] + 0.5 * (r[i] + r[i - 1]) * dt_days
    return out

# =========================================================
# Sample data (used if no upload)
# =========================================================
def make_sample_data():
    start = pd.Timestamp("2020-01-01")
    n_days = int(3 * 365.25)
    dates = pd.date_range(start, periods=n_days, freq="D")
    t_years = (dates - dates[0]).days / 365.25
    qi, Di, b = 1200.0, 0.85, 0.7
    q = arps_rate(t_years, qi, Di, b)
    np.random.seed(1)
    q_noisy = np.clip(q + np.random.normal(scale=0.03 * q.max(), size=q.shape), 0, None)
    return pd.DataFrame({"Date": dates, "OIL": q_noisy})

DEFAULT_DF = make_sample_data()

# =========================================================
# CSV parsing
# =========================================================
def parse_contents(contents):
    content_type, content_string = contents.split(",")
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
    except Exception:
        df = pd.read_csv(io.StringIO(decoded.decode("latin-1")))
    # normalize columns
    cols = {c.lower().strip(): c for c in df.columns}
    date_col = next((cols[c] for c in cols if "date" in c), None)
    oil_col = next((cols[c] for c in cols if c in ("oil","oil_rate","rate","qo","q")), None)
    if date_col is None or oil_col is None:
        raise ValueError("CSV must include 'Date' and 'OIL' (case-insensitive).")
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[date_col], errors="coerce"),
        "OIL": pd.to_numeric(df[oil_col], errors="coerce")
    }).dropna().sort_values("Date")
    return out.reset_index(drop=True)

# =========================================================
# App layout
# =========================================================
app = Dash(__name__)
app.title = "Arps DCA – Stable with Plot Selection"

app.layout = html.Div(style={"maxWidth": "1150px", "margin": "auto", "padding": "16px"}, children=[
    html.H2("Decline Curve Analysis (Arps)"),

    # Top controls
    html.Div(style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "center"}, children=[
        dcc.RadioItems(
            id="theme",
            options=[{"label": "Light", "value": "plotly_white"},
                     {"label": "Dark", "value": "plotly_dark"}],
            value="plotly_white",
            labelStyle={"marginRight": "12px"}
        ),
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag & Drop or ", html.A("Select CSV")]),
            style={"width": "240px", "lineHeight": "38px", "borderWidth": "1px", "borderStyle": "dashed",
                   "borderRadius": "8px", "textAlign": "center", "cursor": "pointer"},
            multiple=False
        ),
        html.Div(id="file-name", style={"fontSize": "12px", "opacity": 0.7}),
        html.Button("Fit Model", id="fit-btn", n_clicks=0, title="Fit on selected range if any; otherwise full history"),
        html.Button("Reset Fit to Full Data", id="reset-fit", n_clicks=0),
        html.Span(id="fit-status", style={"marginLeft": "8px", "fontSize": "12px", "opacity": 0.85}),
    ]),

    html.Div("Tip: Use the box or lasso tool in the top-right of the rate plot to pick a time window for regression. "
             "Click 'Fit Model' to apply. Forecast always starts at the last historical date.",
             style={"fontSize": "12px", "opacity": 0.8, "marginTop": "6px"}),

    html.Hr(),

    # Parameter / forecast controls
    html.Div(style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(240px, 1fr))", "gap": "12px"}, children=[
        html.Div([html.Label("qi (initial rate, e.g., STB/day)"),
                  dcc.Input(id="qi", type="number", value=1200.0, step="any", min=0, style={"width": "100%"})]),
        html.Div([html.Label("Di (1/yr)"),
                  dcc.Input(id="Di", type="number", value=0.8, step="any", min=0, style={"width": "100%"})]),
        html.Div([html.Label("b (hyperbolic exponent)"),
                  dcc.Input(id="b", type="number", value=0.7, step=0.01, min=0, max=2.0, style={"width": "100%"})]),
        html.Div([html.Label("Forecast period (years)"),
                  dcc.Slider(id="forecast-years", min=0, max=30, step=0.5, value=10,
                             marks={0: "0", 5: "5", 10: "10", 20: "20", 30: "30"})]),
        html.Div([html.Label("Forecast step"),
                  dcc.Dropdown(id="forecast-step",
                               options=[{"label": "Daily", "value": "D"},
                                        {"label": "Weekly", "value": "W"},
                                        {"label": "Monthly", "value": "M"}],
                               value="D", clearable=False)]),
        html.Div([html.Label("Y-axis scale"),
                  dcc.RadioItems(id="y-axis-scale",
                                 options=[{"label": "Linear", "value": "linear"},
                                          {"label": "Log", "value": "log"}],
                                 value="linear",
                                 labelStyle={"marginRight": "12px"})]),
        html.Div([html.Label("Selection status"),
                  html.Div(id="selection-label", style={"fontSize": "12px", "opacity": 0.9})]),
        html.Button("Download Forecast CSV", id="download-btn", n_clicks=0, style={"alignSelf": "end"}),
        dcc.Download(id="download-forecast"),
    ]),

    html.Hr(),

    # Graphs
    dcc.Graph(
        id="dca-graph",
        style={"height": "540px"},
        config={"modeBarButtonsToAdd": ["select2d", "lasso2d"], "displaylogo": False},
    ),
    dcc.Graph(id="cum-graph", style={"height": "380px"}),

    # Cumulative summary
    html.Div(id="cum-box",
             style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(220px, 1fr))",
                    "gap": "12px", "marginTop": "8px", "padding": "12px",
                    "border": "1px solid rgba(128,128,128,0.25)", "borderRadius": "8px"}),

    # Stores
    dcc.Store(id="data-store"),
    dcc.Store(id="fitted-params"),
    dcc.Store(id="forecast-data"),
    dcc.Store(id="selection-store"),  # holds selection window (start/end)
])

# =========================================================
# Callbacks
# =========================================================
@app.callback(
    Output("data-store", "data"),
    Output("file-name", "children"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=False
)
def handle_upload(contents, filename):
    if contents is None:
        df = DEFAULT_DF.copy()
        return df.to_json(date_format="iso", orient="split"), "Using sample data"
    try:
        df = parse_contents(contents)
        return df.to_json(date_format="iso", orient="split"), f"Loaded: {filename}"
    except Exception as e:
        df = DEFAULT_DF.copy()
        return df.to_json(date_format="iso", orient="split"), f"Upload error; using sample data. ({e})"

# Capture selection safely (only historical points)
@app.callback(
    Output("selection-store", "data"),
    Output("selection-label", "children"),
    Input("dca-graph", "selectedData"),
    Input("reset-fit", "n_clicks"),
    State("data-store", "data"),
    prevent_initial_call=True
)
def update_selection(selected, reset_clicks, data_json):
    df = pd.read_json(data_json, orient="split")
    trigger = ctx.triggered_id
    if trigger == "reset-fit":
        return None, "Using full dataset for fit."
    # If selection provided, filter only points from the historical trace (curveNumber 0)
    if selected and isinstance(selected, dict) and "points" in selected and selected["points"]:
        xs = [p["x"] for p in selected["points"] if p.get("curveNumber", 0) == 0]
        if not xs:
            # nothing selected from historical markers
            return None, "No historical points selected; using full dataset for fit."
        try:
            start = pd.to_datetime(min(xs))
            end = pd.to_datetime(max(xs))
        except Exception:
            return None, "Selection not understood; using full dataset for fit."
        start = max(start, df["Date"].min())
        end = min(end, df["Date"].max())
        npts = len(xs)
        return {"start": start.isoformat(), "end": end.isoformat(), "n": npts}, f"Selected {npts} pts: {start.date()} → {end.date()}"
    # If selection cleared (clicked outside), keep previous selection unchanged
    raise exceptions.PreventUpdate

# Fit button: run regression on selected window (if any) else full history
@app.callback(
    Output("fitted-params", "data"),
    Output("fit-status", "children"),
    Input("fit-btn", "n_clicks"),
    State("data-store", "data"),
    State("selection-store", "data"),
    prevent_initial_call=True
)
def fit_model(n_clicks, data_json, selection):
    df_all = pd.read_json(data_json, orient="split").sort_values("Date")
    # choose fit df
    if selection:
        start = pd.to_datetime(selection["start"])
        end = pd.to_datetime(selection["end"])
        df = df_all[(df_all["Date"] >= start) & (df_all["Date"] <= end)].copy()
        note = f"Fit complete (selection: {start.date()} → {end.date()})."
    else:
        df = df_all.copy()
        note = "Fit complete (full history)."

    if df.empty or df["OIL"].dropna().le(0).sum() == len(df):
        return None, "No positive-rate points to fit."

    t_years = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25 * 24 * 3600)
    y = df["OIL"].astype(float).values
    mask = np.isfinite(y) & (y > 0)
    if mask.sum() < 5:
        return None, "Not enough positive points for fitting."

    qi0 = float(np.nanmax(y[mask]))
    Di0 = 0.5
    b0 = 0.7
    bounds = ([1e-8, 1e-6, 0.0], [1e9, 5.0, 2.0])

    try:
        popt, _ = curve_fit(arps_rate, t_years[mask], y[mask], p0=[qi0, Di0, b0],
                            bounds=bounds, maxfev=20000)
        qi_opt, Di_opt, b_opt = [float(x) for x in popt]
        return {"qi": qi_opt, "Di": Di_opt, "b": b_opt}, note
    except Exception as e:
        return None, f"Fit failed: {e}"

# Push fitted params to inputs
@app.callback(
    Output("qi", "value"),
    Output("Di", "value"),
    Output("b", "value"),
    Input("fitted-params", "data"),
    State("qi", "value"), State("Di", "value"), State("b", "value"),
    prevent_initial_call=True
)
def update_inputs_from_fit(params, qi_val, Di_val, b_val):
    if not params:
        raise exceptions.PreventUpdate
    return params.get("qi", qi_val), params.get("Di", Di_val), params.get("b", b_val)

# Build figures + cumulative + forecast table (always forecasts from last historical date)
@app.callback(
    Output("dca-graph", "figure"),
    Output("cum-graph", "figure"),
    Output("cum-box", "children"),
    Output("forecast-data", "data"),
    Input("data-store", "data"),
    Input("qi", "value"), Input("Di", "value"), Input("b", "value"),
    Input("forecast-years", "value"), Input("forecast-step", "value"),
    Input("theme", "value"), Input("y-axis-scale", "value"),
)
def update_graphs(data_json, qi, Di, b, forecast_years, step, theme, yaxis_scale):
    df = pd.read_json(data_json, orient="split").sort_values("Date")
    if df.empty:
        raise exceptions.PreventUpdate

    # Model over historical dates for visual comparison
    t_hist_years = (df["Date"] - df["Date"].iloc[0]).dt.total_seconds() / (365.25 * 24 * 3600)
    q_hist_model = arps_rate(t_hist_years, qi, Di, b)

    # Forecast grid (start strictly after last historical date)
    last_date = df["Date"].iloc[-1]
    freq = {"D": "D", "W": "W", "M": "MS"}[step]
    step_days = {"D": 1.0, "W": 7.0, "M": 30.4375}[step]
    periods = int(round(float(forecast_years) * (365.25 / step_days))) if float(forecast_years) > 0 else 0
    if periods > 0:
        # start at next period after last_date
        start_date = last_date + pd.Timedelta(days=1)
        forecast_dates = pd.date_range(start_date, periods=periods, freq=freq)
        t_fore_years = (forecast_dates - df["Date"].iloc[0]).days / 365.25
        q_fore = arps_rate(t_fore_years, qi, Di, b)
    else:
        forecast_dates = pd.DatetimeIndex([])
        q_fore = np.array([])

    # Cumulatives
    cum_hist_total = cumulative_trapz(df["Date"], df["OIL"])               # actual historical
    cum_fore_total = cumulative_trapz(
        np.concatenate([[df["Date"].iloc[-1]], forecast_dates.values]) if len(forecast_dates) else df["Date"].iloc[-1:],
        np.concatenate([[float(q_hist_model[-1])], q_fore]) if len(q_fore) else [float(q_hist_model[-1])]
    ) if len(q_fore) else 0.0
    cum_total = cum_hist_total + cum_fore_total

    # Combined cumulative series for the second plot (actual hist + model forecast)
    comb_dates = list(df["Date"].values) + list(forecast_dates.values)
    comb_rates = list(df["OIL"].astype(float).values) + list(q_fore)
    comb_cum = running_cumulative(comb_dates, comb_rates)

    # Forecast hover cumulative slice
    forecast_cum_for_hover = np.array([])
    if len(forecast_dates) > 0:
        forecast_cum_for_hover = comb_cum[len(df):]

    # Main Rate Graph (historical markers; model line; forecast dashed)
    fig_rate = go.Figure()
    # Trace 0: Historical (selectable)
    fig_rate.add_trace(go.Scatter(
        x=df["Date"], y=df["OIL"], mode="markers", name="Historical",
        marker=dict(size=5),
        selected=dict(marker=dict(size=7)),
        unselected=dict(marker=dict(opacity=0.55)),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Rate: %{y:.2f}<extra></extra>"
    ))
    # Trace 1: Model on historical dates
    fig_rate.add_trace(go.Scatter(
        x=df["Date"], y=q_hist_model, mode="lines", name="Model (hist)",
        line=dict(width=2),
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Model q: %{y:.2f}<extra></extra>"
    ))
    # Trace 2: Forecast
    if len(forecast_dates) > 0:
        fig_rate.add_trace(go.Scatter(
            x=forecast_dates, y=q_fore, mode="lines", name=f"Forecast (+{forecast_years} yr)",
            line=dict(width=2, dash="dash"),
            customdata=forecast_cum_for_hover,
            hovertemplate="Date: %{x|%Y-%m-%d}<br>Rate: %{y:.2f}<br>Cum: %{customdata:,.0f}<extra></extra>"
        ))
    fig_rate.update_layout(
        template=theme,
        yaxis_type=yaxis_scale,
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Oil rate (per day)", rangemode="tozero"),
        hovermode="x unified",
        dragmode="select"  # enables selection by default
    )

    # Cumulative vs Time Graph
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=pd.to_datetime(comb_dates), y=comb_cum, mode="lines", name="Cumulative (hist+forecast)",
        hovertemplate="Date: %{x|%Y-%m-%d}<br>Cum: %{y:,.0f}<extra></extra>"
    ))
    fig_cum.update_layout(
        template=theme,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis=dict(title="Date"),
        yaxis=dict(title="Cumulative Production"),
        hovermode="x unified"
    )

    # Cum summary box
    cum_children = [
        html.Div([html.Div("Cumulative (historical)", style={"fontSize": "12px", "opacity": 0.8}),
                  html.Div(f"{cum_hist_total:,.0f}", style={"fontWeight": 600, "fontSize": "20px"})]),
        html.Div([html.Div(f"Cumulative (forecast, {forecast_years} yr)", style={"fontSize": "12px", "opacity": 0.8}),
                  html.Div(f"{cum_fore_total:,.0f}", style={"fontWeight": 600, "fontSize": "20px"})]),
        html.Div([html.Div("Cumulative (total)", style={"fontSize": "12px", "opacity": 0.8}),
                  html.Div(f"{cum_total:,.0f}", style={"fontWeight": 700, "fontSize": "22px"})]),
        html.Div([html.Div("Parameters", style={"fontSize": "12px", "opacity": 0.8, "marginBottom": "6px"}),
                  html.Div([html.Div(f"qi = {float(qi):,.3f}"),
                            html.Div(f"Di = {float(Di):.4f} 1/yr"),
                            html.Div(f"b  = {float(b):.3f}")])])
    ]

    # Forecast table for CSV: Date, Rate, Cum_Total at forecast dates
    if len(forecast_dates) > 0:
        forecast_df = pd.DataFrame({
            "Date": pd.to_datetime(forecast_dates),
            "Rate": q_fore,
            "Cum_Total": forecast_cum_for_hover
        })
        forecast_json = forecast_df.to_json(date_format="iso", orient="split")
    else:
        forecast_json = pd.DataFrame(columns=["Date", "Rate", "Cum_Total"]).to_json(date_format="iso", orient="split")

    return fig_rate, fig_cum, cum_children, forecast_json

# CSV Download
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

# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
