import numpy as np
import pandas as pd
from tools import utilities as util
import dash
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import matplotlib.pyplot as plt

# Declare files and read in 27-day averaged data as dictionaries
ACE_file = './data/27_day_avg/ACE_27day_avg.csv'
DSCOVR_file = './data/27_day_avg/DSCOVR_27day_avg.csv'
HELIOS1_file = './data/27_day_avg/HELIOS_1_27day_avg.csv'
HELIOS2_file = './data/27_day_avg/HELIOS_2_27day_avg.csv'
STEREO_A_file = './data/27_day_avg/STEREO_A_27day_avg.csv'
STEREO_B_file = './data/27_day_avg/STEREO_B_27day_avg.csv'
sunspot_file = './data/sunspot/dly_sunspot_27day_avg.csv'
ULYSSES_file = './data/27_day_avg/ULYSSES_27day_avg.csv'
ULYSSES_FULL_file = './data/27_day_avg/ULYSSES_FULL_27day_avg.csv'
WIND_file = './data/27_day_avg/WIND_27day_avg.csv'
Solanki_Vieira_phi_file = './data/Solanki_Vieira_open_flux_fix.csv'

ACE_dict = pd.read_csv(ACE_file).to_dict('list')
DSCOVR_dict = pd.read_csv(DSCOVR_file).to_dict('list')
HELIOS1_dict = pd.read_csv(HELIOS1_file).to_dict('list')
HELIOS2_dict = pd.read_csv(HELIOS2_file).to_dict('list')
STEREO_A_dict = pd.read_csv(STEREO_A_file).to_dict('list')
STEREO_B_dict = pd.read_csv(STEREO_B_file).to_dict('list')
ULYSSES_dict = pd.read_csv(ULYSSES_file).to_dict('list')
ULYSSES_FULL_dict = pd.read_csv(ULYSSES_FULL_file).to_dict('list')
WIND_dict = pd.read_csv(WIND_file).to_dict('list')

sunspot_dict = pd.read_csv(sunspot_file).to_dict('list')
Solanki_Vieira_dict = pd.read_csv(Solanki_Vieira_phi_file).to_dict('list')

data_dict = {**ACE_dict, **DSCOVR_dict, **HELIOS1_dict, **HELIOS2_dict, **STEREO_A_dict, **STEREO_B_dict,
             **ULYSSES_dict, **ULYSSES_FULL_dict, **WIND_dict}

# Add the Cranmer 2017 sunspot vs mdot law, restricting data to years past 1970 (index 1938 in data)
# Plot mdot vs year to see it
cranmer_law_fltyr = [yr for i, yr in enumerate(sunspot_dict.get("Year (SILS)")) if i > 1938]
cranmer_law_mdot = [2.21 * 10 ** 9 * (s + 570) for i, s in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1938]

# Add the sunspot data from file with the same constraints as the Cranmer 2017 power law, index-wise
sn_fltyr = cranmer_law_fltyr
sunspot_num = [sn for i, sn in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1938]

# Add the Solanki-Vieira model data, restricting to years past 1970 (index 98617 in data)
sv_fltyr = [yr for i, yr in enumerate(Solanki_Vieira_dict.get("Year (Solanki-Vieira)")) if i > 98617]
sv_phi = [10**22 * phi
          for i, phi in enumerate(Solanki_Vieira_dict.get("Open Flux [10^-22 Mx] (Solanki-Vieira)")) if i > 98617]
fit_factor = 9 / 6
sv_phi = np.array(sv_phi) * fit_factor

# Create spline fits for the sunspot numbers to fit onto spacecraft data, just pass in the list of data as a list of
# lists if it is not already
# sunspot_fits = util.determine_fits(sunspot_dict.get('Year (SILS)'), [sunspot_dict.get('Sunspot Number (SILS)')])
#
#
# # Create sunspot evaluations for all spacecraft datasets
# ACE_sunspot_num = util.evaluate_fits(ACE_dict.get('Year (ACE)'), sunspot_fits)
# DSCOVR_sunspot_num = util.evaluate_fits(DSCOVR_dict.get('Year (DSCOVR)'), sunspot_fits)
# HELIOS1_sunspot_num = util.evaluate_fits(HELIOS1_dict.get('Year (Helios 1)'), sunspot_fits)
# HELIOS2_sunspot_num = util.evaluate_fits(HELIOS2_dict.get('Year (Helios 2)'), sunspot_fits)
# STEREO_A_sunspot_num = util.evaluate_fits(STEREO_A_dict.get('Year (STEREO A)'), sunspot_fits)
# STEREO_B_sunspot_num = util.evaluate_fits(STEREO_B_dict.get('Year (STEREO B)'), sunspot_fits)
# ULYSSES_sunspot_num = util.evaluate_fits(ULYSSES_dict.get('Year (Ulysses)'), sunspot_fits)
# WIND_sunspot_num = util.evaluate_fits(WIND_dict.get('Year (WIND)'), sunspot_fits)
#
# # Update dictionaries to incorporate sunspot numbers and then save the files
# ACE_dict['Sunspot Number (ACE)'] = ACE_sunspot_num[0]
# DSCOVR_dict['Sunspot Number (DSCOVR)'] = DSCOVR_sunspot_num[0]
# HELIOS1_dict['Sunspot Number (Helios 1)'] = HELIOS1_sunspot_num[0]
# HELIOS2_dict['Sunspot Number (Helios 2)'] = HELIOS2_sunspot_num[0]
# STEREO_A_dict['Sunspot Number (STEREO A)'] = STEREO_A_sunspot_num[0]
# STEREO_B_dict['Sunspot Number (STEREO B)'] = STEREO_B_sunspot_num[0]
# ULYSSES_dict['Sunspot Number (Ulysses)'] = ULYSSES_sunspot_num[0]
# WIND_dict['Sunspot Number (WIND)'] = WIND_sunspot_num[0]
#
# path = './data/27_day_avg/'
#
# pd.DataFrame.from_dict(ACE_dict, orient='columns').to_csv(path + 'ACE_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(DSCOVR_dict, orient='columns').to_csv(path + 'DSCOVR_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(HELIOS1_dict, orient='columns').to_csv(path + 'HELIOS2_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(HELIOS2_dict, orient='columns').to_csv(path + 'HELIOS1_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(STEREO_A_dict, orient='columns').to_csv(path + 'STEREO_A_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(STEREO_B_dict, orient='columns').to_csv(path + 'STEREO_B_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(ULYSSES_dict, orient='columns').to_csv(path + 'ULYSSES_27day_avg.csv', index=False)
# pd.DataFrame.from_dict(WIND_dict, orient='columns').to_csv(path + 'WIND_27day_avg.csv', index=False)

powerlaw = lambda x, amp, index: amp * (x**index)
fitfunc = lambda p, x: p[0] + p[1] * x
errfunc = lambda p, x, y: y - fitfunc(p, x)

# Dash plotting begins here
app = dash.Dash()

app.layout = html.Div(children=[
    html.Div(children='''Choose parameters to plot from drop-down boxes and their graph scales'''),

    # x-values dropdown and graph scale
    html.P([
        dcc.Dropdown(id='x-drop-state',
                 options=[{'label': s, 'value': s} for s in data_dict.keys()],
                 multi=True,
                 value=['Year (ACE)']),

        dcc.RadioItems(
            id='x-graph-scale',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='linear'
        )
    ]),

    # y-values dropdown and graph scale
    html.P([
        dcc.Dropdown(id='y-drop-state',
                     options=[{'label': s, 'value': s} for s in data_dict.keys()],
                     multi=True,
                     value=['Mass Loss Rate [g s^-1] (ACE)']
        ),

        dcc.RadioItems(
            id='y-graph-scale',
            options=[
                {'label': 'Linear', 'value': 'linear'},
                {'label': 'Logarithmic', 'value': 'log'}
            ],
            value='linear'
        )
    ]),

    # Line style options
    html.P([
        html.Div(children='''Choose a line style for the data'''),
        dcc.RadioItems(
            id='line-style',
            options=[
                {'label': 'Lines', 'value': 'lines'},
                {'label': 'Markers', 'value': 'markers'},
                {'label': 'Lines and Markers', 'value': 'lines+markers'}
            ],
            value='markers'
        )
    ]),

    # Fit options
    html.P([
        dcc.Checklist(
            id='fit-checkbox',
            options=[
                {'label': 'Show Fit (Use slider to choose percentile for dispersion)', 'value': 'show-fit'}
            ],
            values=['show-fit']
        ),

        dcc.Slider(
            id='fit-percentile-slider',
            min=1,
            max=99,
            step=1,
            value=95,
            vertical=False,
            marks={25: '25th', 50: '50th', 75: '75th', 90: '90th', 95: '95th'}
        ),
    ]),

    # Overlay plots
    html.P([
        dcc.Checklist(
            id='cranmer-checkbox',
            options=[
                {'label': 'Show Cranmer 2017 Power Law for Mdot', 'value': 'show-cranmer'}
            ],
            values=[]
        ),
        dcc.Checklist(
            id='sunspot-checkbox',
            options=[
                {'label': 'Show sunspot data', 'value': 'show-sunspot'}
            ],
            values=[]
        ),
        dcc.Checklist(
            id='sv-checkbox',
            options=[
                {'label': 'Show Solanki-Vieira 2010 Open Flux Model Data', 'value': 'show-sv'}
            ],
            values=[]
        )
    ]),

    # Colour plot radio buttons
    html.P([
        dcc.RadioItems(
            id='colour-radio',
            options=[
                {'label': 'Colour by spacecraft', 'value': 'spacecraft'},
                {'label': 'Colour by sunspot number', 'value': 'sunspot'},
                {'label': 'Colour by time', 'value': 'time'},
                {'label': 'Colour by radial wind speed', 'value': 'vr'},
                {'label': 'Colour by distance', 'value': 'dist'}
            ],
            value='spacecraft'
        )
    ]),

    html.Button(id='plot-button', children='Plot'),
    html.Div(id='output-graph')
],
    style={'width': '1180'})


@app.callback(
    Output(component_id='output-graph', component_property='children'),
    [Input(component_id='plot-button', component_property='n_clicks')],
    [State(component_id='x-drop-state', component_property='value'),
     State(component_id='x-graph-scale', component_property='value'),
     State(component_id='y-drop-state', component_property='value'),
     State(component_id='y-graph-scale', component_property='value'),
     State(component_id='line-style', component_property='value'),
     State(component_id='fit-checkbox', component_property='values'),
     State(component_id='fit-percentile-slider', component_property='value'),
     State(component_id='cranmer-checkbox', component_property='values'),
     State(component_id='sunspot-checkbox', component_property='values'),
     State(component_id='sv-checkbox', component_property='values'),
     State(component_id='colour-radio', component_property='value')]
)
def plot(n, x_data_names, x_graph_scale, y_data_names, y_graph_scale, line_style, show_fit, percentile, show_cranmer,
         show_sunspot, show_sv, colour_value):
    global data_dict
    show_legend = True  # Used for displaying trace and fit names, don't want to show when doing colour-plotting

    # Standard colour scheme (for 'Colour by spacecraft')
    colours = ['#1f77b4',  # muted blue
               '#ff7f0e',  # safety orange
               '#2ca02c',  # cooked asparagus green
               '#d62728',  # brick red
               '#9467bd',  # muted purple
               '#8c564b',  # chestnut brown
               '#e377c2',  # raspberry yogurt pink
               '#8394de',  # light purple
               '#bcbd22',  # curry yellow-green
               '#17becf']  # blue-teal

    fit_colour = '#000000'  # Black
    dispersion_colour = '#808080'  # Grey

    # List of all Plotly scales to pick from
    scls = ['Blackbody',
            'Bluered',
            'Blues',
            'Earth',
            'Electric',
            'Greens',
            'Greys',
            'Hot',
            'Jet',
            'Picnic',
            'Portland',
            'Rainbow',
            'RdBu',
            'Reds',
            'Viridis',
            'YlGnBu',
            'YlOrRd']

    if len(y_data_names) > 1:  # For multiple y data types in the dropdown
        scl = 'Jet'
        identifier_list = [util.strip_identifier(y, False) for y in y_data_names]

        # Create data traces matching the multiple x and y data types by matching data identifiers (spacecraft names)
        if colour_value == 'sunspot':
            sn_vals = [data_dict.get("Sunspot Number (" + i + ")") for i in identifier_list]
            cmin, cmax = util.get_minmax(sn_vals)
            show_legend = False

            data = [
                {'x': data_dict.get(x), 'y': data_dict.get(y), 'mode': line_style,
                 'name': util.strip_identifier(y, False),
                 'marker': go.scatter.Marker(cmin=cmin, color=sn_vals[i], cmax=cmax,
                                             autocolorscale=False,
                                             reversescale=True, colorscale=scl, colorbar={'title': 'Sunspot Number'})}
                for i, x in enumerate(x_data_names) for y in y_data_names
                if util.strip_identifier(x, False) == util.strip_identifier(y, False)
            ]

        elif colour_value == 'time':
            time_vals = [data_dict.get("Year (" + i + ")") for i in identifier_list]
            cmin, cmax = util.get_minmax(time_vals)
            show_legend = False

            data = [
                {'x': data_dict.get(x), 'y': data_dict.get(y), 'mode': line_style,
                 'name': util.strip_identifier(y, False),
                 'marker': go.scatter.Marker(cmin=cmin, color=time_vals[i], cmax=cmax,
                                             autocolorscale=False,
                                             reversescale=True, colorscale=scl, colorbar={'title': 'Year'})}
                for i, x in enumerate(x_data_names) for y in y_data_names
                if util.strip_identifier(x, False) == util.strip_identifier(y, False)
            ]

        elif colour_value == 'vr':
            vr_vals = [data_dict.get("Radial Wind Velocity [km s^-1] (" + i + ")") for i in identifier_list]
            cmin, cmax = util.get_minmax(vr_vals)
            show_legend = False

            data = [
                {'x': data_dict.get(x), 'y': data_dict.get(y), 'mode': line_style,
                 'name': util.strip_identifier(y, False),
                 'marker': go.scatter.Marker(cmin=cmin, color=vr_vals[i], cmax=cmax,
                                             autocolorscale=False,
                                             reversescale=True, colorscale=scl,
                                             colorbar={'title': 'Radial Wind Velocity [km s^-1]'})}
                for i, x in enumerate(x_data_names) for y in y_data_names
                if util.strip_identifier(x, False) == util.strip_identifier(y, False)
            ]

        elif colour_value == 'dist':
            dist_vals = [data_dict.get("Distance [AU] (" + i + ")") for i in identifier_list]
            cmin, cmax = util.get_minmax(dist_vals)
            show_legend = False

            data = [
                {'x': data_dict.get(x), 'y': data_dict.get(y), 'mode': line_style,
                 'name': util.strip_identifier(y, False),
                 'marker': go.scatter.Marker(cmin=cmin, color=dist_vals[i], cmax=cmax,
                                             autocolorscale=False,
                                             reversescale=True, colorscale=scl, colorbar={'title': 'Distance [AU]'})}
                for i, x in enumerate(x_data_names) for y in y_data_names
                if util.strip_identifier(x, False) == util.strip_identifier(y, False)
            ]

        elif colour_value == 'spacecraft':
            data = [
                {'x': data_dict.get(x), 'y': data_dict.get(y), 'mode': line_style,
                 'name': util.strip_identifier(y, False),
                 'line': {'color': colours[i]}}
                for i, x in enumerate(x_data_names) for y in y_data_names
                if util.strip_identifier(x, False) == util.strip_identifier(y, False)
            ]

        # Create the fits if asked for
        if show_fit:
            # First do the least squares regression fit of everything
            f_exp_list = [util.generate_trend(data_dict.get(x), data_dict.get(y), True)
                          for x in x_data_names for y in y_data_names
                          if util.strip_identifier(x, False) == util.strip_identifier(y, False)]

            for i, x in enumerate(x_data_names):
                for y in y_data_names:
                    if util.strip_identifier(x, False) == util.strip_identifier(y, False):
                        # Generate the dispersion envelopes
                        f_upper, f_lower, disp_gradient = \
                            util.get_dispersion(data_dict.get(x), data_dict.get(y), f_exp_list[i][0], percentile, True)

                        # Add fit to data
                        data.append({'x': data_dict.get(x), 'y': f_exp_list[i][0], 'mode': 'lines',
                                     'name': str(f_exp_list[i][1]),
                                     'text': "Power law is " + str(f_exp_list[i][2]) + "x^" + str(f_exp_list[i][3]),
                                     'line': {'color': fit_colour}})

                        # Add dispersion envelopes to data
                        data.append({'x': data_dict.get(x), 'y': f_upper, 'mode': 'lines',
                                     'text': str(percentile) + "th percentile, y = " + str(disp_gradient) + "x",
                                     'line': {'color': dispersion_colour}, 'showlegend': False})
                        data.append({'x': data_dict.get(x), 'y': f_lower, 'mode': 'lines',
                                     'text': str(percentile) + "th percentile, y = " + str(disp_gradient) + "x",
                                     'line': {'color': dispersion_colour}, 'showlegend': False})

    else:  # For a single y data type in the dropdown
        identifier = util.strip_identifier(x_data_names[0], False)
        x_data = data_dict.get(x_data_names[0])
        y_data = data_dict.get(y_data_names[0])

        # Determine colour scale and create the required data trace
        # scl = [[0, "rgb(5, 10, 172)"], [0.35, "rgb(40, 60, 190)"], [0.5, "rgb(70, 100, 245)"],  # Dummy colour scale
        #        [0.6, "rgb(90, 120, 245)"], [0.7, "rgb(106, 137, 247)"], [1, "rgb(220, 220, 220)"]]
        scl = 'Jet'  # Scale can either be a predefined colour scale or a custom one like above, code auto interps

        if colour_value == 'sunspot':
            sn_val = data_dict.get("Sunspot Number (" + identifier + ")")
            show_legend = False

            data = [
                {'x': x_data, 'y': y_data, 'mode': line_style,
                 'name': util.strip_identifier(y_data_names[0], False),
                 'marker': go.scatter.Marker(cmin=np.min(sn_val), color=sn_val, cmax=np.max(sn_val),
                                             autocolorscale=False,
                                             reversescale=True, colorscale=scl, colorbar={'title': 'Sunspot Number'})}
            ]

        elif colour_value == 'time':
            time_val = data_dict.get("Year (" + identifier + ")")
            show_legend = False

            data = [
                {'x': x_data, 'y': y_data, 'mode': line_style, 'name': util.strip_identifier(y_data_names[0], False),
                 'marker': go.scatter.Marker(cmin=np.min(time_val), color=time_val, cmax=np.max(time_val),
                                             autocolorscale=False, reversescale=True, colorscale=scl,
                                             colorbar={'title': 'Year'})}
            ]
        elif colour_value == 'vr':
            vr_val = data_dict.get("Radial Wind Velocity [km s^-1] (" + identifier + ")")
            show_legend = False

            data = [
                {'x': x_data, 'y': y_data, 'mode': line_style, 'name': util.strip_identifier(y_data_names[0], False),
                 'marker': go.scatter.Marker(cmin=np.min(vr_val), color=vr_val, cmax=np.max(vr_val),
                                             autocolorscale=False, reversescale=False, colorscale=scl,
                                             colorbar={'title': 'Radial Wind Velocity [km s^-1]'})}
            ]
        elif colour_value == 'dist':
            dist_val = data_dict.get("Distance [AU] (" + identifier + ")")
            show_legend = False

            data = [
                {'x': x_data, 'y': y_data, 'mode': line_style, 'name': util.strip_identifier(y_data_names[0], False),
                 'marker': go.scatter.Marker(cmin=np.min(dist_val), color=dist_val, cmax=np.max(dist_val),
                                             autocolorscale=False, reversescale=True, colorscale=scl,
                                             colorbar={'title': 'Distance [AU]'})}
            ]

        elif colour_value == 'spacecraft':
            data = [
                {'x': x_data, 'y': y_data, 'mode': line_style, 'name': util.strip_identifier(y_data_names[0], False),
                 'line': {'color': colours[0]}}
            ]

        # Add the fit to the data if asked
        if show_fit:
            # Create the fit
            # First the least squares regression
            f_exp, ks, amp, index = \
                util.generate_trend(x_data, y_data, True)
            # Then the dispersion of the least squares regression
            f_upper, f_lower, disp_gradient = util.get_dispersion(x_data, y_data, f_exp, percentile, True)

            # First do the normal least squares regression of entire dataset
            data.append(
                {'x': x_data, 'y': f_exp, 'name': str(ks), 'mode': 'lines',
                 'text': str(percentile) + "th percentile, power law is " + str(amp) + "x^" + str(index),
                 'line': {'color': fit_colour}}
            )
            data.append({'x': x_data, 'y': f_upper, 'line': {'color': dispersion_colour},
                         'text': str(percentile) + "th percentile, y = " + str(disp_gradient) + "x",
                         'showlegend': False})
            data.append({'x': x_data, 'y': f_lower, 'line': {'color': dispersion_colour},
                         'text': str(percentile) + "th percentile, y = " + str(disp_gradient) + "x",
                         'showlegend': False})

    # Add the Cranmer power law if asked
    if show_cranmer:
        data.append(
            {'x': cranmer_law_fltyr, 'y': cranmer_law_mdot, 'mode': 'lines',
             'name': "Cranmer 2017 Power Law (SN vs Mdot)", 'line': {'color': '#395248'}}
        )

    # Add sunspot data if asked
    if show_sunspot:
        data.append(
            {'x': sn_fltyr, 'y': sunspot_num, 'mode': 'lines',
             'name': "Sunspot Number (SILS)", 'line': {'color': '#eec900'}}
        )

    # Add the Solanki-Vieira data if asked
    if show_sv:
        data.append(
            {'x': sv_fltyr, 'y': sv_phi, 'mode': 'lines', 'name': "Solanki-Vieira 2010 Open Flux Model",
             'line': {'color': '#dka730'}}
        )

    return dcc.Graph(
        id='graph1',
        figure={
            'data': data,
            'layout': go.Layout(
                    xaxis=dict(
                        type=x_graph_scale,
                        autorange=True,
                        showgrid=False,
                        title=util.strip_identifier(x_data_names[0], True),
                        exponentformat='power'
                    ),
                    yaxis=dict(
                        type=y_graph_scale,
                        autorange=True,
                        showgrid=False,
                        title=util.strip_identifier(y_data_names[0], True),
                        exponentformat='power',
                        showexponent='all'
                    ),
                    showlegend=show_legend,
            )
        }
    )


if __name__ == "__main__":
    app.run_server()


# Pyplot plotting down here

# phi_ace = np.array(data_dict.get("Open Flux [Mx] (ACE)"))
# sn_ace = np.array(data_dict.get("Sunspot Number (ACE)"))
#
# time_ace = np.array(data_dict.get("Year (ACE)"))
# mdot_ace = np.array(data_dict.get("Mass Loss Rate [g s^-1] (ACE)"))
#
# time_dscovr = np.array(data_dict.get("Year (DSCOVR)"))
# mdot_dscovr = np.array(data_dict.get("Mass Loss Rate [g s^-1] (DSCOVR)"))
#
# time_helios1 = np.array(data_dict.get("Year (Helios 1)"))
# mdot_helios1 = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Helios 1)"))
#
# time_helios2 = np.array(data_dict.get("Year (Helios 2)"))
# mdot_helios2 = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Helios 2)"))
#
# time_stereoa = np.array(data_dict.get("Year (STEREO A)"))
# mdot_stereoa = np.array(data_dict.get("Mass Loss Rate [g s^-1] (STEREO A)"))
#
# time_stereob = np.array(data_dict.get("Year (STEREO B)"))
# mdot_stereob = np.array(data_dict.get("Mass Loss Rate [g s^-1] (STEREO B)"))
#
# time_ulysses = np.array(data_dict.get("Year (Ulysses)"))
# mdot_ulysses = np.array(data_dict.get("Mass Loss Rate [g s^-1] (Ulysses)"))
#
# time_wind = np.array(data_dict.get("Year (WIND)"))
# mdot_wind = np.array(data_dict.get("Mass Loss Rate [g s^-1] (WIND)"))
#
# # Limit data to 1970 and beyond
# time_sils = [yr for i, yr in enumerate(sunspot_dict.get("Year (SILS)")) if i > 1937]
# sn_sils = [sn for i, sn in enumerate(sunspot_dict.get("Sunspot Number (SILS)")) if i > 1937]
#
# time_sils = np.array(time_sils)
# sn_sils = np.array(sn_sils)
#
#
# # mdot_phi_fit, ks, amp, index = \
# #                 util.generate_trend(phi_ace, mdot_ace, True)
#
# # dispersion_upper = mdot_ace - mdot_phi_fit
# # p = np.percentile(dispersion_upper, 95)
# # idx = int(0.95 * len(dispersion_upper) - 1)
# # x = sorted(phi_ace)[idx]
# # m = p / x  # Find gradient
# # disp_line = m * np.array(phi_ace)
# # f_upper = mdot_phi_fit + disp_line
# # f_lower = mdot_phi_fit - disp_line
#
# # f_upper, f_lower = util.get_dispersion(phi_ace, mdot_ace, mdot_phi_fit)
# #
# # # plt.plot(phi_ace, dispersion_upper, 'r.')
# # plt.plot(phi_ace, mdot_ace, 'k.')
# # plt.plot(phi_ace, mdot_phi_fit, 'b.')
# # # plt.plot(phi_ace, disp_line, 'g-')
# # plt.plot(phi_ace, f_upper, 'm-')
# # plt.plot(phi_ace, f_lower, 'c-')
# # plt.axhline(y=0)
# #
# # plt.show()
#
# plt.figure()
# # Define the mdot vs phi fit from ACE as an unrefined law
# mdot_phi_law_ace = 0.13 * (phi_ace ** 0.568)
# disp_phi_line_ace = 6.71 * 10 ** (-12) * phi_ace
#
# f_upper_phi = mdot_phi_law_ace + disp_phi_line_ace
# f_lower_phi = mdot_phi_law_ace - disp_phi_line_ace
#
# # Define the mdot vs sn fit from ACE as an unrefined law, using sunspot numbers and times from SILS
# mdot_sn_law_ace = 9.9 * 10**11 * (sn_sils**0.075) + 10**11 # Added factor is so mdot does not go to zero when sn goes to zero
# disp_sn_line_ace = 4.03 * 10**9 * sn_sils
#
# f_upper_sn = mdot_sn_law_ace + disp_sn_line_ace
# f_lower_sn = mdot_sn_law_ace - disp_sn_line_ace
#
# # Define the mdot vs phi fit from ACE using Solanki-Vieira open fluxes
# mdot_phi_law_sv = 0.13 * (np.array(sv_phi) ** 0.568)
# disp_phi_line_sv = 6.71 * 10**(-12) * np.array(sv_phi)
#
# f_upper_phi_sv = mdot_phi_law_sv + disp_phi_line_sv
# f_lower_phi_sv = mdot_phi_law_sv - disp_phi_line_sv
#
# # Plot data points
# plt.plot(time_ace, mdot_ace, 'k.', zorder=2)
# plt.plot(time_dscovr, mdot_dscovr, 'k.', zorder=2)
# plt.plot(time_helios1, mdot_helios1, 'k.', zorder=2)
# plt.plot(time_helios2, mdot_helios2, 'k.', zorder=2)
# plt.plot(time_stereoa, mdot_stereoa, 'k.', zorder=2)
# plt.plot(time_stereob, mdot_stereob, 'k.', zorder=2)
# plt.plot(time_ulysses, mdot_ulysses, 'k.', zorder=2)
# plt.plot(time_wind, mdot_wind, 'k.', zorder=2)
#
# # Plot ACE phi model and dispersion lines
# # plt.plot(time_ace, mdot_phi_law_ace, 'r-', label='ACE Open Flux Model', zorder=0)
# # plt.fill_between(time_ace, f_upper_phi, f_lower_phi, color='lightcoral', alpha=0.5, zorder=0)
# # plt.plot(time_ace, f_upper_phi, color='orange', ls='-')
# # plt.plot(time_ace, f_lower_phi, color='orange', ls='-')
#
# # Plot ACE phi model and dispersion lines using Solanki-Vieira times and data
# # plt.plot(sv_fltyr, mdot_phi_law_sv, 'g-', label="ACE (Solanki-Vieira 2010) Open Flux Model ("
# #                                                 + str(fit_factor) + " scaling)", zorder=0)
# # plt.fill_between(sv_fltyr, f_upper_phi_sv, f_lower_phi_sv, color='lime', alpha=0.5, zorder=0)
# # plt.plot(sv_fltyr, f_upper_phi_sv, color='lime', ls='-')
# # plt.plot(sv_fltyr, f_lower_phi_sv, color='lime', ls='-')
#
# # Plot ACE SN model and dispersion lines
# plt.plot(time_sils, mdot_sn_law_ace, 'b-', label='ACE Sunspot Number Model', zorder=0)
# plt.fill_between(time_sils, f_upper_sn, f_lower_sn, color='cyan', alpha=0.5, zorder=0)
# # plt.plot(time_ace, f_upper_sn, 'c-')
# # plt.plot(time_ace, f_lower_sn, 'c-')
#
# # Cranmer power law
# plt.plot(cranmer_law_fltyr, cranmer_law_mdot, color='magenta', ls='-', label='Cranmer 2017 Power Law', zorder=0)
#
# plt.xlabel("Year")
# plt.ylabel(r"Mass Loss Rate [g s$^{-1}$]")
# plt.legend(bbox_to_anchor=(0.35, -0.3), loc=6, ncol=2, mode="expand", borderaxespad=0., frameon=False)
# plt.tight_layout(pad=7)
#
# plt.show()
