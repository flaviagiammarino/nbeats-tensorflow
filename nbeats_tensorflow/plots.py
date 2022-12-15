import plotly.graph_objects as go

def plot(df):

    '''
    Plot the actual values of the time series together with the forecasts and backcasts.

    Parameters:
    __________________________________
    df: pd.DataFrame.
        Data frame with the actual values of the time series, forecasts and backcasts.

    Returns:
    __________________________________
    fig: go.Figure.
        Line chart of the actual values of the time series, forecasts and backcasts.
    '''

    layout = dict(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(
            color='#1b1f24',
            size=8,
        ),
        legend=dict(
            font=dict(
                color='#1b1f24',
                size=10,
            ),
        ),
        xaxis=dict(
            title='Time',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
        ),
        yaxis=dict(
            title='Value',
            color='#424a53',
            tickfont=dict(
                color='#6e7781',
                size=6,
            ),
            linecolor='#eaeef2',
            mirror=True,
            showgrid=False,
            zeroline=False,
        ),
    )

    data = []

    data.append(
        go.Scatter(
            x=df['time_idx'],
            y=df['actual'],
            name='Actual',
            mode='lines',
            line=dict(
                color='#afb8c1',
                width=1
            )
        )
    )

    data.append(
        go.Scatter(
            x=df['time_idx'],
            y=df['forecast'],
            name='Forecast',
            mode='lines',
            line=dict(
                color='#0969da',
                width=1
            )
        )
    )

    if 'backcast' in df.columns:

        data.append(
            go.Scatter(
                x=df['time_idx'],
                y=df['backcast'],
                name='Backcast',
                mode='lines',
                line=dict(
                    color='#8250df',
                    width=1
                )
            )
        )

    fig = go.Figure(data=data, layout=layout)

    return fig
