import plotly.graph_objects as go

def plot_results(df):

    '''
    Plot the actual and predicted values.

    Parameters:
    __________________________________
    df: pd.DataFrame
        Data frame with actual and predicted values.

    Returns:
    __________________________________
    fig: go.Figure
        Line chart of actual and predicted values.
    '''

    layout = dict(
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(t=10, b=10, l=10, r=10),
        font=dict(
            color='#000000',
            size=10,
        ),
        legend=dict(
            font=dict(
                color='#000000',
            ),
        ),
        xaxis=dict(
            title='Time',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
            mirror=True,
            showgrid=False,
        ),
        yaxis=dict(
            title='Value',
            color='#000000',
            tickfont=dict(
                color='#3a3a3a',
            ),
            linecolor='#d9d9d9',
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
                color='#b3b3b3',
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
                color='#1D6996',
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
                    color='#94346E',
                    width=1
                )
            )
        )

    fig = go.Figure(data=data, layout=layout)

    return fig
