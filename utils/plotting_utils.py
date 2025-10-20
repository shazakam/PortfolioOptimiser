import plotly.express as px
import pandas as pd
import numpy as np

def plot_ratio_over_time(pe_ratio:pd.DataFrame, ratio, title, industry_average):

    pe_ratio_melted = pe_ratio.melt(
        id_vars='Date',
        var_name='Metric',
        value_name= ratio #'P/E'
    )

    pe_fig = px.line(
            pe_ratio_melted,
            x='Date',
            y= ratio, #'P/E',
            color='Metric',     
            markers=True,        
            title= title #'P/E Ratios Over Time'
        )
    
    pe_fig.for_each_trace(
        lambda t: t.update(line=dict(dash='dot')) if t.name == industry_average else ()#'Current Industry Average P/E' else ()
    )

    pe_fig.update_xaxes(
    title_text="Date",
    tickformat="%b %Y",   # e.g. Jan 2024
    showgrid=True,
    tickangle=45          # tilt labels for readability
    )

    return pe_fig