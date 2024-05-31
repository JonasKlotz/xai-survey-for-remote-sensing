import plotly.graph_objects as go
from plotly.subplots import make_subplots

from visualization.plot_metrics.plot_helpers import save_fig

COLORSCALE = "RdBu_r"


def plot_cutmix_thresh_matrices(
    acc_dfs, titles, visualization_save_dir=None, title=None
):
    total_plots = len(acc_dfs)
    cols = 3
    rows = total_plots // cols + 1
    subplot_height = 500  # You can adjust this as needed
    subplot_width = 500  # You can adjust this as needed
    titles = [t.replace("_", " ") for t in titles]
    fig = make_subplots(
        rows=rows,
        cols=cols,
        subplot_titles=titles,
        vertical_spacing=0.1,
        horizontal_spacing=0.1,  # Increase horizontal spacing if needed
    )

    for i, df in enumerate(acc_dfs):
        # Calculate the row and column to place each subplot
        row = (i // cols) + 1
        col = (i % cols) + 1
        index_list = df.index.tolist()
        index_list = [str(i) for i in index_list]
        fig.add_trace(
            go.Heatmap(
                z=df.values,
                x=df.columns,
                y=index_list,
                colorscale="HOT_r",
                zmin=0,
                zmax=1,
                text=df.values,  # Set text to the same values as 'z'
                texttemplate="%{text:.2f}",  # Format the text; adjust the format as needed
                showscale=True,  # Ensures that the color scale is shown
            ),
            row=row,
            col=col,
        )

    # Adjust the size of each subplot cell to be square based on the number of cells
    aspect_ratio = len(df.columns) / float(len(df.index))

    # Set layout size to maintain square cells
    fig_width = cols * subplot_width * aspect_ratio
    fig_height = rows * subplot_height / aspect_ratio
    if not title:
        title = "Accuracy for different CutMix and Segmentation Thresholds"
    fig.update_layout(
        title_text=title,
        height=fig_height,
        width=fig_width,
        title_font_size=25,
    )
    # Updating axis titles
    for i in range(total_plots):
        fig["layout"][f'xaxis{"" if i == 0 else i + 1}'].title = "t map Threshold"
        fig["layout"][f'yaxis{"" if i == 0 else i + 1}'].title = "t cam Threshold"
        # update the font size of the axis titles
        fig["layout"][f'xaxis{"" if i == 0 else i + 1}'].titlefont = dict(size=20)
        fig["layout"][f'yaxis{"" if i == 0 else i + 1}'].titlefont = dict(size=20)
    fig.update_layout(font=dict(size=20))

    fig.show()

    save_fig(fig, title, visualization_save_dir)
