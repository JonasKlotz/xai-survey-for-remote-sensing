import plotly.graph_objects as go


def barplot(values, techniques, dataset_name="BEN"):
    # Sort the data by values
    sorted_indices = sorted(range(len(values)), key=lambda k: values[k], reverse=True)
    sorted_techniques = [techniques[i] for i in sorted_indices]
    sorted_values = [values[i] for i in sorted_indices]

    # Create a bar chart with colors
    fig = go.Figure(
        [
            go.Bar(
                x=sorted_techniques,
                y=sorted_values,
                text=sorted_values,  # Add text to the bars
                textposition="auto",  # Position text in the center of the bars
                marker=dict(
                    color=sorted_values, colorscale="YlOrRd"
                ),  # Applying a yellow-orange-red colorscale
            )
        ]
    )

    # Update layout for better visualization
    title = f"{dataset_name}: Comparison of Explanation Methods"
    fig.update_layout(
        title=title,
        xaxis_title="Explanation Methods",
        yaxis_title="Sum of the Normalised Metrics",
        yaxis=dict(range=[min(sorted_values) - 0.1, max(sorted_values) + 0.1]),
        width=1000,  # Set width to 1000 pixels
        height=1000,  # Set height to 1000 pixels
        font=dict(size=20),  # Increase font size
    )

    # Save the plot as a PDF
    fig.write_image(
        f"/home/jonasklotz/Studys/MASTERS/results_22_4_final/new_plots/barplot_{dataset_name}.pdf"
    )

    # Show the plot (optional)
    fig.show()


if __name__ == "__main__":
    # Data for the plot
    techniques = [
        "Guided GradCAM",
        "LIME",
        "DeepLift",
        "Integrated Gradients",
        "LRP",
        "GradCAM",
        "Occlusion",
    ]
    values = [9.66, 10.21, 9.2, 10, 9.7, 9.75, 10.42]

    data_set = "Caltech101"
    barplot(values, techniques, data_set)

    values = [10.41, 9.91, 10.19, 10.25, 10.08, 10.3, 10.06]
    data_set = "DeepGlobe"
    barplot(values, techniques, data_set)

    values = [6.44, 6.17, 6.29, 6.49, 6.34, 6.44, 6.37]
    data_set = "BEN"
    barplot(values, techniques, data_set)
