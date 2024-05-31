import pandas as pd
import plotly.express as px


def get_data():
    # Creating dataframes for each dataset
    data_caltech101 = {
        "Guided GradCAM": [1.83, 2.17, 3.04, 0.11, 1.97],
        "LIME": [1.01, 2.53, 3.41, 0.56, 1.88],
        "DeepLift": [1.61, 2.25, 2.95, 0.27, 1.77],
        "Integrated Gradients": [1.64, 2.60, 3.25, 0.24, 1.74],
        "LRP": [1.62, 2.57, 3.14, 0.26, 1.64],
        "GradCAM": [1.01, 2.45, 2.86, 0.55, 2.07],
        "Occlusion": [0.94, 2.58, 3.13, 0.56, 2.36],
    }
    df_caltech101 = pd.DataFrame(
        data_caltech101,
        index=[
            "Complexity",
            "Faithfulness",
            "Localisation",
            "Randomisation",
            "Robustness",
        ],
    )
    # Dictionary of maximum values
    max_values = {
        "Complexity": 2,
        "Faithfulness": 3,
        "Localisation": 5,
        "Randomisation": 1,
        "Robustness": 3,
    }

    # Normalizing the dataframe
    df_caltech101 = df_caltech101.divide(pd.Series(max_values), axis="index")

    data_deepglobe = {
        "Guided GradCAM": [1.77, 3.09, 3.39, 0.14, 2.02],
        "LIME": [0.96, 3.19, 3.33, 0.54, 1.89],
        "DeepLift": [1.55, 3.09, 3.19, 0.4, 1.96],
        "Integrated Gradients": [1.60, 3.19, 3.19, 0.32, 1.95],
        "LRP": [1.62, 3.18, 3.06, 0.29, 1.93],
        "GradCAM": [0.82, 3.19, 3.46, 0.64, 2.19],
        "Occlusion": [0.80, 3.22, 3.32, 0.52, 2.20],
    }
    df_deepglobe = pd.DataFrame(
        data_deepglobe,
        index=[
            "Complexity",
            "Faithfulness",
            "Localisation",
            "Randomisation",
            "Robustness",
        ],
    )
    max_values = {
        "Complexity": 2,
        "Faithfulness": 5,
        "Localisation": 5,
        "Randomisation": 1,
        "Robustness": 3,
    }

    # Normalizing the dataframe
    df_deepglobe = df_deepglobe.divide(pd.Series(max_values), axis="index")

    data_bigearthnet = {
        "Guided GradCAM": [1.78, 2.22, 0, 0.17, 2.3],
        "LIME": [1.14, 2.44, 0, 0.45, 2.31],
        "DeepLift": [1.57, 2.21, 0, 0.29, 2.26],
        "Integrated Gradients": [1.57, 2.39, 0, 0.3, 2.27],
        "LRP": [1.59, 2.27, 0, 0.29, 2.25],
        "GradCAM": [1.04, 2.49, 0, 0.58, 2.34],
        "Occlusion": [0.99, 2.44, 0, 0.57, 2.38],
    }
    df_bigearthnet = pd.DataFrame(
        data_bigearthnet,
        index=[
            "Complexity",
            "Faithfulness",
            "Localisation",
            "Randomisation",
            "Robustness",
        ],
    )

    max_values = {
        "Complexity": 2,
        "Faithfulness": 4,
        "Localisation": 1,
        "Randomisation": 1,
        "Robustness": 3,
    }

    # Normalizing the dataframe
    df_bigearthnet = df_bigearthnet.divide(pd.Series(max_values), axis="index")
    return df_caltech101, df_deepglobe, df_bigearthnet


def main():
    df_caltech101, df_deepglobe, df_bigearthnet = get_data()
    # Adding dataset source to each DataFrame
    df_caltech101["Dataset"] = "Caltech101"
    df_deepglobe["Dataset"] = "DeepGlobe"
    df_bigearthnet["Dataset"] = "BigEarthNet"

    # Concatenating all dataframes into one
    df_combined = pd.concat([df_caltech101, df_deepglobe, df_bigearthnet], axis=0)

    # Melting the DataFrame for easier plotting with Plotly
    df_melted = df_combined.reset_index().melt(
        id_vars=["index", "Dataset"], var_name="Method", value_name="Score"
    )
    df_melted.rename(columns={"index": "Category"}, inplace=True)

    # Plotting each category with Plotly
    for category in df_melted["Category"].unique():
        fig = px.bar(
            df_melted[df_melted["Category"] == category],
            x="Method",
            y="Score",
            color="Dataset",
            barmode="group",
            title=f"Performance of XAI Methods for category: {category}",
        )

        fig.update_layout(
            width=1500,  # Set width to 1000 pixels
            height=1500,  # Set height to 1000 pixels
            font=dict(size=25),  # Increase font size
            xaxis_title="Explanation Methods",
            yaxis_title="Sum of the normalised xAI Metrics",
        )
        save_path = f"new_plots/barplot_{category}.pdf"
        fig.write_image(save_path)
        fig.show()


if __name__ == "__main__":
    main()
