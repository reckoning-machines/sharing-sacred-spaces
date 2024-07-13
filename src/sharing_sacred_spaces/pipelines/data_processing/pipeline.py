from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_fbi_data, preprocess_gini_data, create_dataset


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_fbi_data,
                inputs="fbi_data",
                outputs="preprocessed_fbi_data",
                name="preprocess_fbi_data_node",
            ),
            node(
                func=preprocess_gini_data,
                inputs=[],
                outputs="preprocessed_gini_data",
                name="preprocess_gini_data_node",
            ),
            node(
                func=create_dataset,
                inputs=["preprocessed_fbi_data", "preprocessed_gini_data"],
                outputs="dataset",
                name="create_dataset_node",
            ),
        ]
    )
