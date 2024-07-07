from kedro.pipeline import Pipeline, node, pipeline

from .nodes import preprocess_fbi_data


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=preprocess_fbi_data,
                inputs="fbi_data",
                outputs="preprocessed_fbi_data",
                name="preprocess_fbi_data_node",
            ),
        ]
    )
