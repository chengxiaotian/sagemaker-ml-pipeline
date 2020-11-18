from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(column_names=['label', 'features'], target_column_name='label')


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as natural language.
    text = HEADER.as_feature_indices(['features'])

    text_processors = Pipeline(
        steps=[
            (
                'multicolumntfidfvectorizer',
                MultiColumnTfidfVectorizer(
                    max_df=0.9684,
                    min_df=0.013108614232209739,
                    analyzer='word',
                    max_features=10000
                )
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[('text_processing', text_processors, text)]
    )

    return Pipeline(
        steps=[
            ('column_transformer', column_transformer
            ), ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(
        labels=['1'], fill_label_value='0', include_unseen_class=True
    )
