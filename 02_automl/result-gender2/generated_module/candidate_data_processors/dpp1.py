from sagemaker_sklearn_extension.decomposition import RobustPCA
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(
    column_names=[
        'ifa', 'label', 'bundle_vec', 'persona_segment_vec', 'persona_L1_vec',
        'persona_L2_vec', 'persona_L3_vec', 'device_vendor_vec',
        'device_name_vec', 'device_manufacturer_vec', 'device_model_vec',
        'device_year_of_release_vec', 'dev_platform_vec', 'major_os_vec'
    ],
    target_column_name='label'
)


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features contain a relatively small number of unique items.
    categorical = HEADER.as_feature_indices(['dev_platform_vec'])

    # These features can be parsed as natural language.
    text = HEADER.as_feature_indices(
        [
            'ifa', 'bundle_vec', 'persona_segment_vec', 'persona_L1_vec',
            'persona_L2_vec', 'persona_L3_vec', 'device_vendor_vec',
            'device_name_vec', 'device_manufacturer_vec', 'device_model_vec',
            'device_year_of_release_vec', 'major_os_vec'
        ]
    )

    categorical_processors = Pipeline(
        steps=[('thresholdonehotencoder', ThresholdOneHotEncoder(threshold=5))]
    )

    text_processors = Pipeline(
        steps=[
            (
                'multicolumntfidfvectorizer',
                MultiColumnTfidfVectorizer(
                    max_df=0.9365,
                    min_df=0.011235955056179775,
                    analyzer='word',
                    max_features=10000
                )
            )
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ('categorical_processing', categorical_processors,
             categorical), ('text_processing', text_processors, text)
        ]
    )

    return Pipeline(
        steps=[
            ('column_transformer',
             column_transformer), ('robustpca', RobustPCA(n_components=53)),
            ('robuststandardscaler', RobustStandardScaler())
        ]
    )


def build_label_transform():
    """Returns the model definition representing feature processing."""

    return RobustLabelEncoder(
        labels=['0'], fill_label_value='1', include_unseen_class=True
    )
