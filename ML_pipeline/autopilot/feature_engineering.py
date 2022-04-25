from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustStandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Given a list of column names and target column name, Header can return the index
# for given column name
HEADER = Header(column_names=['Label', 'comment'], target_column_name='Label')


def build_feature_transform():
    """ Returns the model definition representing feature processing."""

    # These features can be parsed as natural language.

    text = HEADER.as_feature_indices(['comment'])

    text_processors = Pipeline(
        steps=[
            (
                'multicolumntfidfvectorizer',
                MultiColumnTfidfVectorizer(
                    max_df=0.9645,
                    min_df=0.01020408163265306,
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
        labels=['negative'],
        fill_label_value='positive',
        include_unseen_class=True
    )
