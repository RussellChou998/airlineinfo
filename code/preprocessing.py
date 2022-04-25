import numpy as np
import pandas as pd
import string
import re
from sklearn.utils import resample

df = pd.read_csv( f"airline-reviews.csv")
df = df[df['title'].notna()] # drop rows where title text is missing



# Remove 'Verified' text before the first pipe '|' operator for each review 
cleaned_comments = [re.sub(r'^.*?\| ','', record) for record in df['comment'].values]

# ensure the same
assert len(cleaned_comments) == df.shape[0], 'The length of the cleaned comments does not match the original DataFrame length'

# replace comments with cleaned comments
df['comment'] = cleaned_comments




