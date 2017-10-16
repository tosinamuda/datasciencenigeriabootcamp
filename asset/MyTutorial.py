from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(MyDataset, test_size=0.2, random_state=42)

MyDataset["median_income"].hist()

# Divide by 1.5 to limit the number of income categories
MyDataset["income_cat"] = np.ceil(MyDataset["median_income"] / 1.5)
# Label those above 5 as 5
MyDataset["income_cat"].where(MyDataset["income_cat"] < 5, 5.0, inplace=True)

MyDataset["income_cat"].value_counts()
MyDataset["income_cat"].hist()


from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(MyDataset, MyDataset["income_cat"]):
    strat_train_set = MyDataset.loc[train_index]
    strat_test_set = MyDataset.loc[test_index]


MyDataset["income_cat"].value_counts() / len(MyDataset)


for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

MyDataset["rooms_per_household"] = MyDataset["total_rooms"]/MyDataset["households"]
MyDataset["bedrooms_per_room"] = MyDataset["total_bedrooms"]/MyDataset["total_rooms"]
MyDataset["population_per_household"]=MyDataset["population"]/MyDataset["households"]




median = MyDataset["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows



from sklearn.preprocessing import Imputer
imputer = Imputer(strategy="median")


MyDataset_num = MyDataset.drop("ocean_proximity", axis=1)
imputer.fit(MyDataset_num)
imputer.statistics_

MyDataset_num.median().values


# Transform the training set:
X = imputer.transform(MyDataset_num)

MyDataset_tr = pd.DataFrame(X, columns=MyDataset_num.columns,
                          index = list(MyDataset.index.values))

MyDataset_tr.loc[sample_incomplete_rows.index.values]


imputer.strategy




MyDataset_tr = pd.DataFrame(X, columns=MyDataset_num.columns)
MyDataset_tr.head()


# Now let's preprocess the categorical input feature, `ocean_proximity`:



MyDataset_cat = MyDataset["ocean_proximity"]
MyDataset_cat.head(10)


MyDataset_cat_encoded, MyDataset_categories = MyDataset_cat.factorize()
MyDataset_cat_encoded[:10]





MyDataset_categories




from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
MyDataset_cat_1hot = encoder.fit_transform(MyDataset_cat_encoded.reshape(-1,1))
MyDataset_cat_1hot


MyDataset_cat_1hot.toarray()


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.preprocessing import LabelEncoder
from scipy import sparse

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, encoding='onehot', categories='auto', dtype=np.float64,
                 handle_unknown='error'):
        self.encoding = encoding
        self.categories = categories
        self.dtype = dtype
        self.handle_unknown = handle_unknown

    def fit(self, X, y=None):


        if self.encoding not in ['onehot', 'onehot-dense', 'ordinal']:
            template = ("encoding should be either 'onehot', 'onehot-dense' "
                        "or 'ordinal', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.handle_unknown not in ['error', 'ignore']:
            template = ("handle_unknown should be either 'error' or "
                        "'ignore', got %s")
            raise ValueError(template % self.handle_unknown)

        if self.encoding == 'ordinal' and self.handle_unknown == 'ignore':
            raise ValueError("handle_unknown='ignore' is not supported for"
                             " encoding='ordinal'")

        X = check_array(X, dtype=np.object, accept_sparse='csc', copy=True)
        n_samples, n_features = X.shape

        self._label_encoders_ = [LabelEncoder() for _ in range(n_features)]

        for i in range(n_features):
            le = self._label_encoders_[i]
            Xi = X[:, i]
            if self.categories == 'auto':
                le.fit(Xi)
            else:
                valid_mask = np.in1d(Xi, self.categories[i])
                if not np.all(valid_mask):
                    if self.handle_unknown == 'error':
                        diff = np.unique(Xi[~valid_mask])
                        msg = ("Found unknown categories {0} in column {1}"
                               " during fit".format(diff, i))
                        raise ValueError(msg)
                le.classes_ = np.array(np.sort(self.categories[i]))

        self.categories_ = [le.classes_ for le in self._label_encoders_]

        return self

    def transform(self, X):
        X = check_array(X, accept_sparse='csc', dtype=np.object, copy=True)
        n_samples, n_features = X.shape
        X_int = np.zeros_like(X, dtype=np.int)
        X_mask = np.ones_like(X, dtype=np.bool)

        for i in range(n_features):
            valid_mask = np.in1d(X[:, i], self.categories_[i])

            if not np.all(valid_mask):
                if self.handle_unknown == 'error':
                    diff = np.unique(X[~valid_mask, i])
                    msg = ("Found unknown categories {0} in column {1}"
                           " during transform".format(diff, i))
                    raise ValueError(msg)
                else:
                    # Set the problematic rows to an acceptable value and
                    # continue `The rows are marked `X_mask` and will be
                    # removed later.
                    X_mask[:, i] = valid_mask
                    X[:, i][~valid_mask] = self.categories_[i][0]
            X_int[:, i] = self._label_encoders_[i].transform(X[:, i])

        if self.encoding == 'ordinal':
            return X_int.astype(self.dtype, copy=False)

        mask = X_mask.ravel()
        n_values = [cats.shape[0] for cats in self.categories_]
        n_values = np.array([0] + n_values)
        indices = np.cumsum(n_values)

        column_indices = (X_int + indices[:-1]).ravel()[mask]
        row_indices = np.repeat(np.arange(n_samples, dtype=np.int32),
                                n_features)[mask]
        data = np.ones(n_samples * n_features)[mask]

        out = sparse.csc_matrix((data, (row_indices, column_indices)),
                                shape=(n_samples, indices[-1]),
                                dtype=self.dtype).tocsr()
        if self.encoding == 'onehot-dense':
            return out.toarray()
        else:
            return out


# The `CategoricalEncoder` expects a 2D array containing one or more categorical input features. We need to reshape `MyDataset_cat` to a 2D array:

#from sklearn.preprocessing import CategoricalEncoder # in future versions of Scikit-Learn

cat_encoder = CategoricalEncoder()
MyDataset_cat_reshaped = MyDataset_cat.values.reshape(-1, 1)
MyDataset_cat_1hot = cat_encoder.fit_transform(MyDataset_cat_reshaped)
MyDataset_cat_1hot


# The default encoding is one-hot, and it returns a sparse array. You can use `toarray()` to get a dense array:


MyDataset_cat_1hot.toarray()


# Alternatively, you can specify the encoding to be `"onehot-dense"` to get a dense matrix rather than a sparse matrix:

cat_encoder = CategoricalEncoder(encoding="onehot-dense")
MyDataset_cat_1hot = cat_encoder.fit_transform(MyDataset_cat_reshaped)
MyDataset_cat_1hot





cat_encoder.categories_


# Let's create a custom transformer to add extra attributes:
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
MyDataset_extra_attribs = attr_adder.transform(MyDataset.values)


MyDataset_extra_attribs = pd.DataFrame(MyDataset_extra_attribs, columns=list(MyDataset.columns)+["rooms_per_household", "population_per_household"])
MyDataset_extra_attribs.head()


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

MyDataset_num_tr = num_pipeline.fit_transform(MyDataset_num)


MyDataset_num_tr


from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values



num_attribs = list(MyDataset_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs)),
        ('cat_encoder', CategoricalEncoder(encoding="onehot-dense")),
    ])


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

MyDataset_prepared = full_pipeline.fit_transform(MyDataset)
MyDataset_prepared


MyDataset_prepared.shape


# # Select and train a model 

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(MyDataset_prepared, MyDataset_labels)


# let's try the full pipeline on a few training instances
some_data = MyDataset.iloc[:5]
some_labels = MyDataset_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))


print("Labels:", list(some_labels))


some_data_prepared

from sklearn.metrics import mean_squared_error

MyDataset_predictions = lin_reg.predict(MyDataset_prepared)
lin_mse = mean_squared_error(MyDataset_labels, MyDataset_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse


from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(MyDataset_labels, MyDataset_predictions)
lin_mae


from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(MyDataset_prepared, MyDataset_labels)


MyDataset_predictions = tree_reg.predict(MyDataset_prepared)
tree_mse = mean_squared_error(MyDataset_labels, MyDataset_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse


# # Fine-tune your model


from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, MyDataset_prepared, MyDataset_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)



def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)

lin_scores = cross_val_score(lin_reg, MyDataset_prepared, MyDataset_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)



from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(random_state=42)
forest_reg.fit(MyDataset_prepared, MyDataset_labels)


MyDataset_predictions = forest_reg.predict(MyDataset_prepared)
forest_mse = mean_squared_error(MyDataset_labels, MyDataset_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse


from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, MyDataset_prepared, MyDataset_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)



scores = cross_val_score(lin_reg, MyDataset_prepared, MyDataset_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()


from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(MyDataset_prepared, MyDataset_labels)
MyDataset_predictions = svm_reg.predict(MyDataset_prepared)
svm_mse = mean_squared_error(MyDataset_labels, MyDataset_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse

from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error')
grid_search.fit(MyDataset_prepared, MyDataset_labels)


# The best hyperparameter combination found:

grid_search.best_params_



grid_search.best_estimator_


cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)


pd.DataFrame(grid_search.cv_results_)


