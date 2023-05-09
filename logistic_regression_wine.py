import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from statsmodels.miscmodels.ordinal_model import OrderedModel


wd = pd.read_csv('winequality-red.csv', sep=";")

# Set 5 samples aside
wine_data, test_sample = train_test_split(wd, test_size=5, random_state=42)

# splitting the data into independent and dependent
X = wine_data.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
y = wine_data.iloc[:, 11]
k = X.drop("residual sugar", axis=1)

# scaling
sc = StandardScaler()
X = sc.fit_transform(k)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# fit/train model
classifierElastic = LogisticRegression(random_state=0, penalty='elasticnet', solver='saga', l1_ratio=0.1, max_iter=3000, C=0.01)
classifierElastic.fit(X_train, y_train)

classifierL1 = LogisticRegression(random_state=0, penalty='l1', solver='saga', max_iter=3000, C=0.01)
classifierL1.fit(X_train, y_train)

classifierL2 = LogisticRegression(random_state=0, penalty='l2', solver='saga', max_iter=3000, C=0.01)
classifierL2.fit(X_train, y_train)

# predictions
y_pred = classifierElastic.predict(X_test)
result = pd.DataFrame({'Actual' : y_test, 'Predicted' : y_pred})


Y_pred_l1 = classifierL1.predict(X_test)
Y_pred_l2 = classifierL2.predict(X_test)

print("Elastic net model acc: " + str(accuracy_score(y_test, y_pred) * 100), '%')
print("l1 model acc: " + str(accuracy_score(y_test, Y_pred_l1) * 100), '%')
print("l2 model acc: " + str(accuracy_score(y_test, Y_pred_l2) * 100), '%')


# Ordinal
cat_type = pd.CategoricalDtype(ordered=True)
wine_data['quality'] = wine_data['quality'].astype(cat_type)

# split the data
ordered_train, ordered_test, sample_train, sample_test = train_test_split(X, wine_data['quality'], test_size=0.25,
                                                                              random_state=42)

# ordered logit model
mod_prob = OrderedModel(wine_data['quality'], X, distr='logit')
res_prob = mod_prob.fit(method='newton', disp=False)

#
predicted = res_prob.model.predict(res_prob.params, exog=ordered_test)
pred_out = predicted.argmax(1)
print("Ordered Logit Model acc:")
print((np.asarray(sample_test.values.codes) == pred_out).mean() * 100, '%')

print("-----------------")

# prepare data for test sample
# split test sample
J = test_sample.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]
h = test_sample.iloc[:, 11]
M = J.drop("residual sugar", axis=1)

J = sc.fit_transform(M)

# scale test sample
k_pred = classifierElastic.predict(J)
k_pred_l1 = classifierL1.predict(J)
k_pred_l2 = classifierL2.predict(J)
print("Test Sample Elastic net acc: " + str(accuracy_score(h, k_pred) * 100), '%')
print("Test Sample l1 acc: " + str(accuracy_score(h, k_pred_l1) * 100), '%')
print("Test Sample l2 acc: " + str(accuracy_score(h, k_pred_l2) * 100), '%')

# ordered logit model
cat_type = pd.CategoricalDtype(ordered=True)
test_sample['quality'] = test_sample['quality'].astype(cat_type)
h = h.astype(cat_type)
predicted = res_prob.model.predict(res_prob.params, exog=J)
pred_out = predicted.argmax(1)
print("Test Sample Ordered Logit Model acc at 5 samples:")
print((np.asarray(h.values.codes) == pred_out).mean() * 100, '%')




