import pandas as pd

dataset = pd.read_csv('credit.csv')

X = dataset.iloc[:, 2:11].values
y = dataset.iloc[:, -1:].values


# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split
# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# Building the Random Forest Classifier (RANDOM FOREST)
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
rfc.fit(X_train,Y_train)

# predictions
y_pred = rfc.predict(X_test)

#printing the confusion matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(Y_test, y_pred)
print(conf_matrix)
#manual input
at = float(input("Enter amount per transaction : "))
tran_amt = float(input("Enter transaction amount: "))
is_dec = float(input("Enter 1 if your transaction is declined else 0: "))
trans_dec = float(input("Enter total number of declined transaction per day: "))
for_trans = float(input("Enter 1 if it is foreign transaction else 0: "))
trans_risk = float(input("Enter 1 if transaction is under high risk else 0: "))
d_ch_amt = float(input("Enter Daily chargeback average amount: "))
m_ch_amt6 = float(input("Enter 6 month average chargeback amount: "))
m_ch_amt = float(input("Enter 6 month chargeback frequency: "))

test = [[at,tran_amt,is_dec,trans_dec,for_trans,trans_risk,d_ch_amt,m_ch_amt6,m_ch_amt]]


y_pred2 = rfc.predict(test)
print(y_pred2)
if y_pred2 == 1:
    print("It is a Fraud")
else:
    print("It is a normal")