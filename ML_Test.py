import yfinance as yf
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# List of stock symbols
symbols = ['AAPL', 'MSFT', 'GOOG', 'AMZN', 'META', 'TSLA', 'NVDA', 'NFLX', 'ADBE', 'INTC']

for symbol in symbols:
    print(f"Processing {symbol}...")

    # Download historical data for desired ticker symbol 
    data = yf.download(symbol,'2020-01-01','2021-01-01')

    # Use only Close price for prediction
    data = data[['Close']]

    # Predict for the next 'n' days
    n = 1

    # Create a new column (target) shifted 'n' units up
    data['Predicted'] = data[['Close']].shift(-n)

    # Create a binary target variable
    data['Target'] = (data['Predicted'].shift(-1) > data['Predicted']).astype(int)

    # Create the independent data set (X)
    X = data.drop(['Predicted', 'Target'], axis=1)[:-n]

    # Create the dependent data set (y)
    y = data['Target'][:-n]

    # Split the data into 80% training and 20% testing
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Define the individual classifiers
    clf1 = GaussianNB()
    clf2 = LogisticRegression(penalty='l1', solver='liblinear')  # L1 regularization
    clf3 = DecisionTreeClassifier()
    clf4 = SGDClassifier()

    # Define the ensemble classifier
    eclf = VotingClassifier(estimators=[('gnb', clf1), ('lr', clf2), ('dt', clf3), ('sgd', clf4)], voting='hard')

    # Train the ensemble classifier on the training data
    eclf.fit(x_train, y_train)

    # Evaluate the classifier on the test data
    print(f"{symbol} Model Accuracy: ", eclf.score(x_test, y_test))

