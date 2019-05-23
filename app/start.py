from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
  
from flask import Flask, render_template, url_for, request

app = Flask(__name__, static_url_path='/static')

@app.route('/',methods=['POST','GET'])
def home():
    
    bedrooms=request.form.get('bedrooms',type=int)
    if bedrooms is None or bedrooms=='':
        bedrooms=2
    
    yearbuilt=request.form.get('yearbuilt',type=int)
    #yearbuilt=float(yearbuilt)
    if yearbuilt is None or yearbuilt=='':
        yearbuilt=2009
    
    lotarea=request.form.get('lotarea',5000)
    #lotarea=int(lotarea)
    if lotarea is None or lotarea=='':
        lotarea=4500
    #bedrooms =  request.form["bedrooms"]
    
    #<--- script starts from here --->
    
    import itertools
    import tensorflow as tf
    import pandas as pd
    import numpy as np
    #from pylab import rcParams
    import matplotlib

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler

    tf.logging.set_verbosity(tf.logging.INFO)
    sess = tf.InteractiveSession()

    train = pd.read_csv('app/train.csv')
    train = train.select_dtypes(exclude=['object'])
    print("")
    train.drop('Id',axis = 1, inplace = True)
    train.fillna(0,inplace=True)

    test = pd.read_csv('app/testing2 - test.csv')

    # Take input from the user
    #print("Enter the Lot Area")
    test.iloc[0,4]=int(lotarea)           #int(input())
    #print("Enter the Year Of Built")
    test.iloc[0,19]=int(yearbuilt)          #int(input())
    #print("Enter the number of bedrooms")
    test.iloc[0,51]=int(bedrooms)          #int(input())
    #---------------------------------------

    test = test.select_dtypes(exclude=['object'])
    ID = test.Id
    test.fillna(0,inplace=True)
    test.drop('Id',axis = 1, inplace = True)

    print("")

    from sklearn.ensemble import IsolationForest

    clf = IsolationForest(max_samples = 100, random_state = 42)
    clf.fit(train)
    y_noano = clf.predict(train)
    y_noano = pd.DataFrame(y_noano, columns = ['Top'])
    y_noano[y_noano['Top'] == 1].index.values

    train = train.iloc[y_noano[y_noano['Top'] == 1].index.values]
    train.reset_index(drop = True, inplace = True)

    import warnings
    warnings.filterwarnings('ignore')

    col_train = list(train.columns)
    col_train_bis = list(train.columns)

    col_train_bis.remove('SalePrice')

    mat_train = np.matrix(train)
    mat_test  = np.matrix(test)
    mat_new = np.matrix(train.drop('SalePrice',axis = 1))
    mat_y = np.array(train.SalePrice).reshape((1314,1))

    prepro_y = MinMaxScaler()
    prepro_y.fit(mat_y)

    prepro = MinMaxScaler()
    prepro.fit(mat_train)

    prepro_test = MinMaxScaler()
    prepro_test.fit(mat_new)

    train = pd.DataFrame(prepro.transform(mat_train),columns = col_train)
    test  = pd.DataFrame(prepro_test.transform(mat_test),columns = col_train_bis)

    COLUMNS = col_train
    FEATURES = col_train_bis
    LABEL = "SalePrice"

    # Columns for tensorflow
    feature_cols = [tf.contrib.layers.real_valued_column(k) for k in FEATURES]

    # Training set and Prediction set with the features to predict
    training_set = train[COLUMNS]
    prediction_set = train.SalePrice

    # Train and Test 
    x_train, x_test, y_train, y_test = train_test_split(training_set[FEATURES] , prediction_set, test_size=0.33, random_state=42)
    y_train = pd.DataFrame(y_train, columns = [LABEL])
    training_set = pd.DataFrame(x_train, columns = FEATURES).merge(y_train, left_index = True, right_index = True)
    training_set.head()

    # Training for submission
    training_sub = training_set[col_train]


    y_test = pd.DataFrame(y_test, columns = [LABEL])
    testing_set = pd.DataFrame(x_test, columns = FEATURES).merge(y_test, left_index = True, right_index = True)
    testing_set.head()


    tf.logging.set_verbosity(tf.logging.ERROR)
    regressor = tf.contrib.learn.DNNRegressor(feature_columns=feature_cols, 
                                              activation_fn = tf.nn.relu, hidden_units=[200, 100, 50, 25, 12])#,
                                             #optimizer = tf.train.GradientDescentOptimizer( learning_rate= 0.1 ))

    training_set.reset_index(drop = True, inplace =True)

    def input_fn(data_set, pred = False):

        if pred == False:

            feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}
            labels = tf.constant(data_set[LABEL].values)

            return feature_cols, labels

        if pred == True:
            feature_cols = {k: tf.constant(data_set[k].values) for k in FEATURES}

            return feature_cols

    regressor.fit(input_fn=lambda: input_fn(training_set), steps=2000)
    ev = regressor.evaluate(input_fn=lambda: input_fn(testing_set), steps=1)
    loss_score1 = ev["loss"]
    loss=loss_score1*1545676
    y = regressor.predict(input_fn=lambda: input_fn(testing_set))
    predictions = list(itertools.islice(y, testing_set.shape[0]))
    predictions = pd.DataFrame(prepro_y.inverse_transform(np.array(predictions).reshape(434,1)),columns = ['Prediction'])

    predictor=predictions.iloc[0,:]
    #print(str(predictor).split(" ")[4].split("\n")[0])
    
    #<-- Script ends here -->
    final_answer=str(str(predictor).split(" ")[4].split("\n")[0])
    return render_template('index.html',answer=final_answer)

@app.route('/output/')
def videopage():
    return render_template('videopage.html')

@app.route('/about/')
def info():
    return render_template('about.html')

@app.route('/contact/')
def contactinfo():
    return render_template('contact.html')

@app.route('/portfolio/')
def portfolio():
    return render_template('portfolio.html')

if __name__=="__main__":
    app.run(debug=True)