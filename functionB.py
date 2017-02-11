import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans

def predict_optimal_answer (input_vals) :
    df = pd.read_csv('./savecsv.csv')

    df.head()
    x_train = df.drop(['stress'],axis = 1)
    y_train = df['stress']
    # from sklearn.tree import DecisionTreeClassifier
    # from sklearn.svm import SVC 
    # from sklearn.neighbors import KNeighborsClassifier
    def dist(x1, x2):
        return np.sqrt(((x1 - x2)**2).sum())
        # return abs(x1-x2).sum()
        # return np.sqrt(0.5*((np.sqrt(x1) - np.sqrt(x2))**2).sum())
    def knn(X_train, x, y_train, k=5):
        vals = [] 
        for ix in range(X_train.shape[0]):
            v = [dist(x, X_train[ix, :]), y_train[ix]]
            vals.append(v)
        
        updated_vals = sorted(vals, key=lambda x: x[0])
        pred_arr = np.asarray(updated_vals[:k])
        pred_arr = np.unique(pred_arr[:, 1], return_counts=True)
        pred = pred_arr[1].argmax()
        return int(pred_arr[0][pred])
    # clf = DecisionTreeClassifier()
    # clf = SVC()
    # clf = KNeighborsClassifier()
    # clf.fit(x_train,y_train)
    user = input_vals

    sscore = 21
    sleep_hours = 5
    work_hours = 10
    freetime = 5
    holperyear = 0

    sscore = user

    x = [sscore,sleep_hours,work_hours,freetime,holperyear]
    # pred = clf.predict(x)
    # print pred
    xnp_train = np.asarray(x_train)
    ynp_train = np.asarray(y_train)
    pred = knn(xnp_train,x,ynp_train)

    def findNearest (X_train, x, y_train) :
        minval = 9999999999
        minval2 = 9999999999
        z = X_train[0]
        z2 = X_train[0]
        for ix in range(X_train.shape[0]):
            f = dist(x, X_train[ix, :])
            if (minval > f) and (y_train[ix] == 0) :
                if (minval2>f) :
                    minval = minval2
                    z = z2
                    minval2 = f
                    z2 = X_train[ix, :]
                else :
                    minval = f
                    z = X_train[ix, :]
        mean = z2
        
        for k in range(len(z)) :
            mean[k] =float(z[k]+z2[k])/2.0
            
        return mean


    if pred == 1 :
        p = findNearest(xnp_train,x,ynp_train)
    else :
        p = x

    p = p.append(pred)
    # last element p[-1] is whether or not the person is in stress . 1 is yes , 0 is not in stress

    return p




