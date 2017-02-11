import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans



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
    

def dist(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum())
    # return abs(x1-x2).sum()
    # return np.sqrt(0.5*((np.sqrt(x1) - np.sqrt(x2))**2).sum())

def convertToDict(mylist):
    d = {"sscore": str(mylist[0]),
        "sleep_hours": str(mylist[1]),
        "work_hours": str(mylist[2]),
        "freetime": str(mylist[3]),
        "holperyear": str(mylist[4]),
        "stress": str(mylist[5]),
        "new_sscore": str(mylist[6]),
        "old_sscore": str(mylist[7])
        }
    return d




def predict_optimal_answer (input_vals) :
    df = pd.read_csv('./save2csv.csv')
    x_train = df.drop(['stress'],axis = 1)
    y_train = df['stress']

    # user = input_vals
    # sscore = user['sscore']
    # sleep_hours = user['sleep_hours']
    # work_hours = user['work_hours']
    # freetime = user['freetime']
    # holperyear = user['holperyear']

    sscore = 18
    sleep_hours = 8
    work_hours = 12
    freetime = 0
    holperyear = 0
    # sscore = user
    x = [sscore,sleep_hours,work_hours,freetime,holperyear]
    # pred = clf.predict(x)
    # print pred
    xnp_train = np.asarray(x_train)
    ynp_train = np.asarray(y_train)
    pred = knn(xnp_train,x,ynp_train)

    if pred == 1 :
        p = findNearest(xnp_train,x,ynp_train)
    else :
        p = x

    vals = []
    # vals = new_solution - old_solution

    for ix in range(len(x)) :
        diff = p[ix] - x[ix]
        vals.append(diff)
    # pred = 0 is not stressed and pred = 1 is stressed 
    vals.append(pred)
    vals.append(p[0])
    vals.append(x[0])
    # last element p[-1] is whether or not the person is in stress . 1 is yes , 0 is not in stress
    # print x
    # print p
    # print vals
    print 'INPUT-SCHEDULE:   ', x
    print 'OPTIMAL-SCHEDULE: ',p
    print 'OUTPUT-SCHEDULE:  ', vals
    vals = convertToDict(vals)
    print vals
    return vals

if __name__ == "__main__" :
    predict_optimal_answer(1)


