import pandas as pd
import requests
from io import StringIO
import re
from flask import Flask,jsonify
from sklearn.preprocessing import MinMaxScaler
from imblearn.under_sampling import CondensedNearestNeighbour
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import HuberRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.feature_selection import f_regression
from sklearn.preprocessing import LabelEncoder
import time


# Create flask app
flask_app = Flask(__name__)

df_train=None
df_test=None
X_train=None
X_test=None
y_train=None
y_test=None
classifier=None
algo=None
sampling=None
fs=None
is_train= False
is_test= False
is_fs= False
is_sampling= False
is_algo = False
read_only=False
ans=None


@flask_app.route('/')
def home():
    if globals()['is_train'] == False:
        return "Load the train data"
    elif globals()['is_test'] == False:
        return "Load the test data"
    elif globals()['is_fs'] == False:
        return "Choose whether or not to apply feature selection"
    elif globals()['is_sampling'] == False:
        return "Choose whether or not to apply sampling"
    elif globals()['is_algo'] == False:
        return "Choose the algorithm to apply"
    else :
        return "You are ready for prediction"


@flask_app.route('/train/<path:p>',methods=['PUT'])
def train(p):
    if globals()['read_only']==True:
        return "Read only mode enabled"
    response_API = requests.get(p)
    s=str(response_API.content,'utf-8')
    data = StringIO(s) 
    globals()['df_train']=pd.read_csv(data,low_memory=False)

    globals()['X_train']=globals()['df_train'].iloc[:,:-1]
    globals()['y_train']=globals()['df_train'].iloc[:,-1]

    if(type(globals()['y_train']) != int and type(globals()['y_train']) != float):
        le = LabelEncoder()
        globals()['y_train']=le.fit_transform(globals()['y_train'])
        globals()['y_train']=pd.DataFrame(globals()['y_train'])
        print(globals()['y_train'])
    print(globals()['X_train'])

    # Mean imputation
    globals()['X_train'] = globals()['X_train'].apply(lambda x:x.fillna(x.mean(skipna=True)),axis=0)

    # Feature scaling
    sc = MinMaxScaler()
    globals()['X_train'] = sc.fit_transform(globals()['X_train'])
    globals()['X_train']=pd.DataFrame(globals()['X_train'])
    
    globals()['df_train']=pd.concat([globals()['X_train'],globals()['y_train']],axis=1)
    print(globals()['df_train'])
    globals()['is_train']=True
    FS_Train()
    SAMP()
    return ALGO()

def FS_Train():
    if globals()['read_only']==True:
        return "Read only mode enabled"
    if globals()['is_train'] == True:
        n=globals()['fs']
        n=str(n)
        globals()['X_train']=globals()['df_train'].iloc[:,:-1]
        globals()['y_train'] = globals()['df_train'].iloc[:,-1]
        X=globals()['X_train']
        y=globals()['y_train']

        columns =[]
        for i in range(0,len(X.columns)):
            columns.append(i)
        X.columns = columns

        if re.search("IG\d",str(n)):
            from sklearn.feature_selection import mutual_info_classif as mf
            scores = pd.DataFrame(mf(X,y).reshape(-1,1),columns=['scores'],index=X.columns)
            d=scores.index
            d=pd.DataFrame(d)
            Three = pd.concat([d,scores],axis=1)

            a=None
            try:
                a=n[2:]
                a=int(a)
            except:
                a=15

            ans = (Three.nlargest(a,'scores')).iloc[:,0].tolist()
            globals['ans']=ans
            globals()['X_train'] = X.loc[:,ans]
            globals()['is_fs']=True

        elif re.search('CHI\d',n):
            a=None
            try:
                a=n[3:]
                a=int(a)
            except:
                a=15

            from sklearn.feature_selection import SelectKBest
            from sklearn.feature_selection import chi2
            features = SelectKBest(score_func=chi2,k=a)
            fit= features.fit(X,y)

            scores = pd.DataFrame(fit.scores_)

            globals()['X_train'].columns = columns
            columns=pd.DataFrame(columns)

            dataX=pd.concat([columns,scores],axis=1)
            dataX.columns = ['0','1']
            ans = dataX.nlargest(a,'1')
            ans=ans.iloc[:,0].tolist()

            globals()['X_train'] = X.loc[:,ans]
            globals['ans']=ans
            globals()['is_fs']=True

        elif re.search('ETC\d',n):
            a=None
            try:
                a=n[3:]
                a=int(a)
            except:
                a=15

            from sklearn.ensemble import ExtraTreesClassifier as ETC
            initial = ETC()
            initial.fit(X,y)
            s = pd.DataFrame(initial.feature_importances_)

            columns =[]
            for i in range(0,len(X.columns)):
                columns.append(i)

            X.columns = columns
            columns=pd.DataFrame(columns)

            dataX1 = pd.concat([columns,s],axis=1)

            dataX1.columns = ['Columns','s']
            ans = dataX1.nlargest(a,'s')
            ans = ans.iloc[:,0].tolist()


            globals()['X_train'] = X.loc[:,ans]
            globals['ans']=ans
            globals()['is_fs']=True


        elif re.search('ANOVA\d',n):
            a=None
            try:
                a=n[5:]
                a=int(a)
            except:
                a=15
            f=SelectKBest(score_func=f_regression,k=a)
            t_fit=f.fit(globals()['X_train'],globals()['y_train'])

            fr_scores=pd.DataFrame(t_fit.scores_)
            globals()['df_train']=pd.concat([globals()['X_train'],globals()['y_train'],fr_scores],axis=1,ignore_index=True)
            globals()['df_train'] = globals()['df_train'].sort_values(by=globals()['df_train'].columns[-1],ascending=False)
            globals()['X_train'] = globals()['df_train'].iloc[:,:a]
            globals()['y_train'] = globals()['df_train'].iloc[:,-2]
            globals['ans']=globals['X_train'].columns

            print(globals()['X_train'])
            globals()['is_fs']=True
            return "ANOVA feature selection completed"
        print(globals()['X_train'])
        globals()['is_fs']=True
    return

@flask_app.route('/test/<path:p>',methods=['POST'])
def test(p):

    if globals()['read_only']==True:
        return "Read only mode enabled"
    response_API = requests.get(p)
    s=str(response_API.content,'utf-8')
    data = StringIO(s) 
    globals()['df_test'] = pd.read_csv(data,low_memory=False)

    globals()['X_test']=globals()['df_test'].iloc[:,:-1]
    globals()['y_test']=globals()['df_test'].iloc[:,-1]

    if(type(globals()['y_test']) != int and type(globals()['y_test']) != float):
        le = LabelEncoder()
        globals()['y_test']=le.fit_transform(globals()['y_test'])
        globals()['y_test']=pd.DataFrame(globals()['y_test'])

    # Mean imputation
    globals()['X_test'] = globals()['X_test'].apply(lambda x:x.fillna(x.mean(skipna=True)),axis=0)

    # Feature scaling
    sc = MinMaxScaler()
    globals()['X_test'] = sc.fit_transform(globals()['X_test'])
    globals()['X_test']=pd.DataFrame(globals()['X_test'])

    globals()['df_test']=pd.concat([globals()['X_test'],globals()['y_test']],axis=1)

    print(globals()['X_test'].isnull().any().tolist())
    globals()['is_test']=True
    if(ans is not None):
        return FS_test()
    return "Mean imputation and MinMaxScaling of Testing Data is complete"


def FS_test():
    n=globals()['fs']
    if globals()['read_only']==True:
        return "Read only mode enabled"
    if globals()['is_test']==True:
        globals()['X_test']=globals()['df_test'].iloc[:,:-1]
        X=globals()['X_test']
        columns =[]
        for i in range(0,len(X.columns)):
            columns.append(i)
        X.columns = columns
        globals()['X_test']=X
        globals()['X_test']=globals()['X_test'].loc[:,ans]
        return "Feature Selection of Testing Data completed"


@flask_app.route("/FS/<n>", methods = ["PUT"])
def FeatureSelection(n):
    if globals()['read_only']==True:
        return "Read only mode enabled"
    globals()['fs']==n
    return "Feature Selection initialization step completed"


@flask_app.route("/FS", methods = ["GET"])
def FS_():
    if globals()['fs']=='NO':
        return "No feature selection technique is applied on the dataset"
    return "ANOVA feature selection technique is applied on the dataset"


@flask_app.route("/SAMPLING/<v>", methods = ["PUT"])
def SAMPLING(v):
    if globals()['read_only']==True:
        return "Read only mode enabled"
    globals()['sampling']=v
    globals()['is_sampling']=True
    return "Sampling Selection Complete"


def SAMP():
    if globals()['read_only']==True:
        return
    v=globals()['sampling']
    if globals()['is_train']==True:
        if v=='UNDER':
            # define the undersampling method
            undersample = CondensedNearestNeighbour(n_neighbors=1)
            # transform the dataset
            globals()['X_train'], globals()['y_train'] = undersample.fit_resample(globals()['X_train'], globals()['y_train'])
        elif v=='OVER':
            # instantiating the random over sampler 
            ros = RandomOverSampler()
            # resampling X, y
            globals()['X_train'], globals()['y_train'] = ros.fit_resample(globals()['X_train'], globals()['y_train'])
    return 


@flask_app.route("/ALGO/<al>", methods = ["PUT"])
def Algorithm(al):
    if globals()['read_only']==True:
        return "Read only mode enabled"
    globals()['algo']=al
    globals()['is_algo']==True
    return "Algorithm selection complete"


def ALGO():
    if globals()['read_only']==True:
        return "Read only mode enabled"
    al=globals()['algo']
    stime=time.time()
    if al=="SVC":
        globals()['classifier'] = SVC(kernel='rbf', random_state = 1)
        globals()['classifier'].fit(globals()['X_train'],globals()['y_train'])
        return jsonify("SVC model is successfully trained and the training time is {}".format(time.time()-stime))
    if al=="MLP":
        globals()['classifier'] = MLPClassifier(hidden_layer_sizes=(6,5),random_state=5,verbose=True,learning_rate_init=0.01)
        globals()['classifier'].fit(globals()['X_train'],globals()['y_train'])
        return jsonify("MLP model is successfully trained and the training time is {}".format(time.time()-stime))
    if al=="RF":
        globals()['classifier'] = RandomForestRegressor(n_estimators = 100, random_state = 0)
        globals()['classifier'].fit(globals()['X_train'], globals()['y_train'])
        return jsonify("RF model is successfully trained and the training time is {}".format(time.time()-stime))
    if al=="GDBT":
        GR = GradientBoostingRegressor(n_estimators = 200, max_depth = 1, random_state = 1) 
        globals()['classifier'] = GR.fit(globals()['X_train'], globals()['y_train']) 
        return jsonify("GDBT model is successfully trained and the training time is {}".format(time.time()-stime))
    if al=="RUSBT":
        globals()['classifier'] = HuberRegressor()
        globals()['classifier'] = globals()['classifier'].fit(globals()['X_train'],globals()['y_train']) 
        return jsonify("RUSBT model is successfully trained and the training time is {}".format(time.time()-stime))
    return "Select an algorithm to implement"


@flask_app.route("/PREDICT/<values>", methods = ["POST"])
def predict(values):
    stime=time.time()
    to_predict=values.split(',')

    for i in range(len(to_predict)):
        to_predict[i] = float(to_predict[i])

    prediction = globals()['classifier'].predict([to_predict])
    if(prediction>0.5):
        prediction=1
    else:
        prediction=0
    return jsonify("The prediction is {} and testing time is {}".format(prediction,time.time()-stime))


@flask_app.route("/PREDICT/test", methods = ["POST"])
def predict_test():
    stime=time.time()

    globals()['y_test'] = globals()['y_test'].fillna(0)

    globals()['X_test']=pd.DataFrame(globals()['X_test'])
    print(globals()['X_test'].isnull().any().tolist())
    v=globals()['X_test'].isnull().any().tolist()
    X_test = globals()['X_test']
    import math
    for i in range(len(X_test)):
        for j in range(len(X_test.columns)):
            if(math.isnan(X_test.iloc[i,j])):
                X_test.iloc[i,j]=0
    X_test = pd.DataFrame(X_test)
    prediction = globals()['classifier'].predict(X_test)
    print(prediction)
    for i in range(len(prediction)):
        if(prediction[i]>0.5):
            prediction[i]=1
        else:
            prediction[i]=0
    return jsonify("The accuracy of model is {} and testing time is {}".format(accuracy_score(globals()['y_test'],prediction),time.time()-stime))


@flask_app.route("/SAMPLING", methods = ["GET"])
def Sampling_():
    if globals()['sampling']=='NO':
        return "No sampling technique is applied on the dataset"
    
    return "{}-sampling technique is applied on the dataset".format(globals()['sampling'])


@flask_app.route("/ALGO", methods = ["GET"])
def ALGO_():
    return "{}-technique is applied on the dataset".format(globals()['algo'])


@flask_app.route("/READONLY", methods = ["POST"])
def read_only():
    globals()['read_only']=True
    return jsonify("The system is set to read only mode")


if __name__ == "__main__":

    flask_app.run(debug=True)