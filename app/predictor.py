import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor


class Predictor:
    data = pd.DataFrame()
    cars_set = set()
    X, Y = pd.DataFrame(), pd.DataFrame()
    xtrain, xtest, ytrain, ytest = [], [], [], []
    classifier = RandomForestRegressor()
    scaler = StandardScaler()

    def __init__(self) -> None:
        url='https://docs.google.com/spreadsheets/d/11FtA9l3Mov2HTfoseedSGJerFLrD9qHfkGJaVp3HLuA/edit?usp=sharing'
        dwn_url='https://drive.google.com/uc?id=' + url.split('/')[-2]
        self.data = pd.read_csv(dwn_url, thousands=',')
        self.data[:] = self.data.dropna()
        self.preprocess()
        self.predictt()

    def preprocess(self):
        for i, val in enumerate(self.data['selling_price']):
            if isinstance(val, str):
                if 'Cr' in val:
                    self.data[:] = self.data.drop(i)
        self.data['company'] = self.data['full_name'].str.split(' ', expand=True)[
            0]
        for i, val in enumerate(self.data['engine']):
            if val == " ":
                self.data[:] = self.data.drop(i)
        for i, val in enumerate(self.data['max_power']):
            if val == " ":
                self.data[:] = self.data.drop(i)
            if val == 'null ':
                self.data[:] = self.data.drop(i)
        self.data = pd.get_dummies(data=self.data, columns=[
                                   'seller_type', 'fuel_type', 'transmission_type', 'company'], drop_first=True)
        self.cars_set = set(self.data.iloc[:, 3])
        self.data['selling_price'] = pd.to_numeric(self.data['selling_price'])
        for i, val in enumerate(self.data['selling_price']):
            if val > 300:
                self.data['selling_price'][i] /= 100000
        self.data = self.data.dropna()
        self.data = self.data[self.data['selling_price'] < 15]
        self.X = self.data.iloc[:, 3:]
        self.Y = self.data.iloc[:, 4]
        self.X = self.X.drop(
            ['selling_price', 'full_name', 'owner_type'], axis=1)
        self.X['engine'] = pd.to_numeric(self.X['engine'])
        self.X['max_power'] = pd.to_numeric(self.X['max_power'])
        self.X = self.X.dropna()
        self.split()

    def split(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.X, self.Y, test_size=0.2, random_state=1)
        self.xtrain = self.scaler.fit_transform(self.xtrain)
        self.xtest = self.scaler.transform(self.xtest)

    def do_prediction(self):
        self.classifier.fit(self.xtrain, self.ytrain)
        prediction = self.classifier.predict(self.xtest)
        cross_validation_score = self.cross_val(self.xtrain, self.ytrain)
        error = mean_absolute_error(self.ytest, prediction)
        return error, cross_validation_score, prediction

    def cross_val(self, xtrain, ytrain):
        accuracies = cross_val_score(
            estimator=self.classifier, X=xtrain, y=ytrain, cv=5)
        return accuracies.mean()

    def predictt(self):
        error, score, pred = self.do_prediction()
        print('Random Forest Regressor MAE: {}'.format(round(error, 2)))
        print('Cross validation score: {}'.format(round(score, 2)))
        print(pred)

    def predict_single(self, name, km, year):
        x = self.data.loc[self.data['full_name'] == name].iloc[0, 5:]
        x = x.drop(['owner_type'], axis=0)
        x['engine'] = pd.to_numeric(x['engine'])
        x['max_power'] = pd.to_numeric(x['max_power'])
        x['km_driven'] = km
        x['year'] = year
        x = self.scaler.transform([x])
        y = self.classifier.predict(x)
        return y[0]
