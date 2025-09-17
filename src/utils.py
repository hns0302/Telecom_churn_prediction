import joblib
def save_model(model, path='models/model.pkl'):
    joblib.dump(model, path)

def load_model(path='models/model.pkl'):
    return joblib.load(path)