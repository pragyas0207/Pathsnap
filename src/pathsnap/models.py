from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

MODEL_NAMES = ('logistic', 'naive_bayes', 'decision_tree', 'random_forest', 'hist_boost')


def make_model(name, seed=42, params=None):
    p = params or {}
    if name == 'logistic':
        return make_pipeline(StandardScaler(), LogisticRegression(C=p.get('C', 1.), max_iter=500, random_state=seed))
    if name == 'naive_bayes':
        return GaussianNB(var_smoothing=p.get('var_smoothing', 1e-9))
    if name == 'decision_tree':
        return DecisionTreeClassifier(max_depth=p.get('max_depth', 12),
            min_samples_leaf=p.get('min_samples_leaf', 20), random_state=seed)
    if name == 'random_forest':
        return RandomForestClassifier(n_estimators=p.get('n_estimators', 80),
            max_depth=p.get('max_depth', 16), min_samples_leaf=p.get('min_samples_leaf', 10),
            max_features=p.get('max_features', 'sqrt'), n_jobs=1, random_state=seed)
    if name == 'hist_boost':
        return HistGradientBoostingClassifier(max_iter=p.get('max_iter', 100),
            max_leaf_nodes=p.get('max_leaf_nodes', 31), learning_rate=p.get('learning_rate', .1),
            l2_regularization=p.get('l2_regularization', 1.),
            min_samples_leaf=p.get('min_samples_leaf', 30), early_stopping=False, random_state=seed)
    raise ValueError(f'Unknown model: {name}')


def search_space(trial, name):
    if name == 'logistic':
        return {'C': trial.suggest_float('C', .001, 100, log=True)}
    if name == 'naive_bayes':
        return {'var_smoothing': trial.suggest_float('var_smoothing', 1e-12, 1e-2, log=True)}
    if name == 'decision_tree':
        return {'max_depth': trial.suggest_int('max_depth', 4, 24),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 100, log=True)}
    if name == 'random_forest':
        return {'n_estimators': trial.suggest_int('n_estimators', 40, 160, step=40),
                'max_depth': trial.suggest_int('max_depth', 6, 24),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 5, 60, log=True),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', .8])}
    if name == 'hist_boost':
        return {'max_iter': trial.suggest_int('max_iter', 60, 160, step=20),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 7, 63),
                'learning_rate': trial.suggest_float('learning_rate', .02, .25, log=True),
                'l2_regularization': trial.suggest_float('l2_regularization', .01, 20, log=True),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 100, log=True)}
    raise ValueError(name)
