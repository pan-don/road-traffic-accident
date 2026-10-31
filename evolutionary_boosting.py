import numpy as np
from cmaes import CMA, CatCMAwM
from sklearn.metrics import f1_score, recall_score
from art_resampling import ARTDataset, art_refresh_dataset_from_predictions
import xgboost as xgb

def class_weights(y_arr, scheme="none"):
    if scheme == "none":
        return None
    counts = np.bincount(y_arr, minlength=3)
    if scheme == "sqrt":
        w = 1.0 / np.sqrt(np.maximum(counts, 1))
    elif scheme == "effective_number":
        beta = 0.999
        eff = (1.0 - np.power(beta, counts)) / (1.0 - beta)
        w = 1.0 / np.maximum(eff, 1e-6)
    else:
        return None
    w = w / w.sum() * 3
    return w[y_arr]


def run_evo_cmaes(evo_budget, depths, schemes, X_train, y_train, X_val, y_val, seed):
    opt = CMA(mean=np.array([0.5] * 6), sigma=0.3, bounds=np.array([[0.0, 1.0]] * 6), seed=seed)
    best_res = (-np.inf, None, None, None, None, None)
    evals, hist = 0, []
    while evals < evo_budget:
        sols = []
        for _ in range(opt.population_size):
            sol = opt.ask()
            lr = float(np.exp(np.log(0.03) + sol[0] * (np.log(0.15) - np.log(0.03))))
            c_val = 0.25 + 0.45 * sol[1]
            sf = 1.0 + 3.0 * sol[2]
            ss = 1.0 + 1.5 * sol[3]
            d = depths[min(int(sol[4] * len(depths)), len(depths) - 1)]
            sch = schemes[min(int(sol[5] * len(schemes)), len(schemes) - 1)]
            p = {"learning_rate": lr, "max_depth": d}
            scales = np.array([sf, ss, 1.0])
            ds = ARTDataset(np.arange(len(X_train)).reshape(-1, 1), y_train, c=c_val, cls_weights=np.full(3, 1/3), seed=seed, enable_art=True)
            m = xgb.XGBClassifier(n_estimators=100, subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                                 tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0, **p)
            for r in range(1, 5):
                sample_idx = ds.X[:, 0].astype(int)
                sw = class_weights(ds.y, sch)
                fit_kwargs = {"sample_weight": sw} if sw is not None else {}
                m.fit(X_train.iloc[sample_idx], ds.y, **fit_kwargs)
                P_val = m.predict_proba(X_val)
                if r < 4:
                    art_refresh_dataset_from_predictions(ds, y_val, P_val.argmax(1), 3, {"batch_size": 256})
                    
            p_val = (P_val * scales).argmax(1)
            wf1 = f1_score(y_val, p_val, average="weighted")
            rec = recall_score(y_val, p_val, average=None, zero_division=0)
            fitness = wf1 * (1.0 + 0.6 * rec[0] + 0.3 * rec[1])
            if wf1 < 0.72:
                fitness -= (0.72 - wf1) * 4.0
            sols.append((sol, -fitness))
            evals += 1
            if fitness > best_res[0]:
                best_res = (fitness, p, c_val, scales, sch, m)

        opt.tell(sols)
        hist.append(best_res[0])        
    return best_res, hist


def run_evo_catcmawm(evo_budget, depths, schemes, X_train, y_train, X_val, y_val, seed):
    opt = CatCMAwM(x_space=[[0.0, 1.0]] * 4, z_space=[depths], c_space=[len(schemes)], seed=seed)
    best_res = (-np.inf, None, None, None, None, None)
    evals, hist = 0, []
    while evals < evo_budget:
        sols = []
        for _ in range(opt.population_size):
            sol = opt.ask()
            lr = float(np.exp(np.log(0.03) + sol.x[0] * (np.log(0.15) - np.log(0.03))))
            c_val = 0.25 + 0.45 * sol.x[1]
            sf = 1.0 + 3.0 * sol.x[2]
            ss = 1.0 + 1.5 * sol.x[3]
            d = int(sol.z[0])
            sch = schemes[int(np.argmax(sol.c[0]))]
            p = {"learning_rate": lr, "max_depth": d}
            scales = np.array([sf, ss, 1.0])
            ds = ARTDataset(np.arange(len(X_train)).reshape(-1, 1), y_train, c=c_val, cls_weights=np.full(3, 1/3), seed=seed, enable_art=True)
            m = xgb.XGBClassifier(n_estimators=100, subsample=0.8, colsample_bytree=0.8, enable_categorical=True,
                                 tree_method="hist", n_jobs=-1, random_state=seed, verbosity=0, **p)
            for r in range(1, 5):
                sample_idx = ds.X[:, 0].astype(int)
                sw = class_weights(ds.y, sch)
                fit_kwargs = {"sample_weight": sw} if sw is not None else {}
                m.fit(X_train.iloc[sample_idx], ds.y, **fit_kwargs)
                P_val = m.predict_proba(X_val)
                if r < 4:
                    art_refresh_dataset_from_predictions(ds, y_val, P_val.argmax(1), 3, {"batch_size": 256})
                    
            p_val = (P_val * scales).argmax(1)
            wf1 = f1_score(y_val, p_val, average="weighted")
            rec = recall_score(y_val, p_val, average=None, zero_division=0)
            fitness = wf1 * (1.0 + 0.6 * rec[0] + 0.3 * rec[1])
            if wf1 < 0.72:
                fitness -= (0.72 - wf1) * 4.0
            sols.append((sol, -fitness))
            evals += 1
            if fitness > best_res[0]:
                best_res = (fitness, p, c_val, scales, sch, m)
                
        opt.tell(sols)
        hist.append(best_res[0])
    return best_res, hist
