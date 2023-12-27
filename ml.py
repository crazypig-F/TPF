import numpy as np
import pandas as pd

import path
from model.regression import RegressionModel
from model.classify import ClassifyModel
from core import dominant_microbe
from sklearn.metrics import r2_score
from scipy.stats import linregress


def physic_microbe():
    microbe = pd.read_csv(path.TEMP_MICROBE_MERGE_TOP_20_PARALLEL, index_col=0)
    physic = pd.read_csv(path.TEMP_PHYSIC_PARALLEL, index_col=0)
    real = {}
    predict = {}
    score = []
    for col in microbe.columns:
        reg = RegressionModel(physic, microbe[col], test_size=0.2, seed=42)
        r, p, s = reg.train()
        print(col, s)
        real[col] = r
        predict[col] = p
        score.append(s)
    pd.DataFrame(real).to_csv(path.TEMP_ML_REAL_MICROBE)
    pd.DataFrame(predict).to_csv(path.TEMP_ML_PRED_MICROBE)
    pd.DataFrame({"Score": score}, index=microbe.columns).to_csv(path.TEMP_ML_SCORE_MICROBE)


def microbe_amino():
    microbe = pd.read_csv(path.TEMP_MICROBE_MERGE_TOP_20_PARALLEL, index_col=0)
    core_microbe = pd.read_csv(path.TEMP_MICROBE_CORE_AMINO_MICRO, index_col=0)
    amino = pd.read_csv(path.TEMP_AMINO_PARALLEL, index_col=0)
    core_amino = pd.read_csv(path.TEMP_MICROBE_CORE_AMINO, index_col=0)
    real = {}
    predict = {}
    score = []
    best_score = []
    std_list = []
    microbe = microbe.loc[:, core_microbe.loc[core_microbe['degree'] >= 5, :].index]
    # amino = amino.loc[:, core_amino.loc[core_amino['degree'] >= 4, :].index]
    print(core_microbe.loc[core_microbe['degree'] >= 5, :].index)
    print(core_amino.loc[core_amino['degree'] >= 4, :].index)
    for col in amino.columns:
        reg = RegressionModel(microbe, amino[col], seed=0)
        # reg.get_feature(col, microbe.columns)
        # reg.search()
        # exit()
        r, p, m, std, bs = reg.train()
        print(col, m, std, bs)
        real[col] = r
        predict[col] = p
        score.append(m)
        best_score.append(bs)
        std_list.append(std)
        print("r2", r2_score(r, p))
        print("r2", linregress(r, p).rvalue**2, "P", linregress(r, p).pvalue)
    pd.DataFrame(real).to_csv(path.TEMP_ML_REAL_AMINO)
    pd.DataFrame(predict).to_csv(path.TEMP_ML_PRED_AMINO)
    pd.DataFrame({"Score": score, "BestScore": best_score, "Std": std_list}, index=amino.columns).to_csv(path.TEMP_ML_SCORE_AMINO1)
    # pd.DataFrame({"Score": score, "BestScore": best_score, "Std": std_list}, index=amino.columns).to_csv(path.TEMP_ML_SCORE_AMINO2)


def phase_predict():
    phase_map = {"AZ": 0, "BZ": 1, "CZ": 2, "DZ": 3}
    microbe = pd.read_csv(path.TEMP_MICROBE_MERGE_TOP_20_PARALLEL, index_col=0)
    core = pd.read_csv(path.TEMP_MICROBE_CORE_MICRO, index_col=0)
    dominant = dominant_microbe()
    core = core.loc[core['degree'] >= 5, :].index.to_list()
    key = sorted(list(set(dominant) & set(core)))
    print(key)
    microbe = microbe.loc[:, key]
    physic = pd.read_csv(path.TEMP_PHYSIC_PARALLEL, index_col=0)
    physic.columns = ["starch", "moisture", "SP", "RS", "TA", "TE"]
    data = pd.concat([microbe, physic], axis=1)
    y = np.array([phase_map[i[:2]] for i in physic.index])
    clf = ClassifyModel(data, y, seed=0)
    s = clf.train()
    print(s)


def main():
    # phase_predict()
    # physic_microbe()
    microbe_amino()


if __name__ == '__main__':
    main()
