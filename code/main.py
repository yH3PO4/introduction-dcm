import numpy as np
import pandas as pd
from scipy.optimize import minimize

df = pd.read_csv('ensyu_mnl/ensyu.csv', encoding='shift-jis')


def fr(x: np.array) -> float:
    b1, b2, b3, b4, d1, f1 = x  # b1~b4: 定数項, d1: 所要時間, f1: 運賃
    
    train: pd.Series = df["代替手段生成可否train"] * np.exp(
    d1 * df["総所要時間train"] / 100 + f1 * df["費用train"] / 100 + b1
    )
    bus: pd.Series = df["代替手段生成可否bus"] * np.exp(
    d1 * df["総所要時間bus"] / 100 + f1 * df["費用bus"] / 100 + b2
    )
    car: pd.Series = df["代替手段生成可否car"] * np.exp(
    d1 * df["所要時間car"] / 100 + b3
    )
    bike: pd.Series = df["代替手段生成可否bike"] * np.exp(
    d1 * df["所要時間bike"] / 100 + b4
    )
    walk: pd.Series = df["代替手段生成可否walk"] * np.exp(
    d1 * df["所要時間walk"] / 100
    )
    
    deno: pd.Series = car + train + bus + bike + walk
    
    Ptrain: pd.Series = df["代替手段生成可否train"] * (train / deno)
    Pbus: pd.Series = df["代替手段生成可否bus"] * (bus / deno)
    Pcar: pd.Series = df["代替手段生成可否car"] * (car / deno)
    Pbike: pd.Series = df["代替手段生成可否bike"] * (bike / deno)
    Pwalk: pd.Series = df["代替手段生成可否walk"] * (walk / deno)
    
    Ptrain = Ptrain.where(Ptrain != 0, 1)
    Pbus = Pbus.where(Pbus != 0, 1)
    Pcar = Pcar.where(Pcar != 0, 1)
    Pbike = Pbike.where(Pbike != 0, 1)
    Pwalk = Pwalk.where(Pwalk != 0, 1)
    
    Ctrain: pd.Series = df["代表交通手段"] == "鉄道"
    Cbus: pd.Series = df["代表交通手段"] == "バス"
    Ccar: pd.Series = df["代表交通手段"] == "自動車"
    Cbike: pd.Series = df["代表交通手段"] == "自転車"
    Cwalk: pd.Series = df["代表交通手段"] == "徒歩"
    
    LL: float = np.sum(
    Ctrain * np.log(Ptrain)
    + Cbus * np.log(Pbus)
    + Ccar * np.log(Pcar)
    + Cbike * np.log(Pbike)
    + Cwalk * np.log(Pwalk)
    )
    return LL
    
    
def mf(x: np.array) -> float:
    return -fr(x)
    
    
def hessian(x: np.array) -> np.array:
    h = 10 ** -4  # 数値微分用の微小量
    n = len(x)
    res = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            e_i, e_j = np.zeros(n), np.zeros(n)
            e_i[i] = 1
            e_j[j] = 1
            
            res[i][j] = (fr(x + h * e_i + h * e_j)
            - fr(x + h * e_i - h * e_j)
            - fr(x - h * e_i + h * e_j)
            + fr(x - h * e_i - h * e_j)) / (4 * h * h)
    return res
    
    
def tval(x: np.array) -> np.array:
    return x / np.sqrt(-np.diag(np.linalg.inv(hessian(x))))
    
    
x0 = np.zeros(6)
res = minimize(mf, x0, method="Nelder-Mead")
print(res)

print(f"tval: {tval(res.x)}")
L0 = fr(x0)
print(f"L0: {L0:.4f}")  # 初期尤度
LL = -res.fun
print(f"LL: {LL:.4f}")  # 最終尤度
R = 1 - LL / L0
print(f"R: {R:.4f}")  # 決定係数
R_adj = 1 - (LL - len(x0)) / L0
print(f"R_adj: {R2:.4f}")  # 修正済み決定係数
