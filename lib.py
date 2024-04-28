import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import lmfit
from uncertainties import ufloat
from uncertainties import unumpy as unp
from icecream import ic
from lmfit.models import LinearModel, GaussianModel, LorentzianModel
from scipy.stats import levy_stable
from landaupy import landau

def to_int(x):
    try:
        number = int(x)
        return number
    except ValueError:
        return x
    
def integrate(x, y) -> float:
    tot: float = np.trapz(x=x, y=y)
    return tot

def mk_err_dict(err_arr: np.double, color: str = 'black') -> dict:
    return dict(type='data',
                color=color,
                array=err_arr)

def scatter_ufloats(x_uarr: np.ndarray, y_uarr: np.ndarray, color="blue", fig=None, name=None, row=1, col=1, legendgroup="First test", showlegend=False):
    """
    Adds trace of ufloats markers to fig.
    """
    fig.add_trace(go.Scatter(x=unp.nominal_values(x_uarr),
                             y=unp.nominal_values(y_uarr),
                             error_x=mk_err_dict(unp.std_devs(x_uarr)),
                             error_y=mk_err_dict(unp.std_devs(y_uarr)),
                             mode="markers",
                             marker_color=color,
                             name=name,
                             legendgroup=legendgroup,
                             showlegend=showlegend),
                             row=row, col=col)
    return 0

def line(x, A):
    return A*x
def first_exp(x, B, C):
    return B*x*np.exp(C*np.sqrt(x))
def second_exp(x, D, E):
    return D*np.exp(E*x)
def logistic(x, mu, c):
    return (1/(1+np.exp(- (x+mu)/c))+1)/2
def lin_square(x, A, B, C, D, E, F):
    return logistic(-x, A, E)*B*x + (1-logistic(-x, A, E))*np.exp(C*x)*D + F
def landau_func(x, mu, xi, A):
    return A*landau.pdf(x, mu, xi)
def sincos(x, N, k):
    return N*np.sin(x)*np.cos(x)*np.exp(-k/np.cos(x))

line_mod = lmfit.Model(line)
exp1_mod = lmfit.Model(first_exp)
exp2_mod = lmfit.Model(second_exp)
landau_mod = lmfit.Model(landau_func)
GaussianModel()
LinearModel()
sincos_mod = lmfit.Model(sincos)

#mod = line_mod + exp1_mod + exp2_mod
#params = mod.make_params(A=230, B=1.9e-17, C=3e-10, D=2e-13, E=4e2, F=0.1)
#param_values = dict(x=test[:,0], A=230, B=1.9e-17, C=3e-10, D=2e-3, E=4e-5, F=0.1)
#res = mod.fit(data=yval, params=params, x=xval, weights=1/yerr, nan_policy="omit")

x = np.linspace(0, 100, 200)
y = np.linspace(1, 2, 200)

mod = GaussianModel() + landau_mod
params = mod.make_params(amplitude=5, center=10, sigma=1, mu=30, xi=dict(value=2, min=0.01), A=1)
res = mod.fit(data=y, params=params, x=x, weights=1/y, nan_policy="omit")
print(res.fit_report())