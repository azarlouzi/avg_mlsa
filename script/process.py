from typing import Callable, Dict, List, Tuple, Union, Any
import numpy.typing as npt
import argparse
import numpy as np
from collections import defaultdict
import csv
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import scipy.stats as stats
import scipy.optimize as opt
import statistics as st
from sklearn.mixture import GaussianMixture
import seaborn as sns

sns.set()

def gaussian_pdf(x, mu, sigma):
   return stats.norm.pdf((x - mu) / sigma)

def plot_joint_density(algorithm:str, xlim:Tuple[float,float], ylim:Tuple[float,float],
                       VaR_errors: List[float], ES_errors: List[float], mean:npt.NDArray, cov:npt.NDArray,
                       ellipse_data:Dict[str,Union[npt.NDArray,float]], ES_variance:float,
                       VaR_label: str, ES_label: str, outpath: str):

   fig, axs = plt.subplots(2, 2, figsize=(10, 10), height_ratios=(1, 6), width_ratios=(6, 1))
   #plt.autoscale(False)
   fig.suptitle(algorithm, size=24)
   fig.subplots_adjust(hspace=0.05, wspace=0.05)

   ellipse = mpl.patches.Ellipse(ellipse_data["center"], ellipse_data["semi_major_axis"],
                                 ellipse_data["semi_minor_axis"], angle=ellipse_data["angle"],
                                 facecolor="none", edgecolor="tab:red",
                                 linewidth=3, alpha=0.9, clip_box=axs[1,0].bbox)
   axs[1,0].add_artist(ellipse)
   axs[1,0].scatter(VaR_errors, ES_errors, alpha=0.4)
   axs[1,0].set_xlabel(VaR_label, size=18)
   axs[1,0].set_ylabel(ES_label, size=18)
   axs[1,0].set_xlim(xlim)
   axs[1,0].set_ylim(ylim)

   x = np.array([xlim[0] + (xlim[1]-xlim[0]) * i/1000 for i in range(0, 1001)])
   y = gaussian_pdf(x, mean[0], math.sqrt(cov[0,0]))
   axs[0,0].plot(x, y, '-', color='tab:red', linewidth=2.2)
   sns.histplot(VaR_errors, bins=10, stat="probability", alpha=0.675, ax=axs[0,0])
   axs[0,0].set_xlabel('')
   axs[0,0].set_ylabel("pdf", size=18)
   axs[0,0].set_xlim(xlim)
   axs[0,0].set_ylim((0, 0.5))
   axs[0,0].tick_params(bottom=False, left=False, labelbottom=False)
   axs[0,0].set_yticks([0, 0.25, 0.5], labels=['', '0.25', '0.5'])

   x = np.array([ylim[0] + (ylim[1] - ylim[0]) * i / 1000 for i in range(0, 1001)])
   y = gaussian_pdf(x, mean[1], math.sqrt(cov[1,1]))
   fit, = axs[1,1].plot(y, x, '-', color='tab:red', linewidth=2.2)
   x = np.array([ylim[0] + (ylim[1] - ylim[0]) * i / 20 for i in range(0, 21)])
   y = gaussian_pdf(x, mean[1], math.sqrt(ES_variance))
   MC, = axs[1,1].plot(y, x, 'x', color='black', markersize=12)
   sns.histplot(y=ES_errors, bins=10, stat="probability", alpha=0.675, ax=axs[1,1])
   axs[1,1].set_ylabel('')
   axs[1,1].set_xlabel("pdf", size=18)
   axs[1,1].set_ylim(ylim)
   axs[1,1].set_xlim((0, 0.5))
   axs[1,1].tick_params(left=False, bottom=False, labelleft=False)
   axs[1,1].set_xticks([0, 0.25, 0.5], labels=['', '0.25', '0.5'])

   axs[0,1].axis('off')
   axs[0,1].legend((fit, MC), ("fit", "MC"), loc="lower left", fontsize=15)

   plt.tight_layout()
   fname = algorithm.lower().replace("-","_").replace(" ", "_") + ".pdf"
   fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def fit_gaussian(xy:npt.NDArray)->Tuple[npt.NDArray,npt.NDArray]:
   model = GaussianMixture(n_components=1).fit(xy)
   mean = model.means_[0]
   cov = model.covariances_[0]
   return mean, cov

def lim_interval(mean:npt.NDArray, cov:npt.NDArray)->Tuple[Tuple[float,float],Tuple[float,float]]:
   mean1, mean2 = mean[0], mean[1]
   sd1, sd2 = np.sqrt(cov[0,0]), np.sqrt(cov[1,1])
   xlim = [np.round(mean1 - 3*sd1), np.round(mean1 + 3*sd1)]
   ylim = [np.round(mean2 - 3*sd2), np.round(mean2 + 3*sd2)]
   xlen = xlim[1] - xlim[0]
   ylen = ylim[1] - ylim[0]
   adj = np.abs(ylen - xlen)/2
   if xlen > ylen:
      ylim[0] -= adj
      ylim[1] += adj
   else:
      xlim[0] -= adj
      xlim[1] += adj
   return tuple(xlim), tuple(ylim)

def compute_ellipse(mean:npt.NDArray, cov:npt.NDArray)->Dict[str,Union[npt.NDArray,float,float,float]]:
   v, w = np.linalg.eigh(cov)
   v = 2*np.sqrt(stats.chi2.ppf(0.95, 2)*v)
   u = w[0]/np.linalg.norm(w[0])
   angle = np.arctan2(u[1], u[0]) * 180/np.pi
   return {
      "center": mean,
      "semi_major_axis": v[0],
      "semi_minor_axis": v[1],
      "angle": angle,
   }

def sub_process(filename:str, outpath:str, VaR:float, ES:float)->None:
   out = {
      "VaR": VaR,
      "ES": ES,
   }
   with open(filename) as f:
      d = json.load(f)

   h = d["precision"]
   beta = d["beta"]

   biased_VaR_values, biased_ES_values = [], []
   for entry in d["biased_risk_measures"]:
      VaR, ES = entry
      biased_VaR_values.append(VaR)
      biased_ES_values.append(ES)
   biased_VaR = st.mean(biased_VaR_values)
   biased_ES = st.mean(biased_ES_values)

   out["biased_VaR"] = biased_VaR
   out["biased_ES"] = biased_ES

   VaR_errors = []
   VaR_avg_errors = []
   ES_errors = []
   ES_avg_errors = []
   ES_variances = []
   ES_avg_variances = []

   # SA
   keys = d["sa_simulations"]["header"]
   for values in d["sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h,-beta)*(data["VaR"] - VaR))
      VaR_avg_errors.append(math.pow(h,-1)*(data["VaR_avg"] - VaR))
      ES_errors.append(math.pow(h,-1)*(data["ES"] - ES))
      ES_variances.append(data["ES_var"])
   mean, cov = fit_gaussian(np.stack((VaR_errors, ES_errors), axis=1))
   ES_variance = st.mean(ES_variances)
   xlim, ylim = lim_interval(mean, cov)
   out["Unbiased SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_variance,
   }
   plot_joint_density("Unbiased SA", xlim, ylim, VaR_errors, ES_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_variance,
                      r"$h^{-\beta}(\xi^{0}_{\lceil h^{-2}\rceil}-\xi^{0}_{\star})$",
                      r"$h^{-1}(\chi^{0}_{\lceil h^{-2}\rceil}-\chi^{0}_{\star})$",
                      outpath)
   mean, cov = fit_gaussian(np.stack((VaR_avg_errors, ES_errors), axis=1))
   xlim, ylim = lim_interval(mean, cov)
   out["Averaged Unbiased SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_variance,
   }
   plot_joint_density("Averaged Unbiased SA", xlim, ylim, VaR_avg_errors, ES_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_variance,
                      r"$h^{-1}(\bar{\xi}^{0}_{\lceil h^{-2}\rceil}-\xi^{0}_{\star})$",
                      r"$h^{-1}(\chi^{0}_{\lceil h^{-2}\rceil}-\chi^{0}_{\star})$",
                      outpath)

   VaR_errors.clear()
   VaR_avg_errors.clear()
   ES_errors.clear()
   ES_avg_errors.clear()
   ES_variances.clear()
   ES_avg_variances.clear()

   # Nested SA
   keys = d["nested_sa_simulations"]["header"]
   for values in d["nested_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h,-beta)*(data["VaR"] - VaR))
      VaR_avg_errors.append(math.pow(h,-1)*(data["VaR_avg"] - VaR))
      ES_errors.append(math.pow(h,-1)*(data["ES"] - ES))
      ES_variances.append(data["ES_var"])
   mean, cov = fit_gaussian(np.stack((VaR_errors, ES_errors), axis=1))
   ES_variance = st.mean(ES_variances)
   xlim, ylim = lim_interval(mean, cov)
   out["Nested SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_variance,
   }
   plot_joint_density("Nested SA", xlim, ylim, VaR_errors, ES_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_variance,
                      r"$h^{-\beta}(\xi^{h}_{\lceil h^{-2}\rceil}-\xi^{0}_{\star})$",
                      r"$h^{-1}(\chi^{h}_{\lceil h^{-2}\rceil}-\chi^{0}_{\star})$",
                      outpath)
   mean, cov = fit_gaussian(np.stack((VaR_avg_errors, ES_errors), axis=1))
   xlim, ylim = lim_interval(mean, cov)
   out["Averaged Nested SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_variance,
   }
   plot_joint_density("Averaged Nested SA", xlim, ylim, VaR_avg_errors, ES_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_variance,
                      r"$h^{-1}(\bar{\xi}^{h}_{\lceil h^{-2}\rceil}-\xi^{0}_{\star})$",
                      r"$h^{-1}(\chi^{h}_{\lceil h^{-2}\rceil}-\chi^{0}_{\star})$",
                      outpath)

   VaR_errors.clear()
   VaR_avg_errors.clear()
   ES_errors.clear()
   ES_avg_errors.clear()
   ES_variances.clear()
   ES_avg_variances.clear()

   # Multilevel SA
   keys = d["ml_sa_simulations"]["header"]
   for values in d["ml_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h,-1)*(data["VaR"] - biased_VaR))
      ES_errors.append(math.pow(h,-1/beta-(2*beta-1)/(4*beta*(1+beta)))*(data["ES"] - biased_ES))
      ES_variances.append(data["ES_var"])
   mean, cov = fit_gaussian(np.stack((VaR_errors, ES_errors), axis=1))
   ES_variance = st.mean(ES_variances)
   xlim, ylim = lim_interval(mean, cov)
   out["Multilevel SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_variance,
   }
   plot_joint_density("Multilevel SA", xlim, ylim, VaR_errors, ES_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_variance,
                      r"$h_L^{-1}(\xi^{h_L}_{\mathbf{N}}-\xi^{h_L}_{\star})$",
                      r"$h_L^{-\frac{1}{\beta}-\frac{2\beta-1}{4\beta(1+\beta)}}(\chi^{h_L}_{\mathbf{N}}-\chi^{h_L}_{\star})$",
                      outpath)

   VaR_errors.clear()
   VaR_avg_errors.clear()
   ES_errors.clear()
   ES_avg_errors.clear()
   ES_variances.clear()
   ES_avg_variances.clear()

   # Averaged Multilevel SA
   keys = d["avg_ml_sa_simulations"]["header"]
   for values in d["avg_ml_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_avg_errors.append(math.pow(h,-1)*(data["VaR_avg"] - biased_VaR))
      ES_avg_errors.append(math.pow(h,-9./8)*(data["ES"] - biased_ES))
      ES_avg_variances.append(data["ES_var"])
   mean, cov = fit_gaussian(np.stack((VaR_avg_errors, ES_avg_errors), axis=1))
   ES_avg_variance = st.mean(ES_avg_variances)
   xlim, ylim = lim_interval(mean, cov)
   out["Averaged Multilevel SA"] = {
      "mean": mean.tolist(),
      "covariance": cov.tolist(),
      "ES_variance": ES_avg_variance,
   }
   plot_joint_density("Averaged Multilevel SA", xlim, ylim, VaR_avg_errors, ES_avg_errors,
                      mean, cov, compute_ellipse(mean, cov), ES_avg_variance,
                      r"$h_L^{-1}(\bar{\xi}^{h_L}_{\mathbf{N}}-\xi^{h_L}_{\star})$",
                      r"$h_L^{-\frac{9}{8}}(\chi^{h_L}_{\mathbf{N}}-\chi^{h_L}_{\star})$",
                      outpath)

   print(json.dumps(out, indent=4))

def compute_rate_swap_model_risk_measures(
   r:float, S_0:float, kappa:float, sigma:float,
   Delta:float, T:float, delta:float, # in days
   leg_0:float, alpha:float)->Tuple[float,float]:
   """ Compute exact values of the risk measures """

   def discount(t):
      return math.exp(-r*t)

   def reset(i):
      return i*Delta

   Delta = Delta/360.0
   T = T/360.0
   delta = delta/360.0
   d = int(T/Delta)

   nominal = leg_0/(S_0*sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(1,d+1)))
   eta = nominal*sigma*math.sqrt((1-math.exp(-2*kappa*delta))/(2*kappa))* \
         sum(discount(reset(i))*Delta*math.exp(kappa*reset(i-1)) for i in range(2,d+1))

   VaR = eta*stats.norm.ppf(alpha)
   ES = eta*stats.norm.pdf(VaR/eta)/(1-alpha) 

   return VaR, ES

def process(filename:str, outpath:str)->None:
   r     = 0.02
   S_0   = 100.0 # in basis points
   kappa = 0.12
   sigma = 0.2
   Delta = 90.0  # in days
   T     = 360.0 # in days
   delta = 7.0   # in days
   leg_0 = 1e4   # in basis points
   alpha = 0.85

   VaR, ES = compute_rate_swap_model_risk_measures(r, S_0, kappa, sigma, Delta,
                                                   T, delta, leg_0, alpha)
   sub_process(filename, outpath, VaR, ES)

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Process and plot SA results")
   p.add_argument("input_json", help="Input Json file name")
   p.add_argument("output_path", help="Output path for figures")
   args = p.parse_args()

   if not os.path.exists(args.input_json) or not os.path.isfile(args.input_json):
      raise FileNotFoundError(f"file {args.input_json} not found")
   if not os.path.exists(args.output_path) or not os.path.isdir(args.output_path):
      raise IOError(f"directory {args.output_path} not found")

   process(args.input_json, args.output_path)
