from typing import Callable, Dict, List, Tuple, Union
import argparse
from collections import defaultdict
import csv
import json
import math
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
from statistics import mean
import seaborn as sns

sns.set(context="paper", style="darkgrid", palette="crest")

def plot_joint_density(algorithm:str, color:str, VaR_errors:List[float], ES_errors:List[float],
                       VaR_label:str , ES_label:str, outpath:str):
   p = sns.jointplot(x=VaR_errors, y=ES_errors, kind="kde",
                     fill=True, alpha=0.5, color=color,
                     height=8, ratio=5)
   p.plot_joint(sns.scatterplot, color="blue")
   p.fig.suptitle(algorithm, fontsize=16)
   p.fig.subplots_adjust(hspace=0, wspace=0, top=0.95, right=0.95)
   p.ax_joint.set_xlabel(xlabel=VaR_label, fontsize=14)
   p.ax_joint.set_ylabel(ylabel=ES_label, fontsize=14)
   #p.ax_joint.set_xticklabels(p.ax_joint.get_xticks(), size=10)
   #p.ax_joint.set_yticklabels(p.ax_joint.get_yticks(), size=10)

   plt.tight_layout()
   fname = algorithm.lower().replace("-","_").replace(" ", "_") + ".pdf"
   p.fig.savefig(os.path.join(outpath, fname), bbox_inches="tight", pad_inches=0)

def sub_process(filename:str, outpath:str, VaR:float, ES:float)->None:
   with open(filename) as f:
      d = json.load(f)

   h = d["precision"]
   beta = d["beta"]
   VaR_h = d["biased_VaR"]
   ES_h = d["biased_ES"]

   VaR_errors = []
   VaR_avg_errors = []
   ES_errors = []
   ES_avg_errors = []

   # SA
   keys = d["sa_simulations"]["header"]
   for values in d["sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h, -beta)*(data["VaR"] - VaR))
      VaR_avg_errors.append(math.pow(h, -1)*(data["VaR_avg"] - VaR))
      ES_errors.append(math.pow(h, -1)*(data["ES"] - ES))
   plot_joint_density("Unbiased SA", "tab:red", VaR_errors, ES_errors,
                      r"$h^{-\beta}(\xi_{\lceil h^{-2}\rceil}-\xi_{\star})$",
                      r"$h^{-1}(\chi_{\lceil h^{-2}\rceil}-\chi_{\star})$",
                      outpath)
   plot_joint_density("Averaged Unbiased SA", "tab:orange", VaR_avg_errors, ES_errors,
                      r"$h^{-1}(\bar{\xi}_{\lceil h^{-2}\rceil}-\xi_{\star})$",
                      r"$h^{-1}(\chi_{\lceil h^{-2}\rceil}-\chi_{\star})$",
                      outpath)

   VaR_errors.clear()
   VaR_avg_errors.clear()
   ES_errors.clear()
   ES_avg_errors.clear()

   # Nested SA
   keys = d["nested_sa_simulations"]["header"]
   for values in d["nested_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h, -beta)*(data["VaR"] - VaR))
      VaR_avg_errors.append(math.pow(h, -1)*(data["VaR_avg"] - VaR))
      ES_errors.append(math.pow(h, -1)*(data["ES"] - ES))
   plot_joint_density("Nested SA", "tab:red", VaR_errors, ES_errors,
                      r"$h^{-\beta}(\xi^{h}_{\lceil h^{-2}\rceil}-\xi_{\star})$",
                      r"$h^{-1}(\chi^{h}_{\lceil h^{-2}\rceil}-\chi_{\star})$",
                      outpath)
   plot_joint_density("Averaged Nested SA", "tab:orange", VaR_avg_errors, ES_errors,
                      r"$h^{-1}(\bar{\xi}^{h}_{\lceil h^{-2}\rceil}-\xi_{\star})$",
                      r"$h^{-1}(\chi^{h}_{\lceil h^{-2}\rceil}-\chi_{\star})$",
                      outpath)

   VaR_errors.clear()
   VaR_avg_errors.clear()
   ES_errors.clear()
   ES_avg_errors.clear()

   # Multilevel SA
   keys = d["ml_sa_simulations"]["header"]
   for values in d["ml_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_errors.append(math.pow(h, -1)*(data["VaR"] - VaR_h))
      ES_errors.append(math.pow(h, -1/beta-(2*beta-1)/(4*beta*(1+beta)))*(data["ES"] - ES_h))
   keys = d["avg_ml_sa_simulations"]["header"]
   for values in d["ml_sa_simulations"]["rows"]:
      data = dict(zip(keys, values))
      if data["status"] == "failure":
         continue
      VaR_avg_errors.append(math.pow(h, -(beta+3)/(4*beta))*(data["VaR_avg"] - VaR_h))
      ES_avg_errors.append(math.pow(h, -1/beta-(2*beta-1)/(4*beta*(1+beta)))*(data["ES"] - ES_h))
   plot_joint_density("Multilevel SA", "tab:red", VaR_errors, ES_errors,
                      r"$h_L^{-1}(\xi^{h_L}_{\mathbf{N}}-\xi^{h_L}_{\star})$",
                      r"$h_L^{-\frac{1}{\beta}-\frac{2\beta-1}{4\beta(1+\beta)}}(\chi^{h_L}_{\mathbf{N}}-\chi^{h_L}_{\star})$",
                      outpath)
   plot_joint_density("Averaged Multilevel SA", "tab:orange", VaR_avg_errors, ES_avg_errors,
                      r"$h_L^{-1}(\bar{\xi}^{h_L}_{\mathbf{N}}-\xi^{h_L}_{\star})$",
                      r"$h_L^{-\frac{9}{8}}(\chi^{h_L}_{\mathbf{N}}-\chi^{h_L}_{\star})$",
                      outpath)

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

   VaR = eta*norm.ppf(alpha)
   ES = eta*norm.pdf(VaR/eta)/(1-alpha) 

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
   print("{")
   print("\t\"VaR\": %.5f," % VaR)
   print("\t\"ES\": %.5f" % ES)
   print("}")

   sub_process(filename, outpath, VaR, ES)

if __name__ == "__main__":
   p = argparse.ArgumentParser(description="Process and plot SA results")
   p.add_argument("input_json", help="Json file name")
   p.add_argument("output_path", help="Output path for figures")
   args = p.parse_args()

   if not os.path.exists(args.input_json) or not os.path.isfile(args.input_json):
      raise FileNotFoundError(f"file {args.input_json} not found")
   if not os.path.exists(args.output_path) or not os.path.isdir(args.output_path):
      raise IOError(f"directory {args.output_path} not found")

   process(args.input_json, args.output_path)
