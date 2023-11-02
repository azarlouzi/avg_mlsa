#include "helpers.h"
#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"
#include "option_model.h"
#include "swap_model.h"
#include <cmath>
#include <cstdio>
#include <exception>
#include <string>

void print_comma(bool yes) {
   if (yes) {
      std::printf(",\n");
   } else {
      std::printf("\n");
   }
}

Risk_Measures biased_estimator(IN     double            xi_0,
                               IN     double            chi_0,
                               IN     double            alpha,
                               IN     double            precision,
                               IN     const Step&       gamma,
                               IN OUT Nested_Simulator& nested_simulator) {
   long int n = 100000L;
   return nested_sa(xi_0, chi_0, alpha, precision, n, gamma, nested_simulator);
}

void subroutine(IN     double            xi_0,
                IN     double            chi_0,
                IN     double            alpha,
                IN     double            beta,
                IN     double            precision,
                IN     const Step&       sa_gamma,
                IN     const Step&       nested_sa_gamma,
                IN     const Step&       ml_sa_gamma,
                IN OUT Simulator&        simulator,
                IN OUT Nested_Simulator& nested_simulator,
                IN OUT ML_Simulator&     ml_simulator) {
   Risk_Measures risk_measures;

   risk_measures = biased_estimator(xi_0, chi_0, alpha, precision,
                                    nested_sa_gamma, nested_simulator);

   std::printf("{\n");
   std::printf("\t\"precision\": %.10f,\n", precision);
   std::printf("\t\"beta\": %.10f,\n", beta);
   std::printf("\t\"biased_VaR\": %.10f,\n", risk_measures.VaR);
   std::printf("\t\"biased_ES\": %.10f,\n", risk_measures.ES);

   long int n; // n >> 1
   double scaler = 1;

   int n_runs = 5000;
   int i;

   double h_0 = 1./32; // precision < h_0 < 1
   double M = 2;
   int L = ml_sa_optimal_levels(precision, h_0, M);
   double h[L+1];
   long int N[L+1];

   // Averaged ML SA

   std::printf("\t\"avg_ml_sa_simulations\": {\n");
   std::printf("\t\t\"header\": [\"id\", \"status\", \"VaR\", \"VaR_avg\", \"ES\"],\n");
   std::printf("\t\t\"rows\": [\n");

   for (i = 0; i < n_runs; i++) {
      try {
         configure_avg_ml_sa(h_0, M, L, scaler, h, N);
         risk_measures = ml_sa(xi_0, chi_0, alpha, L, h, N, ml_sa_gamma, nested_simulator, ml_simulator);
         std::printf("\t\t\t[%d, \"success\", %.10f, %.10f, %.10f]",
                     i+1, risk_measures.VaR, risk_measures.VaR_avg, risk_measures.ES);
      } catch (const std::exception& e) {
         std::printf("\t\t\t[%d, \"failure\", \"%s\",\"na\",\"na\"]", i+1, e.what());
      }
      print_comma(i < n_runs-1);
   }
   std::printf("\t\t]\n");
   std::printf("\t},\n");

   // ML SA

   std::printf("\t\"ml_sa_simulations\": {\n");
   std::printf("\t\t\"header\": [\"id\", \"status\", \"VaR\", \"VaR_avg\", \"ES\"],\n");
   std::printf("\t\t\"rows\": [\n");

   for (i = 0; i < n_runs; i++) {
      try {
         configure_ml_sa(beta, h_0, M, L, scaler, h, N);
         risk_measures = ml_sa(xi_0, chi_0, alpha, L, h, N, ml_sa_gamma, nested_simulator, ml_simulator);
         std::printf("\t\t\t[%d, \"success\", %.10f, %.10f, %.10f]",
                     i+1, risk_measures.VaR, risk_measures.VaR_avg, risk_measures.ES);
      } catch (const std::exception& e) {
         std::printf("\t\t\t[%d, \"failure\", \"%s\",\"na\",\"na\"]", i+1, e.what());
      }
      print_comma(i < n_runs-1);
   }
   std::printf("\t\t]\n");
   std::printf("\t},\n");

   // Nested SA

   std::printf("\t\"nested_sa_simulations\": {\n");
   std::printf("\t\t\"header\": [\"id\", \"status\", \"VaR\", \"VaR_avg\", \"ES\"],\n");
   std::printf("\t\t\"rows\": [\n");

   for (i = 0; i < n_runs; i++) {
         try {
         n = nested_sa_optimal_steps(precision, scaler);
         risk_measures = nested_sa(xi_0, chi_0, alpha, precision, n, nested_sa_gamma, nested_simulator);
         std::printf("\t\t\t[%d, \"success\", %.10f, %.10f, %.10f]",
                     i+1, risk_measures.VaR, risk_measures.VaR_avg, risk_measures.ES);
      } catch (const std::exception& e) {
         std::printf("\t\t\t[%d, \"failure\", \"%s\",\"na\",\"na\"]", i+1, e.what());
      }
      print_comma(i < n_runs-1);
   }
   std::printf("\t\t]\n");
   std::printf("\t},\n");

   // SA

   std::printf("\t\"sa_simulations\": {\n");
   std::printf("\t\t\"header\": [\"id\", \"status\", \"VaR\", \"VaR_avg\", \"ES\"],\n");
   std::printf("\t\t\"rows\": [\n");

   for (i = 0; i < n_runs; i++) {
      try {
         n = sa_optimal_steps(precision, scaler);
         risk_measures = sa(xi_0, chi_0, alpha, n, sa_gamma, simulator);
         std::printf("\t\t\t[%d, \"success\", %.10f, %.10f, %.10f]",
                     i+1, risk_measures.VaR, risk_measures.VaR_avg, risk_measures.ES);
      } catch (const std::exception& e) {
         std::printf("\t\t\t[%d, \"failure\", \"%s\",\"na\",\"na\"]", i+1, e.what());
      }
      print_comma(i < n_runs-1);
   }
   std::printf("\t\t]\n");
   std::printf("\t}\n");

   std::printf("}\n");
}

void run() {
   double alpha = 0.85; // 0.0 < alpha < 1
   double precision = 1./256; // 0 < precision < 1
   double beta = 0.9; // 0 < beta < 1

   double r = 0.02;
   double S_0 = 100.0; // in basis points
   double kappa = 0.12;
   double sigma = 0.2;
   Time Delta = Time {y: 0, m: 3, d: 0};
   Time T = Time {y: 1, m: 0, d: 0}; 
   Time delta = Time {y: 0, m: 0, d: 7};
   double leg_0 = 1e4; // in basis points

   Swap_Simulator        simulator        (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_Nested_Simulator nested_simulator (r, S_0, kappa, sigma, Delta, T, delta, leg_0);
   Swap_ML_Simulator     ml_simulator     (r, S_0, kappa, sigma, Delta, T, delta, leg_0);

   double xi_0 = 2;
   double chi_0 = 3;

   Gamma sa_gamma(1, beta, 0L);
   Gamma nested_sa_gamma(0.1, beta, 250L);
   Gamma ml_sa_gamma(0.1, beta, 1500L);

   subroutine(xi_0, chi_0, alpha, beta, precision,
              sa_gamma, nested_sa_gamma, ml_sa_gamma,
              simulator, nested_simulator, ml_simulator);
}

int main(int argc, char* argv[]) {
   run();
}
