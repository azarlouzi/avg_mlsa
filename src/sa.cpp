#include "sa.h"
#include <cmath>

double Gamma::operator()(long int n) const {
   return gamma_1 / (std::exp(beta * std::log(double(smoothing + n))));
}

double H_1(double alpha, double xi, double x) {
   return 1 - heaviside(x-xi)/(1-alpha);
}

double H_2(double alpha, double chi, double xi, double x) {
   return chi - (xi + positive_part(x-xi)/(1-alpha));
}

long int sa_optimal_steps(double precision, double scaler) {
   return (long int) (scaler/(std::pow(precision, 2)));
}

Risk_Measures sa(IN double           xi_0,
                 IN double           chi_0,
                 IN double           alpha,
                 IN long int         n,
                 IN const Step&      step,
                 IN const Simulator& simulator) {
   double xi = xi_0;
   double xi_avg = 0;
   double chi = chi_0;
   double X_0;
   for (long int i = 0L; i < n; i++) {
      X_0 = simulator();
      chi = chi - H_2(alpha, chi, xi, X_0)/double(i+1L);
      xi = xi - step(i+1L)*H_1(alpha, xi, X_0);
      xi_avg = (1 - 1/double(i+1L))*xi_avg + xi/double(i+1L);
   }
   return Risk_Measures {
      .VaR = xi,
      .VaR_avg = xi_avg,
      .ES = chi,
   };
}

