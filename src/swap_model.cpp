#include "swap_model.h"
#include "helpers.h"
#include <cmath>
#include <cstdio>

double sd_amortization(double rate, double time) {
   return std::sqrt((1-std::exp(-2*rate*time))/(2*rate));
}

double Discount::operator()(double t) const {
   return std::exp(-r*t);
}

double Reset::operator()(int i) const {
   return Delta*i;
}

Swap_Simulator::Swap_Simulator(double r,
                               double S_0,
                               double kappa,
                               double sigma,
                               Time   Delta,
                               Time   T,
                               Time   delta,
                               double leg_0):
   Simulator(),
   r(r),
   S_0(S_0),
   kappa(kappa),
   sigma(sigma),
   Delta(time_to_years(Delta)),
   T(time_to_years(T)),
   delta(time_to_years(delta)),
   leg_0(leg_0),
   d(int(this->T/this->Delta)),
   discount(r),
   reset(this->Delta)
{
   double tmp = 0;

   for (int i = 1; i <= d; i++) {
      tmp += discount(reset(i))*(this->Delta)*std::exp(kappa*reset(i-1));
   }
   nominal = leg_0/(S_0*tmp);

   tmp -= discount(reset(1))*(this->Delta);
   tmp *= sd_amortization(kappa, this->delta);
   factor = nominal*sigma*tmp;
}

double Swap_Simulator::operator()() const {
   double simulation = factor*gaussian();
   return simulation;
}

Swap_Nested_Payoff::Swap_Nested_Payoff(double r,
                                       double S_0,
                                       double kappa,
                                       double sigma,
                                       double Delta,
                                       double leg_0,
                                       int    d):
   r(r),
   S_0(S_0),
   kappa(kappa),
   sigma(sigma),
   Delta(Delta),
   leg_0(leg_0),
   d(d),
   discount(r),
   reset(Delta)
{
   double tmp = 0;
   for (int i = 1; i <= d; i++) {
      tmp += discount(reset(i))*(this->Delta)*std::exp(kappa*reset(i-1));
   }
   nominal = leg_0/(S_0*tmp);
}

double Swap_Nested_Payoff::operator()(double y, double* z) const {
   double payoff = 0;
   double tmp = y;
   for (int i = 2; i <= d; i++) {
      tmp += z[i-2];
      payoff += discount(reset(i))*Delta*std::exp(kappa*reset(i-1))*tmp;
   }
   payoff *= nominal*sigma;
   return payoff;
}

Swap_Nested_Simulator::Swap_Nested_Simulator(double r,
                                             double S_0,
                                             double kappa,
                                             double sigma,
                                             Time   Delta,
                                             Time   T,
                                             Time   delta,
                                             double leg_0):
   Nested_Simulator(),
   kappa(kappa),
   sigma(sigma),
   Delta(time_to_years(Delta)),
   T(time_to_years(T)),
   delta(time_to_years(delta)),
   d(int(this->T/this->Delta)),
   phi(r, S_0, kappa, sigma, this->Delta, leg_0, d) {}

double Swap_Nested_Simulator::operator()() const {
   double Y = sd_amortization(kappa, delta)*gaussian();
   double Z[d-1];

   long int K = (long int) std::ceil(1./h);
   double X_h = 0;
   for (long int k = 0L; k < K; k++) {
      Z[0] = sd_amortization(kappa, Delta-delta)*gaussian();
      for (int i = 1; i < d-1; i++) {
         Z[i] = sd_amortization(kappa, Delta)*gaussian();
      }
      X_h += phi(Y, Z)/double(K);
   }

   return X_h;
}

Swap_ML_Simulator::Swap_ML_Simulator(double r,
                                     double S_0,
                                     double kappa,
                                     double sigma,
                                     Time   Delta,
                                     Time   T,
                                     Time   delta,
                                     double leg_0):
   ML_Simulator(),
   kappa(kappa),
   sigma(sigma),
   Delta(time_to_years(Delta)),
   T(time_to_years(T)),
   delta(time_to_years(delta)),
   d(int(this->T/this->Delta)),
   phi(r, S_0, kappa, sigma, this->Delta, leg_0, d) {}

ML_Simulations Swap_ML_Simulator::operator()() const {
   double Y = sd_amortization(kappa, delta)*gaussian();
   double Z[d-1];

   long int K_coarse = (long int) std::ceil(1./h_coarse);
   double X_h_coarse = 0;
   for (long int k = 0L; k < K_coarse; k++) {
      Z[0] = sd_amortization(kappa, Delta-delta)*gaussian();
      for (int i = 1; i < d-1; i++) {
         Z[i] = sd_amortization(kappa, Delta)*gaussian();
      }
      X_h_coarse += phi(Y, Z)/double(K_coarse);
   }

   long int K_fine = (long int) std::ceil(1./h_fine);
   double X_h_fine = X_h_coarse*double(K_coarse)/double(K_fine);
   for (long int k = 0L; k < (K_fine - K_coarse); k++) {
      Z[0] = sd_amortization(kappa, Delta-delta)*gaussian();
      for (int i = 1; i < d-1; i++) {
         Z[i] = sd_amortization(kappa, Delta)*gaussian();
      }
      X_h_fine += phi(Y, Z)/double(K_fine);
   }

   return ML_Simulations {
      .coarse = X_h_coarse,
      .fine = X_h_fine,
   };
}
