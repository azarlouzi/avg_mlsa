#include "nested_sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

class Nested_Simulator_Exception: public std::exception {
public:
   Nested_Simulator_Exception(double h): h(h) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Simulation bias parameter should satisfy: h > 0. Got instead: h = %.10f", h);
      return msg;
   }
private:
   double h;
};

void Nested_Simulator::set_bias_parameter(double h) {
   this->h = h;
   verify_bias_parameter();
}

void Nested_Simulator::verify_bias_parameter() const {
   if (!(h > 0)) {
      throw Nested_Simulator_Exception(h);
   }
}

long int nested_sa_optimal_steps(double precision, double scaler) {
   return sa_optimal_steps(precision, scaler);
}

Risk_Measures nested_sa(IN     double            xi_0,
                        IN     double            chi_0,
                        IN     double            alpha,
                        IN     double            h,
                        IN     long int          n,
                        IN     const Step&       step,
                        IN OUT Nested_Simulator& simulator) {
   simulator.set_bias_parameter(h);
   double xi = xi_0;
   double xi_avg = 0;
   double chi = chi_0;
   double var, V1, V2, tmp;
   V1 = V2 = tmp = 0;

   double X_h;
   for (long int i = 0L; i < n; i++) {
      X_h = simulator();

      tmp = positive_part(X_h - xi);
      V1 += std::pow(tmp, 2)/double(n);
      V2 += tmp/double(n);

      chi = chi - H_2(alpha, chi, xi, X_h)/double(i+1L);
      xi = xi - step(i+1L)*H_1(alpha, xi, X_h);
      xi_avg = (1 - 1/double(i+1L))*xi_avg + xi/double(i+1L);
   }
   var = (V1 - std::pow(V2, 2))/std::pow(1 - alpha, 2);

   return Risk_Measures {
      .VaR = xi,
      .VaR_avg = xi_avg,
      .ES = chi,
      .ES_var = var,
   };
}
