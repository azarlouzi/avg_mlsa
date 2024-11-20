#include "ml_sa.h"
#include <cmath>
#include <cstdio>
#include <exception>

class ML_Simulator_Exception : public std::exception {
public:
   ML_Simulator_Exception(double h_coarse, double h_fine): h_coarse(h_coarse), h_fine(h_fine) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Multilevel simulation bias parameters should satisfy h_coarse > h_fine > 0. Got instead: h_coarse = %.10f, h_fine = %.10f", h_coarse, h_fine);
      return msg;
   }
private:
   double h_coarse;
   double h_fine;
};

void ML_Simulator::set_bias_parameters(double h_coarse, double h_fine) {
   this->h_coarse = h_coarse;
   this->h_fine = h_fine;
   verify_bias_parameters();
}

void ML_Simulator::verify_bias_parameters() const {
   if (!(h_coarse > 0) || !(h_fine > 0) || !(h_coarse > h_fine)) {
      throw ML_Simulator_Exception(h_coarse, h_fine);
   }
}

class ML_SA_Optimal_Levels_Exception : public std::exception {
public:
   ML_SA_Optimal_Levels_Exception(double precision, double h_0):
      precision(precision), h_0(h_0) {}
   const char* what() const throw() override {
      char* msg = new char[exception_message_length];
      std::sprintf(msg, "Parameters h_0 and precision should satisfy: precision < h_0. Got instead: precision = %.10f, h_0 = %.10f.", precision, h_0);
      return msg;
   }

private:
   double precision;
   double h_0;
};

void verify_ml_sa_optimal_levels(double precision, double h_0) {
   if (!(precision < h_0)) {
      throw ML_SA_Optimal_Levels_Exception(precision, h_0);
   }
}

int ml_sa_optimal_levels(double precision, double h_0, double M) {
   verify_ml_sa_optimal_levels(precision, h_0);
   return int(std::ceil(std::log(h_0/precision)/std::log(M)));
}

void configure_h(IN     double  h_0,
                 IN     double  M,
                 IN     int     L,
                    OUT double* h) {
   h[0] = h_0;
   for (int l = 1; l < L+1; l++) {
      h[l] = h[l-1]/M;
   }
}

void configure_ml_sa(IN     double    beta,
                     IN     double    h_0,
                     IN     double    M,
                     IN     int       L,
                     IN     double    scaler,
                        OUT double*   h,
                        OUT long int* N) {
   configure_h(h_0, M, L, h);

   double tmp = 0;
   for (int l = 0; l < L+1; l++) {
      tmp += std::pow(h[l], (1 - 2*beta)/(2*(1 + beta)));
   }

   tmp = std::pow(tmp, 1./beta);
   tmp *= std::pow(h[L], -2./beta);
   for (int l = 0; l < L+1; l++) {
      N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], 3./(2*(1 + beta))));
   }
}

void configure_avg_ml_sa(IN     double    h_0,
                         IN     double    M,
                         IN     int       L,
                         IN     double    scaler,
                            OUT double*   h,
                            OUT long int* N) {
   configure_h(h_0, M, L, h);

   double tmp = 0;
   for (int l = 0; l < L+1; l++) {
      tmp += std::pow(h[l], -0.25);
   }

   tmp *= std::pow(h[L], -2.0);
   for (int l = 0; l < L+1; l++) {
      N[l] = (long int) std::ceil(scaler * tmp * std::pow(h[l], 0.75));
   }
}

Risk_Measures ml_sa(IN     double            xi_0,
                    IN     double            chi_0,
                    IN     double            alpha,
                    IN     int               L,
		    IN     double            M,
		    IN     double            beta,
                    IN     const double*     h,
                    IN     const long int*   N,
                    IN     const Step&       step,
		    IN     bool              average_out,
                    IN OUT Nested_Simulator& simulator,
                    IN OUT ML_Simulator&     ml_simulator) {
   double var1, var2, tmp, V1, V2;
   V1 = V2 = 0;

   double xi[L+1][2];
   double xi_avg[L+1][2];
   double chi[L+1][2];
   for (int l = 0; l < L+1; l++) {
      xi[l][0] = xi_0;
      xi[l][1] = xi_0;
      xi_avg[l][0] = 0;
      xi_avg[l][1] = 0;
      chi[l][0] = chi_0;
      chi[l][1] = chi_0;
   }

   simulator.set_bias_parameter(h[0]);
   double X;

   for (long int i = 0L; i < N[0]; i++) {
      X = simulator();

      tmp = positive_part(X - xi[0][0]);
      V1 += std::pow(tmp, 2)/double(N[0]);
      V2 += tmp/double(N[0]);

      chi[0][0] = chi[0][0] - H_2(alpha, chi[0][0], xi[0][0], X)/double(i+1L);
      xi[0][0] = xi[0][0] - step(i+1L)*H_1(alpha, xi[0][0], X);
      xi_avg[0][0] = (1-1/double(i+1L))*xi_avg[0][0] + xi[0][0]/double(i+1L);
   }

   var1 = V1 - std::pow(V2, 2);

   ML_Simulations X_ml;
   V1 = V2 = 0;
   for (int l = 1; l < L+1; l++) {
      ml_simulator.set_bias_parameters(h[l-1], h[l]);
      for (long int i = 0L; i < N[l]; i++) {
         X_ml = ml_simulator();

         if (l == L) {
	    tmp = heaviside(X_ml.fine - xi[L][1])*X_ml.G;
	    V1 += std::pow(tmp, 2)/double(N[L]);
	    V2 += tmp/double(N[L]);
	 }

         chi[l][0] = chi[l][0] - H_2(alpha, chi[l][0], xi[l][0], X_ml.coarse)/double(i+1L);
         chi[l][1] = chi[l][1] - H_2(alpha, chi[l][1], xi[l][1], X_ml.fine)/double(i+1L);
         xi[l][0] = xi[l][0] - step(i+1L)*H_1(alpha, xi[l][0], X_ml.coarse);
         xi[l][1] = xi[l][1] - step(i+1L)*H_1(alpha, xi[l][1], X_ml.fine);
         xi_avg[l][0] = (1-1/double(i+1L))*xi_avg[l][0] + xi[l][0]/double(i+1L);
         xi_avg[l][1] = (1-1/double(i+1L))*xi_avg[l][1] + xi[l][1]/double(i+1L);
      }
   }

   var2 = V1 - std::pow(V2, 2);

   double xi_ml = xi[0][0];
   double xi_avg_ml = xi_avg[0][0];
   double chi_ml = chi[0][0];
   for (int l = 1; l < L+1; l++) {
      xi_ml += xi[l][1] - xi[l][0];
      xi_avg_ml += xi_avg[l][1] - xi_avg[l][0];
      chi_ml += chi[l][1] - chi[l][0];
   }

   double var, pow;
   if (average_out) {
      var = var1*std::pow(std::pow(M, 1./4) - 1, 1./2)/std::pow(std::pow(h[0], 3)*M, 1./8);
      var += var2*std::pow(h[0]/M, 1./4);
      var /= std::pow(1 - alpha, 2);
   } else {
      pow = (2*beta-1)/(2*(1+beta));
      var = var1/(h[0]*std::pow(M, pow/beta));
      var += var2/(std::pow(M, pow) - 1);
      var *= std::pow(h[0], pow)*std::pow(std::pow(M, pow) - 1, 1./beta)/std::pow(1 - alpha, 2);
   }

   return Risk_Measures {
      .VaR = xi_ml,
      .VaR_avg = xi_avg_ml,
      .ES = chi_ml,
      .ES_var = var,
   };
}
