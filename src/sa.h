#ifndef _SA_
#define _SA_

#include "helpers.h"

struct Risk_Measures {
   double VaR = 0;
   double VaR_avg = 0;
   double ES = 0;
};

class Simulator {
public:
   virtual double operator()() const=0;
};

class Step {
public:
   virtual double operator()(long int p) const=0;
};

class Gamma: public Step {
public:
   Gamma(double gamma_1 = 1, double beta = 1, long int smoothing = 0L):
      gamma_1(gamma_1), beta(beta), smoothing(smoothing) {}
   double operator()(long int p) const override;
private:
   double gamma_1;
   double beta;
   long int smoothing;
};

double H_1(double alpha, double xi, double x);
double H_2(double alpha, double chi, double xi, double x);

long int sa_optimal_steps(double precision, double scaler);

Risk_Measures sa(IN double           xi_0,
                 IN double           chi_0,
                 IN double           alpha,
                 IN long int         n,
                 IN const Step&      step,
                 IN const Simulator& simulator);

#endif // _SA_
