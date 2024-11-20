#ifndef _NESTED_SA_
#define _NESTED_SA_

#include "helpers.h"
#include "sa.h"

class Nested_Simulator {
public:
   void set_bias_parameter(double h);
   virtual double operator()() const=0;
protected:
   double h = 1;
   void verify_bias_parameter() const;
};

long int nested_sa_optimal_steps(double precision, double scaler);

Risk_Measures nested_sa(IN     double            xi_0,
                        IN     double            chi_0,
                        IN     double            alpha,
                        IN     double            h,
                        IN     long int          n,
                        IN     const Step&       step,
                        IN OUT Nested_Simulator& simulator);

#endif // _NESTED_SA_
