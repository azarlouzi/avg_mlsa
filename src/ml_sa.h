#ifndef _ML_SA_
#define _ML_SA_

#include "helpers.h"
#include "nested_sa.h"
#include "sa.h"

struct ML_Simulations {
   double coarse;
   double fine;
   double G;
};

class ML_Simulator {
public:
   void set_bias_parameters(double h_coarse, double h_fine);
   virtual ML_Simulations operator()() const=0;
protected:
   double h_coarse = 1;
   double h_fine = 0.5;
   void verify_bias_parameters() const;
};

int ml_sa_optimal_levels(double precision, double h_0, double M);

void configure_ml_sa(IN     double    beta,
                     IN     double    h_0,
                     IN     double    M,
                     IN     int       L,
                     IN     double    scaler,
                        OUT double*   h,
                        OUT long int* N);

void configure_avg_ml_sa(IN     double    h_0,
                         IN     double    M,
                         IN     int       L,
                         IN     double    scaler,
                            OUT double*   h,
                            OUT long int* N);

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
                    IN OUT ML_Simulator&     ml_simulator);

#endif // _ML_SA_
