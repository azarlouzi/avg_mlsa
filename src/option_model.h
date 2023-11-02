#ifndef _OPTION_MODEL_
#define _OPTION_MODEL_

#include "ml_sa.h"
#include "nested_sa.h"
#include "sa.h"

class Option_Simulator: public Simulator {
public:
   Option_Simulator(double delta = 0.5, double T = 1);
   double operator()() const override;
private:
   double delta, T;
};

class Option_Nested_Payoff {
public:
   Option_Nested_Payoff(double delta, double T);
   double operator()(double y, double z) const;
private:
   double delta, T;
};

class Option_Nested_Simulator: public Nested_Simulator {
public:
   Option_Nested_Simulator(double delta, double T);
   double operator()() const override;
private:
   double T;
   Option_Nested_Payoff phi;
};

class Option_ML_Simulator: public ML_Simulator {
public:
   Option_ML_Simulator(double delta, double T);
   ML_Simulations operator()() const override;
private:
   Option_Nested_Payoff phi;
};

#endif // _OPTION_MODEL_
