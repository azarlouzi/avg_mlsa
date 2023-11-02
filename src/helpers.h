#ifndef _HELPERS_
#define _HELPERS_

#define IN
#define OUT

const double pi = 3.1415926535;

const int exception_message_length = 500;

struct Time {
   int y = 0;
   int m = 0;
   int d = 0;
};

double time_to_years(Time time);

double gaussian(double m = 0, double sd = 1);
double doleans_dade(double m = 0, double sd = 1);

double heaviside(double x);
double positive_part(double x);

double power(double x, int n);
double power(double x, double y);

#endif // _HELPERS_
