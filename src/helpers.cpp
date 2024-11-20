#include "helpers.h"
#include <chrono>
#include <cmath>
#include <ctime>
#include <random>

double time_to_years(Time time) {
   return double(time.y*360 + time.m*30 + time.d)/360.0;
}

unsigned long now() {
   std::time_t time = std::time(NULL); // long int
   return (unsigned long) (time);
}

std::default_random_engine generator(now());

double gaussian(double m, double sd) {
   std::normal_distribution<double> distribution(m, sd);
   return distribution(generator);
}

double doleans_dade(double m, double sd) {
   double U = gaussian(m, sd);
   return std::exp(U-std::pow(sd, 2.)/2.);
}

double heaviside(double x) {
   return x >= 0 ? 1 : 0;
}

double positive_part(double x) {
   return x > 0 ? x : 0;
}
