#ifndef INTEGRATE_TASK_4_INTEGRATION_H
#define INTEGRATE_TASK_4_INTEGRATION_H

#include <vector>
#include <cmath>
#include <limits>
#include <thread>

#include "conf.h"

void tbb_integration(const Config &conf, double (*func_to_integrate)(double, double),
                    double &outCalculatedIntegral, double &outAbsError, double &outRelError);


#endif //INTEGRATE_TASK_4_INTEGRATION_H
