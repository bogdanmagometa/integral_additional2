#include "integration.h"

#include <cmath>
#include <limits>
#include <thread>
#include <oneapi/tbb/flow_graph.h>

static double sum_over(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                double y_start, double y_end, double delta_y);

struct integration_descr {
    double start_x;
    double start_y;
    double delta_x;
    double delta_y;
    int steps_x;
    int steps_y;
};

class Summer {
public:
    Summer(const Summer &s) {
        summ = s.summ;
    }
    Summer(double *summ_ptr): summ{summ_ptr} {}
    oneapi::tbb::flow::continue_msg operator()(double num) {
        std::lock_guard<std::mutex> lock{m};
        *summ += num;
        return oneapi::tbb::flow::continue_msg{};
    }
private:
    double *summ;
    std::mutex m;
};

void tbb_integration(const Config &conf, double (*func_to_integrate)(double, double),
                    double &outCalculatedIntegral, double &outAbsError, double &outRelError) {
    namespace fl = oneapi::tbb::flow;
    using multi_node = fl::multifunction_node<integration_descr,
            std::tuple<integration_descr>>;

    fl::graph g;

    multi_node splitter(g, fl::unlimited, [] (integration_descr descr, multi_node::output_ports_type &op) {
        int num_parts = 12;
        double cur_start_x = descr.start_x;
        double cur_end_x = descr.start_y;
        for (int n_part = 0; n_part < num_parts; ++n_part) {
            int num_columns = descr.steps_x / num_parts + (n_part < descr.steps_x % num_parts);
            cur_end_x += num_columns * descr.delta_x;
            std::get<0>(op).try_put(integration_descr{cur_start_x, descr.start_y,
                                                      descr.delta_x, descr.delta_y,
                                                      num_columns, descr.steps_y});
            cur_start_x += num_columns * descr.delta_x;
        }

    });

    fl::function_node<integration_descr, double> calculator(g, fl::unlimited, [&func_to_integrate] (integration_descr descr) {
        return sum_over(func_to_integrate, descr.start_x, descr.start_x + descr.delta_x * descr.steps_x, descr.delta_x,
                                    descr.start_y, descr.start_y + descr.delta_y * descr.steps_y, descr.delta_y);
    });

    double summ = 0;
    fl::function_node<double, fl::continue_msg> summer_node(g, fl::serial, Summer{&summ});

    fl::make_edge(std::get<0>(splitter.output_ports()), calculator);
    fl::make_edge(calculator, summer_node);

    int steps_x = conf.init_steps_x;
    int steps_y = conf.init_steps_y;
    double width = conf.x_end - conf.x_start;
    double height = conf.y_end - conf.y_start;
    double delta_x = width / conf.init_steps_x;
    double delta_y = height / conf.init_steps_y;

    // Calculating the integral for the first time
    summ = 0;
    splitter.try_put(integration_descr{conf.x_start, conf.y_start, delta_x, delta_y, steps_x, steps_y});
    g.wait_for_all();
    outCalculatedIntegral = summ * delta_x * delta_y;

    // Approximating the value of integral by increasing number of partitions
    outAbsError = std::numeric_limits<double>::infinity();
    outRelError = std::numeric_limits<double>::infinity();
    int n_iter = 0;
    while (n_iter < conf.max_iter && !(outAbsError < conf.abs_err && outRelError < conf.rel_err)) {
#ifdef DEBUG_OUTPUT
        std::cout << "Integral on iteration #" << n_iter << " = " << outCalculatedIntegral << std::endl;
#endif
        double prev_integral_val = outCalculatedIntegral;

        steps_x *= 2;
        steps_y *= 2;
        delta_x = width / steps_x;
        delta_y = height / steps_y;

        summ = 0;
        splitter.try_put(integration_descr{conf.x_start + delta_x, conf.y_start, delta_x * 2, delta_y * 2, steps_x / 2, steps_y / 2});
        splitter.try_put(integration_descr{conf.x_start, conf.y_start + delta_y, delta_x * 2, delta_y * 2, steps_x / 2, steps_y / 2});
        splitter.try_put(integration_descr{conf.x_start + delta_x, conf.y_start + delta_y, delta_x * 2, delta_y * 2, steps_x / 2, steps_y / 2});
        g.wait_for_all();
        outCalculatedIntegral = outCalculatedIntegral / 4 + summ * delta_x * delta_y;

        ++n_iter;

        outAbsError = std::abs(outCalculatedIntegral - prev_integral_val);

        if (prev_integral_val == 0.0) { // division by 0.0 is UB
            outRelError = std::numeric_limits<double>::infinity();
        } else {
            outRelError = outAbsError / std::abs(prev_integral_val);
        }
    }
}

static double sum_over(double (*func_to_sum)(double, double), double x_start, double x_end, double delta_x,
                       double y_start, double y_end, double delta_y) {
    double sum = 0;
    constexpr double epsilon = 1E-10;
    for (double x_coord = x_start; x_end - x_coord > epsilon; x_coord += delta_x) {
        for (double y_coord = y_start; y_end - y_coord > epsilon; y_coord += delta_y) {
            sum += func_to_sum(x_coord, y_coord);
        }
    }
    return sum;
}
