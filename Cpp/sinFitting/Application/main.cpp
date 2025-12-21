#include <matplot/matplot.h>

#include <Eigen/Dense>

#include <random>
#include <numeric>
#include <algorithm>

#include <nlopt.hpp>

double estFunc(double x, double A, double f, double phi, double bias) {
    return A * sin(2 * matplot::pi * x * f + phi) + bias;
}


int main() {
    using namespace matplot;
    static constexpr std::size_t numberPoints = 2000;

    static constexpr float A = 1;
    static constexpr float freq = 5;
    static constexpr float phi = 60 * pi / 180;
    static constexpr float bias = 0.5;

    static const double mean = 0.0;
    static const double stddev = 0.1;
    std::default_random_engine generator;
    std::normal_distribution<double> dist(mean, stddev);

    std::vector<double> x  = linspace(0, 2 * pi, numberPoints);
    std::vector<double> sinY = transform(x, [&](auto x) { return estFunc(x, A, freq, phi, bias); });
    std::vector<double> sinYNoise = transform(sinY, [&](auto x) { return x + dist(generator); });

    nlopt::opt opt(nlopt::GD_MLSL, 4);

    opt.set_min_objective([&](unsigned n, const double* params, double* grad) {
        double result = 0.f;

        if (grad) {
            grad[0] = 0.f;
            grad[1] = 0.f;
            grad[2] = 0.f;
            grad[3] = 0.f;
        }

        for (std::size_t i = 0; i < numberPoints; ++i) {
            double estY = estFunc(x[i], params[0], params[1], params[2], params[3]);
            double del = sinYNoise[i] - estY;
            result += (del * del);

            if (grad) {
                double coef = -2 * del;
                double sinDf = sin(2 * matplot::pi * x[i] * params[1] + params[2]);
                double cosDf = cos(2 * matplot::pi * x[i] * params[1] + params[2]);
                grad[0] += coef * sinDf;
                grad[1] += coef * params[0] * cosDf * 2 * matplot::pi * x[i];
                grad[2] += coef * params[0] * cosDf;
                grad[3] += coef;
            }
        }

        return result;
    });

    double infinity = 10;

    auto minAmp = *std::min_element(sinYNoise.begin(), sinYNoise.end());
    auto maxAmp = *std::max_element(sinYNoise.begin(), sinYNoise.end());
    auto minTime = *std::min_element(x.begin(), x.end());
    auto maxTime = *std::max_element(x.begin(), x.end());
    double meanValue = std::accumulate(sinYNoise.begin(), sinYNoise.end(), 0.0) / sinYNoise.size();
    auto maxFreq = x.size() / (maxTime - minTime);

    std::vector<double> lowerBounds = {minAmp, 0, -matplot::pi, minAmp};
    std::vector<double> upperBounds = {maxAmp, maxFreq, matplot::pi, maxAmp};
    opt.set_upper_bounds(upperBounds);
    opt.set_lower_bounds(lowerBounds);


    auto initialAmp = (maxAmp - minAmp) / 2;
    auto initialC = meanValue;
    
    std::vector<double> value = {initialAmp, maxFreq / 2, 0.1, initialC};
    auto result = opt.optimize(value);


    std::vector<double> estimateY {};

    if (opt.last_optimize_result() == nlopt::SUCCESS) {
        estimateY = transform(x, [&](auto x) { return estFunc(x, result[0], result[1], result[2], result[3]); });
    }

    hold(on);
    // legend({"Original", "Noisy", "Estimated"});
    plot(x, sinY);
    plot(x, sinYNoise);
    plot(x, estimateY);
    hold(off);
    grid(on);

    show();
    return 0;
}