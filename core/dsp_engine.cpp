#include "dsp_engine.hpp"
#include <vector>
#include <algorithm>
#include <complex>

#define _USE_MATH_DEFINES
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float C = 343.0f;
static constexpr float D = 0.15f;
static constexpr int   FS = 44100;
static constexpr int   W  = 4096;

static float clamp(float x, float lo, float hi) {
    return std::max(lo, std::min(x, hi));
}

static void fft(std::vector<std::complex<float>>& a) {
    size_t n = a.size(); 
    if (n <= 1) return;

    std::vector<std::complex<float>> even(n/2), odd(n/2);
    for(size_t i=0; i<n/2; ++i)
    {
        even[i] = a[i *2];
        odd[i] = a[i*2+1];
    }

    fft(even);
    fft(odd);

    for (size_t i=0; i<n/2;++i) {
        std::complex<float> t = std::polar(1.0f, (float)(-2 * M_PI * i/n)) * odd[i];
        a[i] = even[i]+t;
        a[i + n / 2] = even[i] -t;
    }
}

static void ifft(std::vector<std::complex<float>>& a) {
    for(auto& x : a) x = std::conj(x);

    fft(a);

    float scale = 1.0f / a.size();
    for (auto& x : a) x = std::conj(x) * scale;
} 

static float gcc_phat(const std::vector<float>& sig, const std::vector<float>& ref)
{
    size_t N = sig.size() + ref.size();
    std::vector<std::complex<float>> X(N, 0), Y(N, 0);

    for(size_t i=0; i< sig.size(); ++i) X[i] = sig[i];
    for(size_t i=0; i<ref.size(); ++i) Y[i] = ref[i];

    fft(X);
    fft(Y);

    std::vector<std::complex<float>> R(N);
    for(size_t i=0; i<N; ++i) {
        auto num =X[i] * std::conj(Y[i]);
        float mag = std::abs(num);
        R[i] = (mag > 1e-6f) ? num / mag : std::complex<float>(0,0);
    }

    ifft(R);

    int best = 0;
    float peak = 0.0f;

    for(int i= -int(N/2); i< int(N/2); ++i) {
        int idx = (i+N) %N;
        float v =std::abs(R[idx]);
        if( v > peak) {
            peak = v;
            best = i;
        }
    }
    return float(best) / FS;
}


DSPEngine::DSPEngine()
    : b0_(W*2),
      b1_(W*2),
      b2_(W*2) {}
    
void DSPEngine::push(const float* m0, const float* m1, const float* m2, size_t n)
{
    b0_.push(m0,n);
    b1_.push(m1,n);
    b2_.push(m2,n);
}


bool DSPEngine::ready() const {
    return b0_.size() >= W &&
           b1_.size() >= W &&
           b2_.size() >= W;
}

DoAResult DSPEngine::process()
{
    auto m0 = b0_.get_last(W);
    auto m1 = b1_.get_last(W);
    auto m2 = b2_.get_last(W);

    float t01 = gcc_phat(m1, m0);
    float t02 = gcc_phat(m2, m0);

    float a1 = std::asin(clamp((C*t01)/D, -1, 1)) * 180/M_PI;
    float a2 = std::asin(clamp((C*t02)/(D * 0.5f), -1, 1)) * 180/M_PI;

    bool confident = std::abs(a1 - a2) < 15.0f;

    return { int(a1), confident };
}