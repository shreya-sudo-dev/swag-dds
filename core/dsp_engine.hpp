#ifndef DSP_ENGINE_HPP
#define DSP_ENGINE_HPP

#include "buffer.hpp"
#include <cstddef>

struct DoAResult {
    int angle;
    bool confident;
};

class DSPEngine {
public:
    DSPEngine();

    void push(const float* mic1,
              const float* mic2,
              const float* mic3,
              size_t count);

    bool ready() const;

    DoAResult process();

private:
    AudioBuffer b0_;
    AudioBuffer b1_;
    AudioBuffer b2_;
};

#endif
