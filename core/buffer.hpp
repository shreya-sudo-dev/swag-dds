#ifndef AUDIO_BUFFER_HPP
#define AUDIO_BUFFER_HPP

#include <vector>
#include <cstddef>

class AudioBuffer {
    public:
        explicit AudioBuffer(size_t capacity);
        void push(const float* samples, size_t count);
        std::vector<float> get_last(size_t count) const;
        void clear();
        size_t size() const;
        size_t capacity() const;

    private:
        std::vector<float> buffer_;
        size_t head_;
        size_t size_;
        size_t capacity_;
};

#endif