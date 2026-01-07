#include "buffer.hpp"
#include <algorithm>
#include <stdexcept>

AudioBuffer::AudioBuffer(size_t capacity)
    : buffer_(capacity),
      head_(0),
      size_(0),
      capacity_(capacity) {}

void AudioBuffer::push(const float* samples, size_t count)
{
    if (!samples || count == 0) return;

    for (size_t i = 0; i < count; ++i)
    {
        buffer_[head_] = samples[i];
        head_ = (head_ + 1) % capacity_;

        if (size_ < capacity_) 
        {
            size_++ ; 
        }
    }
}

std::vector<float> AudioBuffer::get_last(size_t count) const 
{
    if (count > size_) 
    {
        throw std::runtime_error("Not enough sampled in buffer");
    }

    std::vector<float> output(count);
    size_t start = (head_ + capacity_ - count) % capacity_;

    for (size_t i=0; i < count; ++i)
    {
        output[i] = buffer_[(start + i) % capacity_];
    }
    return output;
}

void AudioBuffer::clear() { head_ = 0; size_ = 0; }

size_t AudioBuffer::size() const { return size_;}

size_t AudioBuffer::capacity() const { return capacity_; }