#pragma once
// https://stackoverflow.com/questions/1861294/how-to-calculate-execution-time-of-a-code-snippet-in-c
#include <iostream>
#include <chrono>

namespace TimeUtil
{
    class Timer
    {
    public:
        Timer() : beg_(clock_::now()) {}
        void reset() { beg_ = clock_::now(); }
        double elapsed() const
        {
            return std::chrono::duration_cast<second_>(clock_::now() - beg_).count();
        }

    private:
        typedef std::chrono::high_resolution_clock clock_;
        typedef std::chrono::duration<double, std::ratio<1>> second_;
        std::chrono::time_point<clock_> beg_;
    };
} // namespace TimeUtil