#pragma once
#include <stdlib.h>
#include <time.h>
namespace RandUtil
{
    inline int randint(int min = 0, int max = RAND_MAX)
    {
        // srand(time(NULL));
        return rand() % max + min;
    }
} // namespace RandUtil
