#!/usr/bin/env python
# coding: utf-8

from os import system
import ctypes

C_SRC = """
#include <stdlib.h>
#include <stdint.h>

int64_t cdiv(int64_t a, int64_t b)
{
    return a/b;
}

int64_t cmod(int64_t a, int64_t b)
{
    return a%b;
}

#define LOOPY_CALL_WITH_INTEGER_TYPES(MACRO_NAME) \
    MACRO_NAME(short) \
    MACRO_NAME(int) \
    MACRO_NAME(long) \
    MACRO_NAME(int64_t)

#define LOOPY_DEFINE_FLOOR_DIV(TYPE) \
    TYPE loopy_floor_div_##TYPE(TYPE a, TYPE b) \
    { \
        if ((a<0) != (b<0)) \
            a = a - (b + (b<0) - (b>=0)); \
        return a/b; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV)
#undef LOOPY_DEFINE_FLOOR_DIV

#define LOOPY_DEFINE_FLOOR_DIV_POS_B(TYPE) \
    TYPE loopy_floor_div_pos_b_##TYPE(TYPE a, TYPE b) \
    { \
        if (a<0) \
            a = a - (b-1); \
        return a/b; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
#undef LOOPY_DEFINE_FLOOR_DIV_POS_B


#define LOOPY_DEFINE_MOD_POS_B(TYPE) \
    TYPE loopy_mod_pos_b_##TYPE(TYPE a, TYPE b) \
    { \
        TYPE result = a%b; \
        if (result < 0) \
            result += b; \
        return result; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD_POS_B)
#undef LOOPY_DEFINE_MOD_POS_B

#define LOOPY_DEFINE_MOD(TYPE) \
    TYPE loopy_mod_##TYPE(TYPE a, TYPE b) \
    { \
        TYPE result = a%b; \
        if (result < 0 && b > 0) \
            result += b; \
        if (result > 0 && b < 0) \
            result = result + b; \
        return result; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD)
#undef LOOPY_DEFINE_MOD


"""


def main():
    with open("int-experiments.c", "w") as outf:
        outf.write(C_SRC)

    system('gcc -Wall -shared int-experiments.c -o int-experiments.so')

    int_exp = ctypes.CDLL("int-experiments.so")
    for func in [
            int_exp.cdiv,
            int_exp.cmod,
            int_exp.loopy_floor_div_int64_t,
            int_exp.loopy_floor_div_pos_b_int64_t,
            int_exp.loopy_mod_pos_b_int64_t,
            int_exp.loopy_mod_int64_t,
            ]:
        func.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
        func.restype = ctypes.c_longlong

    cdiv = int_exp.cdiv  # noqa
    cmod = int_exp.cmod  # noqa
    int_floor_div = int_exp.loopy_floor_div_int64_t
    int_floor_div_pos_b = int_exp.loopy_floor_div_pos_b_int64_t
    int_mod_pos_b = int_exp.loopy_mod_pos_b_int64_t
    int_mod = int_exp.loopy_mod_int64_t

    m = 50

    for a in range(-m, m):
        for b in range(1, m):
            cresult = int_floor_div_pos_b(a, b)
            presult = a // b
            assert cresult == presult
            if cresult != presult:
                print(a, b, cresult, presult)

    for a in range(-m, m):
        for b in range(-m, m):
            if b == 0:
                continue

            cresult = int_floor_div(a, b)
            presult = a // b
            assert cresult == presult
            if cresult != presult:
                print(a, b, cresult, presult)

    for a in range(-m, m):
        for b in range(1, m):
            cresult = int_mod_pos_b(a, b)
            presult = a % b
            assert cresult == presult

    for a in range(-m, m):
        for b in range(-m, m):
            if b == 0:
                continue

            cresult = int_mod(a, b)
            presult = a % b
            assert cresult == presult
            if cresult != presult:
                print(a, b, cresult, presult)


if __name__ == "__main__":
    main()
