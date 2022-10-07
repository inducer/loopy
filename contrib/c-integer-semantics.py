#!/usr/bin/env python

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
    MACRO_NAME(int8, char) \
    MACRO_NAME(int16, short) \
    MACRO_NAME(int32, int) \
    MACRO_NAME(int64, long long)

#define LOOPY_DEFINE_FLOOR_DIV(SUFFIX, TYPE) \
    TYPE loopy_floor_div_##SUFFIX(TYPE a, TYPE b) \
    { \
        if ((a<0) != (b<0)) \
            a = a - (b + (b<0) - (b>=0)); \
        return a/b; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV)
#undef LOOPY_DEFINE_FLOOR_DIV

#define LOOPY_DEFINE_FLOOR_DIV_POS_B(SUFFIX, TYPE) \
    TYPE loopy_floor_div_pos_b_##SUFFIX(TYPE a, TYPE b) \
    { \
        if (a<0) \
            a = a - (b-1); \
        return a/b; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_FLOOR_DIV_POS_B)
#undef LOOPY_DEFINE_FLOOR_DIV_POS_B


#define LOOPY_DEFINE_MOD_POS_B(SUFFIX, TYPE) \
    TYPE loopy_mod_pos_b_##SUFFIX(TYPE a, TYPE b) \
    { \
        TYPE result = a%b; \
        if (result < 0) \
            result += b; \
        return result; \
    }

LOOPY_CALL_WITH_INTEGER_TYPES(LOOPY_DEFINE_MOD_POS_B)
#undef LOOPY_DEFINE_MOD_POS_B

#define LOOPY_DEFINE_MOD(SUFFIX, TYPE) \
    TYPE loopy_mod_##SUFFIX(TYPE a, TYPE b) \
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

    system("gcc -Wall -shared int-experiments.c -o int-experiments.so")

    int_exp = ctypes.CDLL("int-experiments.so")
    for func in [
            int_exp.cdiv,
            int_exp.cmod,
            int_exp.loopy_floor_div_int64,
            int_exp.loopy_floor_div_pos_b_int64,
            int_exp.loopy_mod_pos_b_int64,
            int_exp.loopy_mod_int64,
            ]:
        func.argtypes = [ctypes.c_longlong, ctypes.c_longlong]
        func.restype = ctypes.c_longlong

    cdiv = int_exp.cdiv  # noqa
    cmod = int_exp.cmod  # noqa
    int_floor_div = int_exp.loopy_floor_div_int64
    int_floor_div_pos_b = int_exp.loopy_floor_div_pos_b_int64
    int_mod_pos_b = int_exp.loopy_mod_pos_b_int64
    int_mod = int_exp.loopy_mod_int64

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

    #print(int_mod(552, -918), 552 % -918)
    print(cmod(23, -11), 23 % -11)


if __name__ == "__main__":
    main()
