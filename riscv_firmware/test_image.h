/**
 * test_image.h — PLACEHOLDER
 * Replace this file by running the training notebook (Step 10).
 * The notebook exports one validation image as an INT8 C array.
 *
 * For now: a dummy all-zero image (will result in "No Crack" prediction).
 */
#ifndef TEST_IMAGE_H
#define TEST_IMAGE_H

#include <stdint.h>

/* True label: set by notebook export */
#define TEST_IMG_TRUE_LABEL 0   /* NO_CRACK (placeholder) */

/* 64 x 64 x 3 = 12288 INT8 values */
static const int8_t g_test_image[64 * 64 * 3] = {
    /* All zeros — replace with notebook output */
    0
    /* C will zero-initialize the rest */
};

#endif /* TEST_IMAGE_H */
