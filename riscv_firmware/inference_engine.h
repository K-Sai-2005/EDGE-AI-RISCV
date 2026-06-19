/**
 * inference_engine.h
 * Lightweight INT8 inference engine for bare-metal RISC-V
 * No heap allocation. No floating-point. No OS dependency.
 *
 * Supports:
 *   - Fully-connected (Dense) layers with INT8 weights & biases
 *   - ReLU and Sigmoid activations (integer approximations)
 *   - Global Average Pooling output (pre-computed in feature vector)
 */

#ifndef INFERENCE_ENGINE_H
#define INFERENCE_ENGINE_H

#include <stdint.h>
#include <stddef.h>

/* ── Status codes ───────────────────────────────────────────────────── */
typedef enum {
    INFERENCE_OK            = 0,
    INFERENCE_ERR_NULL_PTR  = 1,
    INFERENCE_ERR_BAD_MODEL = 2,
    INFERENCE_ERR_SIZE      = 3
} InferenceStatus;

/* ── Layer descriptor ───────────────────────────────────────────────── */
typedef struct {
    const int8_t  *weights;       /* [out_features x in_features] row-major */
    const int32_t *biases;        /* [out_features] */
    int32_t        input_scale;   /* Q8.8 fixed-point scale numerator   */
    int32_t        weight_scale;  /* Q8.8 fixed-point scale numerator   */
    int32_t        output_zero;   /* output zero-point offset            */
    uint16_t       in_features;
    uint16_t       out_features;
    uint8_t        activation;    /* 0=none, 1=relu, 2=sigmoid_approx   */
} LayerDesc;

/* ── Activation codes ───────────────────────────────────────────────── */
#define ACT_NONE    0
#define ACT_RELU    1
#define ACT_SIGMOID 2

/* ── Core API ───────────────────────────────────────────────────────── */

/**
 * run_inference()
 * Full pipeline: parse model flatbuffer header → run dense layers → return score.
 *
 * @param input        INT8 input tensor (IMG_SIZE*IMG_SIZE*CHANNELS values)
 * @param input_len    Length of input array
 * @param model_data   Quantized TFLite flatbuffer bytes (from model_params.c)
 * @param model_len    Length of model_data
 * @param output_score Pointer to receive raw INT8 output (single neuron sigmoid)
 * @return InferenceStatus
 */
InferenceStatus run_inference(
    const int8_t  *input,
    size_t         input_len,
    const uint8_t *model_data,
    size_t         model_len,
    int8_t        *output_score
);

/**
 * dense_layer_int8()
 * Compute one fully-connected layer with INT8 inputs and weights.
 * Uses 32-bit accumulator to prevent overflow.
 *
 * @param input      INT8 input vector [in_features]
 * @param weights    INT8 weight matrix [out_features * in_features]
 * @param biases     INT32 bias vector [out_features]
 * @param output     INT8 output vector [out_features]
 * @param in_feat    Number of input features
 * @param out_feat   Number of output features
 * @param multiplier Requantization multiplier (Q2.30 fixed-point)
 * @param shift      Requantization right-shift amount
 * @param out_zero   Output zero-point
 * @param activation Activation function code (ACT_*)
 */
void dense_layer_int8(
    const int8_t  *input,
    const int8_t  *weights,
    const int32_t *biases,
    int8_t        *output,
    uint16_t       in_feat,
    uint16_t       out_feat,
    int32_t        multiplier,
    int32_t        shift,
    int32_t        out_zero,
    uint8_t        activation
);

/**
 * sigmoid_int8_approx()
 * Piecewise-linear INT8 approximation of sigmoid.
 * Maps INT8 input → INT8 output in range [-128, 127].
 * Accuracy within ±2 LSB of true sigmoid over [-8, 8] input range.
 */
int8_t sigmoid_int8_approx(int32_t x);

/**
 * relu_int8()
 * ReLU: clamp negative values to output_zero_point.
 */
int8_t relu_int8(int32_t x, int32_t zero_point);

#endif /* INFERENCE_ENGINE_H */
