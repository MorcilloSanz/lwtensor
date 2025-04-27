/*
  MIT License
  
  Copyright (c) 2025 Morcillo Sanz
  
  Permission is hereby granted, free of charge, to any person obtaining a copy
  of this software and associated documentation files (the "Software"), to deal
  in the Software without restriction, including without limitation the rights
  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
  copies of the Software, and to permit persons to whom the Software is
  furnished to do so, subject to the following conditions:
  
  The above copyright notice and this permission notice shall be included in all
  copies or substantial portions of the Software.
  
  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
  SOFTWARE.
*/

#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <stdarg.h>

#ifndef ttype
#define ttype double
#endif

struct Tensor {
    int* shape;
    ttype* components;
    unsigned int rank;
};

typedef struct Tensor Tensor;

/**
 * Creates a tensor of a given rank and shape.
 *
 * @param rank The number of dimensions (axes) of the tensor.
 * @param ...  A variable number of integers specifying the size of each dimension.
 * @return     A Tensor structure with allocated memory and components initialized to 0.0.
 */
Tensor create_tensor(unsigned int rank, ...) {

    Tensor tensor;

    va_list args;
    va_start(args, rank);

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * rank);

    for(int i = 0; i < rank; i ++) {
        int s = va_arg(args, int);
        length *= s;
        shape[i] = s;
    }

    va_end(args);

    tensor.rank = rank;
    tensor.shape = shape;
    tensor.components = (ttype*) malloc(sizeof(ttype) * length);

    for(size_t i = 0; i < length; i ++) 
        tensor.components[i] = 0.0;

    return tensor;
}

/**
 * Creates a tensor using a pointer to the shape array.
 *
 * @param rank  The number of dimensions of the tensor.
 * @param shape A pointer to an array of integers defining the size of each dimension.
 * @return      A Tensor structure with allocated components initialized to 0.0.
 *
 * Note: The `shape` pointer is not copied; it is assigned directly. Be cautious with ownership and lifetime.
 */
Tensor create_tensor_byptr(unsigned rank, int* shape) {

    Tensor tensor;

    size_t length = 1;
    for(int i = 0; i < rank; i ++) length *= shape[i];

    tensor.rank = rank;
    tensor.shape = shape;
    tensor.components = (ttype*) malloc(sizeof(ttype) * length);

    for(size_t i = 0; i < length; i ++) 
        tensor.components[i] = 0.0;

    return tensor;
}

/**
 * Creates a deep copy of a given tensor.
 *
 * @param tensor The source Tensor to be copied.
 * @return       A new Tensor structure with its own allocated shape and component arrays.
 */
Tensor create_copy(Tensor tensor) {

    Tensor tensor_copy;
    tensor_copy.rank = tensor.rank;

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * tensor.rank);

    for(int i = 0; i < tensor.rank; i ++) {
        shape[i] = tensor.shape[i];
        length *= tensor.shape[i];
    }

    tensor_copy.shape = shape;
    tensor_copy.components = (ttype*) malloc(sizeof(ttype) * length);

    for(int i = 0; i < length; i ++)
        tensor_copy.components[i] = tensor.components[i];

    return tensor_copy;
}

/**
 * Sets the value of a tensor element at a specified multi-dimensional index.
 *
 * @param tensor The tensor to be modified.
 * @param value  The value to assign to the specified position.
 * @param ...    A sequence of integers indicating the index in each dimension.
 *
 * Note: This function assumes row-major ordering and does not perform bounds checking.
 */
void set_value(Tensor tensor, ttype value, ...) {

    va_list args;
    va_start(args, value);

    int index = 0;
    for(int i = 0; i < tensor.rank; i ++) {

        int subIndex = va_arg(args, int);
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor.shape[j];
        }

        index += subIndex;
    }

    tensor.components[index] = value;
    va_end(args);
}

/**
 * Retrieves the value of a tensor element at a specified multi-dimensional index.
 *
 * @param tensor The tensor to read from.
 * @param ...    A sequence of integers indicating the index in each dimension.
 * @return       The value at the specified position.
 *
 * Note: This function assumes row-major ordering and does not perform bounds checking.
 */
ttype get_value(Tensor tensor, ...) {

    va_list args;
    va_start(args, tensor);

    int index = 0;
    for(int i = 0; i < tensor.rank; i ++) {

        int subIndex = va_arg(args, int);
        for(int j = 0; j < i; j ++) {
            subIndex *= tensor.shape[j];
        }

        index += subIndex;
    }

    ttype value = tensor.components[index];
    va_end(args);

    return value;
}

/**
 * Calculates the total number of elements in a tensor.
 *
 * @param tensor The tensor whose length is to be computed.
 * @return       The product of all dimensions (i.e., the number of components).
 */
size_t get_length(Tensor tensor) {

    size_t length = 1;
    for(int i = 0; i < tensor.rank; i ++) 
        length *= tensor.shape[i];

    return length;
}

/**
 * Adds two tensors element-wise.
 *
 * @param lhs The first operand tensor.
 * @param rhs The second operand tensor.
 * @return    A new tensor containing the element-wise sum of `lhs` and `rhs`.
 *
 * Note: Both tensors must have the same shape. No shape checking is performed.
 */
Tensor sum(Tensor lhs, Tensor rhs) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] + rhs.components[i];

    return tensor;
}

/**
 * Adds a scalar to each element of a tensor.
 *
 * @param lhs    The input tensor.
 * @param scalar The scalar value to add.
 * @return       A new tensor where each element is `lhs[i] + scalar`.
 */
Tensor sum_scalar(Tensor lhs, ttype scalar) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] + scalar;

    return tensor;
}

/**
 * Subtracts one tensor from another element-wise.
 *
 * @param lhs The minuend tensor.
 * @param rhs The subtrahend tensor.
 * @return    A new tensor containing the result of `lhs[i] - rhs[i]` for each element.
 *
 * Note: Both tensors must have the same shape. No shape checking is performed.
 */
Tensor subtract(Tensor lhs, Tensor rhs) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] - rhs.components[i];

    return tensor;
}

/**
 * Subtracts a scalar from each element of a tensor.
 *
 * @param lhs    The input tensor.
 * @param scalar The scalar value to subtract.
 * @return       A new tensor where each element is `lhs[i] - scalar`.
 */
Tensor subtract_scalar(Tensor lhs, ttype scalar) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] - scalar;

    return tensor;
}

/**
 * Divides two tensors element-wise.
 *
 * @param lhs The numerator tensor.
 * @param rhs The denominator tensor.
 * @return    A new tensor where each element is `lhs[i] / rhs[i]`.
 *
 * Note: Both tensors must have the same shape. No shape checking or division-by-zero handling is performed.
 */
Tensor divide(Tensor lhs, Tensor rhs) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] / rhs.components[i];

    return tensor;
}

/**
 * Divides each element of a tensor by a scalar.
 *
 * @param lhs    The input tensor.
 * @param scalar The scalar divisor.
 * @return       A new tensor where each element is `lhs[i] / scalar`.
 *
 * Note: No division-by-zero check is performed.
 */
Tensor divide_scalar(Tensor lhs, ttype scalar) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] / scalar;

    return tensor;
}

/**
 * Performs the Hadamard (element-wise) product of two tensors.
 *
 * @param lhs The first operand tensor.
 * @param rhs The second operand tensor.
 * @return    A new tensor containing the result of `lhs[i] * rhs[i]` for each element.
 *
 * Note: Both tensors must have the same shape. No shape checking is performed.
 */
Tensor hadamard(Tensor lhs, Tensor rhs) {
    
    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] * rhs.components[i];

    return tensor;
}

/**
 * Computes the dot product of two tensors (viewed as flat vectors).
 *
 * @param lhs The first operand tensor.
 * @param rhs The second operand tensor.
 * @return    The sum of element-wise products of `lhs` and `rhs`.
 *
 * Note: This treats both tensors as flat arrays. Shapes must match.
 */
ttype dot(Tensor lhs, Tensor rhs) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    ttype sum = 0.0;
    for(int i = 0; i < length; i ++)
        sum += lhs.components[i] * rhs.components[i];

    return sum;
}

/**
 * Multiplies each element of a tensor by a scalar.
 *
 * @param lhs    The input tensor.
 * @param scalar The scalar value to multiply.
 * @return       A new tensor with each element equal to `lhs[i] * scalar`.
 */
Tensor product_scalar(Tensor lhs, ttype scalar) {

    size_t length = 1;
    int* shape = (int*) malloc(sizeof(int) * lhs.rank);

    for(int i = 0; i < lhs.rank; i ++) {
        shape[i] = lhs.shape[i];
        length *= lhs.shape[i];
    }

    Tensor tensor = create_tensor_byptr(lhs.rank, shape);

    for(int i = 0; i < length; i ++)
        tensor.components[i] = lhs.components[i] * scalar;

    return tensor;
}

/**
 * Frees the memory allocated for a tensor's shape and components.
 *
 * @param tensor The tensor to destroy.
 *
 * Note: Only use this on tensors created via `create_tensor`, `create_tensor_byptr`, or similar functions.
 */
void destroy_tensor(Tensor tensor) {
    free(tensor.shape);
    free(tensor.components);
}