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

#include "tensor.h"

/**
 * A Vector is a specialization of the Tensor structure with rank 1.
 */
typedef struct Tensor Vector;

/**
 * Creates a vector of size n.
 *
 * @param n Number of elements in the vector.
 * @return  A vector initialized with all components set to 0.
 */
Vector create_vector(int n) {
    Vector vector = create_tensor(1, n);
    return vector;
}

/**
 * Creates a 3D vector from an array of three components.
 *
 * @param vec An array of 3 ttype values.
 * @return    A 3D vector initialized with the given values.
 */
Vector create_vector_from(ttype vec[3]) {

    Vector vector = create_tensor(1, 3);

    vector.components[0] = vec[0];
    vector.components[1] = vec[1];
    vector.components[2] = vec[2];
    
    return vector;
}

/**
 * Computes the Euclidean norm (magnitude) of a vector.
 *
 * @param vec Input vector.
 * @return    The magnitude of the vector.
 */
ttype norm(Vector vec) {

    ttype sum = 0.0;
    for(int i = 0; i < vec.shape[0]; i ++)
        sum += vec.components[i] * vec.components[i];

    return sqrt(sum);
}

/**
 * Normalizes the input vector (returns a unit vector in the same direction).
 *
 * @param vec Input vector.
 * @return    A normalized vector.
 */
Vector normalize(Vector vec) {

    ttype modulo = norm(vec);
    Vector vector = create_copy(vec);

    for(int i = 0; i < vector.shape[0]; i ++)
        vector.components[i] /= modulo;

    return vector;
}

/**
 * Computes the cross product of two 3D vectors.
 *
 * @param u First vector.
 * @param v Second vector.
 * @return  A vector representing the cross product u × v.
 *
 * Note: Assumes both vectors are 3D.
 */
Vector cross(Vector u, Vector v) {

    Vector vector = create_vector(u.shape[0]);

    vector.components[0] = u.components[1] * v.components[2] - u.components[2] * v.components[1];
    vector.components[1] = u.components[2] * v.components[0] - u.components[0] * v.components[2];
    vector.components[2] = u.components[0] * v.components[1] - u.components[1] * v.components[0];

    return vector;
}