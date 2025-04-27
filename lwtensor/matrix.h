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
#include "vector.h"

/**
 * A Matrix is a specialization of the Tensor structure with rank 2.
 */
typedef struct Tensor Matrix;

/**
 * Computes the determinant of a square matrix.
 *
 * @param matrix A square matrix.
 * @return       The determinant value.
 */
ttype determinant(Matrix matrix);

/**
 * Creates a matrix with the given number of rows and columns.
 *
 * @param rows Number of rows.
 * @param cols Number of columns.
 * @return     A matrix initialized with all elements set to 0.
 */
Matrix create_matrix(unsigned int rows, unsigned int cols) {
    Matrix matrix = create_tensor(2, rows, cols);
    return matrix;
}

/**
 * Creates an identity matrix of size n x n.
 *
 * @param n Size of the identity matrix.
 * @return  An identity matrix.
 */
Matrix create_indentity(unsigned int n) {

    Matrix matrix = create_matrix(n, n);

    for(int c = 0; c < n; c++) {
        for(int r = 0; r < n; r ++) {
            if(c == r)
                set_value(matrix, 1.0, c, r);
        }
    }

    return matrix;
}

/**
 * Performs matrix multiplication between two matrices.
 *
 * @param lhs Left-hand side matrix.
 * @param rhs Right-hand side matrix.
 * @return    A new matrix resulting from lhs * rhs.
 */
Matrix matmul(Matrix lhs, Matrix rhs) {

    Matrix result = create_matrix(rhs.shape[1], lhs.shape[0]);

    for(int r = 0; r < lhs.shape[0]; r ++) {
        for(int c = 0; c < rhs.shape[1]; c ++) {

            ttype mir = 0.0;
            for(int k = 0; k < lhs.shape[1]; k ++)
                mir += get_value(rhs, k, r) * get_value(lhs, c, k);

            set_value(result, mir, r, c);
        }
    }

    return result;
}

/**
 * Applies a matrix transformation to a vector.
 *
 * @param vec    The vector to be transformed.
 * @param matrix The transformation matrix.
 * @return       The resulting transformed vector.
 */
Vector transform(Vector vec, Matrix matrix) {

    Vector vector = create_vector(vec.shape[0]);

    for(int r = 0; r < matrix.shape[0]; r ++) {

        Vector row_vector = create_vector(matrix.shape[1]);

        for(int c = 0; c < matrix.shape[1]; c ++) {
            ttype value = get_value(matrix, r, c);
            set_value(row_vector, value, c);
        }
        
        ttype value = dot(row_vector, vec);
        vector.components[r] = value;

        destroy_tensor(row_vector);
    }

    return vector;
}

/**
 * Returns the transpose of a matrix.
 *
 * @param matrix Input matrix.
 * @return       Transposed matrix.
 */
Matrix transpose(Matrix matrix) {

    Matrix matrix_transposed = create_matrix(matrix.shape[1], matrix.shape[0]);

    for(int r = 0; r < matrix.shape[0]; r ++) {
        for(int c = 0; c < matrix.shape[1]; c ++) {
            ttype value = get_value(matrix, r, c);
            set_value(matrix_transposed, value, c, r);
        }
    }

    return matrix_transposed;
}

/**
 * Computes the minor of a matrix by excluding a specified row and column.
 *
 * @param matrix Input matrix.
 * @param row    Row to exclude.
 * @param col    Column to exclude.
 * @return       Determinant of the sub-matrix.
 */
ttype minor(Matrix matrix, unsigned int row, unsigned int col) {

    Matrix sub_matrix = create_matrix(matrix.shape[1] - 1, matrix.shape[1] - 1);

    int index = 0;
    for(int r = 0; r < matrix.shape[0]; r ++) {
        for(int c = 0; c < matrix.shape[1]; c ++) {

            if(row != r && col != c) {
                ttype value = get_value(matrix, r, c);
                sub_matrix.components[index] = value;
                index ++;
            }
        }
    }

    ttype det = determinant(sub_matrix);
    destroy_tensor(sub_matrix);

    return det;
}

/**
 * Computes the cofactor of an element at a specific position in a matrix.
 *
 * @param matrix Input matrix.
 * @param row    Row index.
 * @param col    Column index.
 * @return       The cofactor value.
 */
ttype cofactor(Matrix matrix, unsigned int row, unsigned int col) {

    int sign = pow(-1.0, row + col);
    ttype result = sign * minor(matrix, row, col);

    return result;
}

/**
 * Computes the cofactor matrix of a given matrix.
 *
 * @param matrix Input matrix.
 * @return       The cofactor matrix.
 */
Matrix cofactor_matrix(Matrix matrix) {

    Matrix cof_matrix = create_matrix(matrix.shape[0], matrix.shape[1]);

    for(int r = 0; r < matrix.shape[0]; r ++) {
        for(int c = 0; c < matrix.shape[1]; c ++)
            set_value(cof_matrix, cofactor(matrix, r, c), r, c);
    }

    return cof_matrix;
}

/**
 * Computes the adjugate (adjoint) of a matrix.
 *
 * @param matrix Input matrix.
 * @return       The adjugate matrix.
 */
Matrix adjugate_matrix(Matrix matrix) {

    Matrix cof_matrix = cofactor_matrix(matrix);
    Matrix cof_matrix_transposed = transpose(cof_matrix);

    return cof_matrix_transposed;
}

/**
 * Computes the determinant of a matrix.
 *
 * @param matrix Input matrix.
 * @return       The determinant value.
 *
 * Note: Only works for square matrices.
 */
ttype determinant(Matrix matrix) {

    if(matrix.shape[0] != matrix.shape[1])
        return 0.0;

    if(matrix.shape[1] == 2)
        return get_value(matrix, 0, 0) * get_value(matrix, 1, 1) - get_value(matrix, 1, 0) * get_value(matrix, 0, 1);

    ttype result = 0.0;
    for(int r = 0; r < matrix.shape[0]; r ++) {
        ttype cof = cofactor(matrix, r, 0);
        result += get_value(matrix, r, 0) * cof; 
    }

    return result;
}

/**
 * Computes the inverse of a matrix.
 *
 * @param matrix A square matrix.
 * @return       The inverse matrix.
 *
 * Note: Assumes the matrix is invertible. No zero-determinant check is enforced.
 */
Matrix inverse(Matrix matrix) {

    ttype det = determinant(matrix);

    Matrix inv = adjugate_matrix(matrix);
    ttype inv_determinant = 1 / det;
    product_scalar(inv, inv_determinant);

    return inv;
}