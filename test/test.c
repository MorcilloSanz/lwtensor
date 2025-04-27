#include <stdio.h>

#include "../lwtensor/matrix.h"

int main() {

    Matrix matrix = create_indentity(3);
    set_value(matrix, 2.0, 0, 1);
    set_value(matrix, 3.0, 1, 2);

    for(int row = 0; row < matrix.shape[0]; row ++) {
        for(int col = 0; col < matrix.shape[1]; col ++) {
            printf("%f ", get_value(matrix, row, col));
        }
        printf("\n");
    }

    Matrix inv = inverse(matrix);
    Matrix matrix2 = inverse(inv);

    printf("\n");

    for(int row = 0; row < matrix2.shape[0]; row ++) {
        for(int col = 0; col < matrix2.shape[1]; col ++) {
            printf("%f ", get_value(matrix2, row, col));
        }
        printf("\n");
    }

    destroy_tensor(matrix);
    destroy_tensor(matrix2);
    destroy_tensor(inv);

    return 0;
}