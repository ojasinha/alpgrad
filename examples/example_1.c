#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include "../alpgrad/engine.h"
#include "../alpgrad/nn.h"

int main() {
    // Seed the generator
    srand(time(NULL));

    double xs[4][3] = {
        {2.0, 3.0, -1.0},
        {3.0, -1.0, 0.5},
        {0.5, 1.0, 1.0},
        {1.0, 1.0, -1.0}
    };

    double ys[] = {1.0, -1.0, -1.0, 1.0};

    int nouts[] = {4, 4, 1};

    MLP *n = createMLPTanh(3, nouts, 3);

    Value ***ypreds = calloc(4, sizeof(Value **));

    int epoch = 100;
    for (int k = 0; k < epoch; k++) {
        
        // Forward pass
        for (int i = 0; i < 4; i++) {
            ypreds[i] = callMLP(n, xs[i]);
        }

        // Calculate loss
        Value *loss = createValue(0.0);
        for (int i = 0; i < 4; i++) {
            Value *diff = subValueWithNum(ypreds[i][0], ys[i]);
            Value *squared = powValueWithNum(diff, 2.0);
            loss = addValues(loss, squared);
        }

        // Zero grad
        zero_grad(n);

        // Backward pass
        backward(loss);

        // Update params
        int params_count;
        Value **params = _parameters_mlp(n, &params_count);
        for (int i = 0; i < params_count; i++) {
            params[i]->data += -0.05 * params[i]->grad;
        }

        printf("epoch %d: loss = %f\n", k, loss->data);
        
        _destroy_parameters(params);
        destroyValue(loss);
    }

    // Final predictions are really close to ys
    printf("\nFinal predictions:\n");
    for (int i = 0; i < 4; i++) {
        printValue(ypreds[i][0]);
    }

    destroyMLP(n);
    free(ypreds);

    return 0;
}