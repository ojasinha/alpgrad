#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "../alpgrad/engine.h"
#include "../alpgrad/nn.h"

/*
This example is just @Jaykef on Github's port of Karpathy's demo notebook
for his implementation of micrograd in C:
https://github.com/Jaykef/micrograd.c/blob/main/train.c

It's just been modified to fit alpgrad :)
*/

#define N_SAMPLES 100
#define HIDDEN_SIZE 16
#define OUTPUT_SIZE 1

void load_data(const char* filename, double X[N_SAMPLES][2], int y[N_SAMPLES]) {
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        printf("Error opening file: %s\n", filename);
        exit(1);
    }

    char line[1024];
    fgets(line, sizeof(line), file);  // Skip header

    for (int i = 0; i < N_SAMPLES; i++) {
        if (fscanf(file, "%lf,%lf,%d", &X[i][0], &X[i][1], &y[i]) != 3) {
            printf("Error reading line %d\n", i + 1);
            exit(1);
        }
    }

    fclose(file);
}

double loss(MLP* model, double X[N_SAMPLES][2], int y[N_SAMPLES], double *accuracy) {
    double total_loss = 0.0;
    int correct = 0;

    zero_grad(model);

    for (int i = 0; i < N_SAMPLES; i++) {
        // Forward pass
        Value *inputs[2];
        for (int j = 0; j < 2; j++) {
            inputs[j] = createValue(X[i][j]);
        }
        Value **output = callMLPWithValues(model, inputs);

        // Compute margin loss
        Value *target = createValue(y[i] * 2.0 - 1.0);
        Value *margin_loss = reLUValue(addValueWithNum(negValue(mulValues(target, output[0])), 1.0));
        total_loss += margin_loss->data;

        // Accumulate gradients
        backward(margin_loss);

        // Compute accuracy
        if ((y[i] == 1 && output[0]->data > 0) || (y[i] == 0 && output[0]->data <= 0)) {
            correct++;
        }

        // Free memory
        destroyValue(output[0]);
        free(output);
        destroyValue(target);
        destroyValue(margin_loss);
    }

    // Compute regularization loss and add to total loss
    double reg_loss = 0.0;
    double alpha = 1e-4;
    int param_count;
    Value **params = _parameters_mlp(model, &param_count);
    for (int i = 0; i < param_count; i++) {
        reg_loss += params[i]->data * params[i]->data;
    }
    reg_loss *= alpha;
    total_loss += reg_loss;

    // Average loss over samples
    total_loss /= N_SAMPLES;

    // Compute accuracy
    *accuracy = (double)correct / N_SAMPLES * 100.0;

    // Free parameter array
    _destroy_parameters(params);

    return total_loss;
}

int main() {
    srand(time(NULL));

    // load data
    double X[N_SAMPLES][2];
    int y[N_SAMPLES];
    load_data("data/make_moons.csv", X, y);

    // Initialize model
    int layer_sizes[] = {HIDDEN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE};
    MLP *model = createMLPReLU(2, layer_sizes, 3);

    // Training loop
    for (int epoch = 0; epoch < 100; epoch++) {
        double accuracy = 0.0;
        double avg_loss = loss(model, X, y, &accuracy);

        // Update weights
        double learning_rate = 0.01;
        int param_count;
        Value **params = _parameters_mlp(model, &param_count);
        for (int i = 0; i < param_count; i++) {
            params[i]->data -= learning_rate * params[i]->grad;
        }
        _destroy_parameters(params);

        // Print progress
        printf("step %d loss %f, accuracy %f%%\n", epoch, avg_loss, accuracy);
    }

    // destroy model
    destroyMLP(model);

    return 0;
}