#include <stdlib.h>
#include <stdio.h>
#include "../Value.h"
#include "../Neuron.h"

/*
TODO: maybe store parameters instead of building the array everytime.
Might help with performance.
*/

// Forward declarations
double randRange(double min, double max);

Neuron *createNeuron(int nin) {
    // Creates neuron with no activation function
    Value **w = malloc(nin * sizeof(Value *)); 
    Value *b;

    for (int i = 0; i < nin; i++) {
        w[i] = createValue(randRange(-1, 1));
    }
    b = createValue(randRange(-1, 1));

    Neuron *n = malloc(sizeof(Neuron));
    *n = (Neuron){w, b, nin, NULL};

    return n;
}

Neuron *createNeuronTanh(int nin) {
    // Creates neuron with tanh activation
    Value **w = malloc(nin * sizeof(Value *)); 
    Value *b;

    for (int i = 0; i < nin; i++) {
        w[i] = createValue(randRange(-1, 1));
    }
    b = createValue(randRange(-1, 1));

    Neuron *n = malloc(sizeof(Neuron));
    *n = (Neuron){w, b, nin, tanhValue};

    return n;
}

Neuron *createNeuronReLU(int nin) {
    // Creates neuron with reLU activation
    Value **w = malloc(nin * sizeof(Value *)); 
    Value *b;

    for (int i = 0; i < nin; i++) {
        w[i] = createValue(randRange(-1, 1));
    }
    b = createValue(randRange(-1, 1));

    Neuron *n = malloc(sizeof(Neuron));
    *n = (Neuron){w, b, nin, reLUValue};

    return n;
}

void destroyNeuron(Neuron *self) {
    for (int i = 0; i < self->nin; i++) {
        destroyValue(self->w[i]);
    }
    free(self->w);
    destroyValue(self->b);
    free(self);
}

Value *callNeuron(Neuron *self, double *inputs) {
    // w * x + b
    Value *out;
    Value *dot_product = createValue(0.0);
    for (int i = 0; i < self->nin; i++) {
        Value *product = mulValueWithNum(self->w[i], inputs[i]);
        dot_product = addValues(dot_product, product);
    }
    Value *sum = addValues(dot_product, self->b);
    if (self->activation) {
        out = self->activation(sum);
    } else {
        out = sum;
    }

    return out;
}

Value *callNeuronWithValues(Neuron *self, Value **inputs) {
    // w * x + b
    Value *out;
    Value *dot_product = createValue(0.0);
    for (int i = 0; i < self->nin; i++) {
        Value *product = mulValues(self->w[i], inputs[i]);
        dot_product = addValues(dot_product, product);
    }
    Value *sum = addValues(dot_product, self->b);
    if (self->activation) {
        out = self->activation(sum);
    } else {
        out = sum;
    }

    return out;
}

void printNeuron(Neuron *self) {
    int param_count;
    Value **params = _parameters_neuron(self, &param_count);
    int i;

    if (!self->activation) {
        printf("Linear: ");
    }
    else if (self->activation == tanhValue) {
        printf("tanh: ");
    } else {
        printf("ReLU: ");
    }

    printf("weights = { ");
    for (i = 0; i < self->nin; i++) {
        printf("%f ", params[i]->data);
    }
    printf("} | bias = %f\n", self->b->data);

    _destroy_parameters(params);
}

/*
`param_count` is the pointer to where you want to
store the number of parameters.

Always destroy the returned params after use.
*/
Value **_parameters_neuron(Neuron *self, int *param_count) {
    int n = self->nin;
    Value **params = malloc(sizeof(Value *) * (n + 1));
    int i;
    for (i = 0; i < n; i++) {
        params[i] = self->w[i];
    }
    params[i] = self->b;

    *param_count = n + 1;
    return params;
}

void _destroy_parameters(Value **params) {
    free(params);
}

/*
https://stackoverflow.com/a/33059025
*/
double randRange(double min, double max) // Inclusive
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}