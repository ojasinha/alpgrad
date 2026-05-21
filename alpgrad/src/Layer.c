#include <stdlib.h>
#include <stdio.h>
#include "../Value.h"
#include "../Neuron.h"
#include "../Layer.h"

Layer *createLayer(int nin, int nout) {
    Neuron **neurons = malloc(nout * sizeof(Neuron *));

    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuron(nin);
    }

    Layer *l = malloc(sizeof(Layer));
    *l = (Layer){neurons, nin, nout};

    return l;
}

Layer *createLayerTanh(int nin, int nout) {
    Neuron **neurons = malloc(nout * sizeof(Neuron *));

    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuronTanh(nin);
    }

    Layer *l = malloc(sizeof(Layer));
    *l = (Layer){neurons, nin, nout};

    return l;
}

Layer *createLayerReLU(int nin, int nout) {
    Neuron **neurons = malloc(nout * sizeof(Neuron *));

    for (int i = 0; i < nout; i++) {
        neurons[i] = createNeuronReLU(nin);
    }

    Layer *l = malloc(sizeof(Layer));
    *l = (Layer){neurons, nin, nout};

    return l;
}

void destroyLayer(Layer *self) {
    for (int i = 0; i < self->nout; i++) {
        destroyNeuron(self->neurons[i]);
    }
    free(self->neurons);
    free(self);
}

Value **callLayer(Layer *self, double *inputs) {
    Value **outs = malloc(self->nout * sizeof(Value *));

    for (int i = 0; i < self->nout; i++) {
        outs[i] = callNeuron(self->neurons[i], inputs);
    }
    
    return outs;
}

Value **callLayerWithValues(Layer *self, Value **inputs) {
    Value **outs = malloc(self->nout * sizeof(Value *));

    for (int i = 0; i < self->nout; i++) {
        outs[i] = callNeuronWithValues(self->neurons[i], inputs);
    }

    return outs;
}

void printLayer(Layer *self) {
    for (int i = 0; i < self->nout; i++) {
        printf("Neuron %d: ", i);
        printNeuron(self->neurons[i]);;
    }
}

/*
`param_count` is the pointer to where you want to
store the number of parameters.

Always destroy the returned params after use.
*/
Value **_parameters_layer(Layer *self, int *param_count) {
    int in = self->nin;
    int out = self->nout;
    int num_params;
    Value **params = malloc((in + 1) * out * sizeof(Value *));
    int k = 0;

    for (int i = 0; i < out; i++) {
        Value **neuron_params = _parameters_neuron(self->neurons[i], &num_params);
        for (int j = 0; j < in + 1; j++) {
            params[k++] = neuron_params[j];
        }
        _destroy_parameters(neuron_params);
    }

    *param_count = (in + 1) * out;
    return params;
}