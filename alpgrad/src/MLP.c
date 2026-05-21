#include <stdlib.h>
#include <stdio.h>
#include "../Value.h"
#include "../Neuron.h"
#include "../Layer.h"
#include "../MLP.h"

// Forward declaration
void zero_grad(MLP *self);

MLP *createMLP(int nin, int *nouts, int nlayer) {
    Layer **layers = malloc(nlayer * sizeof(Layer *));

    /*
    Instead of combining nin and nouts and then creating
    the layers in one loop I'm creating the first layer
    separately and then just looping over nouts to create
    the hidden and output layers.
    */
    layers[0] = createLayer(nin, nouts[0]);
    for (int i = 0; i < (nlayer - 1); i++) {
        layers[i + 1] = createLayer(nouts[i], nouts[i+1]);
    }

    MLP *m = malloc(sizeof(MLP));
    *m = (MLP){layers, nin, nouts, nlayer};

    return m;
}

MLP *createMLPTanh(int nin, int *nouts, int nlayer) {
    Layer **layers = malloc(nlayer * sizeof(Layer *));

    layers[0] = createLayerTanh(nin, nouts[0]);
    for (int i = 0; i < (nlayer - 1); i++) {
        // Make the last layer linear
        if (i + 1 == nlayer - 1) {
            layers[i + 1] = createLayer(nouts[i], nouts[i+1]);
            break;
        }
        // Make the others non-linear
        layers[i + 1] = createLayerTanh(nouts[i], nouts[i+1]);
    }

    MLP *m = malloc(sizeof(MLP));
    *m = (MLP){layers, nin, nouts, nlayer};

    return m;
}

MLP *createMLPReLU(int nin, int *nouts, int nlayer) {
    Layer **layers = malloc(nlayer * sizeof(Layer *));

    layers[0] = createLayerReLU(nin, nouts[0]);
    for (int i = 0; i < (nlayer - 1); i++) {
        // Make the last layer linear
        if (i + 1 == nlayer - 1) {
            layers[i + 1] = createLayer(nouts[i], nouts[i+1]);
            break;
        }
        // Make the others non-linear
        layers[i + 1] = createLayerReLU(nouts[i], nouts[i+1]);
    }

    MLP *m = malloc(sizeof(MLP));
    *m = (MLP){layers, nin, nouts, nlayer};

    return m;
}

void destroyMLP(MLP *self) {
    for (int i = 0; i < self->nlayer; i++) {
        destroyLayer(self->layers[i]);
    }
    free(self->layers);
    free(self);
}

Value **callMLP(MLP *self, double *x) {
    int n = self->nlayer;

    // Total number of outputs is number of outputs in the final layer
    Value **outs, **tempouts;

    outs = callLayer(self->layers[0], x);
    for (int i = 1; i < n; i++) {
        tempouts =  callLayerWithValues(self->layers[i], outs);
        free(outs);
        outs = tempouts;
    }

    return outs;
}

Value **callMLPWithValues(MLP *self, Value **x) {
    int n = self->nlayer;

    Value **outs, **tempouts;

    outs = callLayerWithValues(self->layers[0], x);
    for (int i = 1; i < n; i++) {
        tempouts = callLayerWithValues(self->layers[i], outs);
        free(outs);
        outs = tempouts;
    }

    return outs;
}

void printMLP(MLP *self) {
    int n = self->nlayer;
    for (int i = 0; i < n; i++) {
        printf("----- Layer %d -----\n", i);
        printLayer(self->layers[i]);
    }
    printf("\n");
}

/*
`param_count` is the pointer to where you want to
store the number of parameters.

Always destroy the returned params after use.
*/
Value **_parameters_mlp(MLP *self, int *param_count) {
    int in = self->nin;
    int *outs = self->nouts;
    int n = self->nlayer;

    int num_params = (in + 1) * outs[0];
    for (int i = 0; i < (n - 1); i++) {
        num_params += (outs[i] + 1) * outs[i + 1];
    }

    int np; // Param count keeper for layers
    Value **params = malloc(num_params * sizeof(Value *));
    int k = 0;

    for (int i = 0; i < n; i++) {
        Value **layer_params = _parameters_layer(self->layers[i], &np);
        for (int j = 0; j < np; j++) {
            params[k++] = layer_params[j];
        }
        _destroy_parameters(layer_params);
    }

    *param_count = num_params;
    return params;
}

void zero_grad(MLP *self) {
    int params_count;
    Value **params = _parameters_mlp(self, &params_count);
    for (int i = 0; i < params_count; i++) {
        params[i]->grad = 0.0;
    }
    _destroy_parameters(params);
}