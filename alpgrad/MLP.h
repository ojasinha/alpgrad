#ifndef MLP_H
#define MLP_H

typedef struct {
    Layer **layers;
    int nin; // Number of inputs
    int *nouts; // Number of outputs in each layer
    int nlayer; // Number of layers
} MLP;

MLP *createMLP(int nin, int *nouts, int nlayer);
MLP *createMLPTanh(int nin, int *nouts, int nlayer);
MLP *createMLPReLU(int nin, int *nouts, int nlayer);

void destroyMLP(MLP *self);

Value **callMLP(MLP *self, double *x);
Value **callMLPWithValues(MLP *self, Value **x);

void printMLP(MLP *self);

void zero_grad(MLP *self);

Value **_parameters_mlp(MLP* self, int *param_count);
void _destroy_parameters(Value **params);

#endif