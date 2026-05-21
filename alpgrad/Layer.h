#ifndef LAYER_H
#define LAYER_H

typedef struct {
    Neuron **neurons;
    int nin; // Number of inputs
    int nout; // Number of ouputs
} Layer;

Layer *createLayer(int nin, int nout);
Layer *createLayerTanh(int nin, int nout);
Layer *createLayerReLU(int nin, int nout);

void destroyLayer(Layer *self);

Value **callLayer(Layer *self, double *inputs);
Value **callLayerWithValues(Layer *self, Value **inputs);

void printLayer(Layer *self);

Value **_parameters_layer(Layer *self, int *param_count);

#endif