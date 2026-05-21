#ifndef NEURON_H
#define NEURON_H

typedef struct {
    Value **w;
    Value *b;
    int nin;
    Value *(*activation)(Value *);
} Neuron;

Neuron *createNeuron(int nin);
Neuron *createNeuronTanh(int nin);
Neuron *createNeuronReLU(int nin);

void destroyNeuron(Neuron *self);

Value *callNeuron(Neuron *self, double *inputs);
Value *callNeuronWithValues(Neuron *self, Value **inputs);

void printNeuron(Neuron *self);

double randRange(double min, double max);

Value **_parameters_neuron(Neuron *self, int *param_count);
void _destroy_parameters(Value **params);

#endif