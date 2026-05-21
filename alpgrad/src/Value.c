#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "../Value.h"

#define N 1024

void _noop(Value *self) {} // No operation

// Forward declarations for hidden functions
void _backward_add(Value *self);
void _backward_mul(Value *self);
void _build_topo(Value *x, Value ***topo, int *topo_count, Value ***visited, int *visited_count, int *arraySize);
void _backward_tanh(Value *self);
void _backward_reLU(Value *self);
void _backward_pow(Value *self);
void _backward_exp(Value *self);

Value *createValue(double x) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){x, 0.0, _noop, {NULL}, NULL, NULL};
    return out;
}

Value *createValueWithLabel(double x, char *label) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){x, 0.0, _noop, {NULL}, NULL, label};
    return out;
}

void destroyValue(Value *self) {
    free(self);
}

Value *addValues(Value *self, Value *other) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){self->data + other->data, 0.0, _backward_add, {self, other}, "+", ""};
    return out;
}

Value *mulValues(Value *self, Value *other) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){self->data * other->data, 0.0, _backward_mul, {self, other}, "*", ""};
    return out;
}

Value *subValues(Value *self, Value *other) {
    Value *negb = negValue(other);
    Value *out = addValues(self, negb);
    return out;
}

Value *addValueWithNum(Value *self, double other) {
    Value *otherV = createValue(other);
    Value *out = addValues(self, otherV);
    return out;
}

Value *mulValueWithNum(Value *self, double other) {
    Value *otherV = createValue(other);
    Value *out = mulValues(self, otherV);
    return out;
}

Value *subValueWithNum(Value *self, double other) {
    Value *otherV = createValue(-other);
    Value *out = addValues(self, otherV);
    return out;
}

Value *negValue(Value *self) {
    Value *out = mulValueWithNum(self, -1.0);
    return out;
}

Value *divValues(Value *dividend, Value *divisor) {
    Value *out = mulValues(dividend, powValueWithNum(divisor, -1.0));
    return out;
}

Value *divValueWithNum(Value *dividend, double divisor) {
    Value *divisorV = createValue(divisor);
    Value *out = divValues(dividend, divisorV);
    return out;
}

Value *divNumWithValue(double dividend, Value *divisor) {
    Value *dividendV = createValue(dividend);
    Value *out = divValues(dividendV, divisor);
    return out;
}

Value *tanhValue(Value *self) {
    double t = tanh(self->data);
    Value *out = malloc(sizeof(Value));
    *out = (Value){t, 0.0, _backward_tanh, {self, NULL}, "tanh", ""};
    return out;
}

Value *reLUValue(Value *self) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){fmax(0, self->data), 0.0, _backward_reLU, {self, NULL}, "reLU", ""};
    return out;
}

Value *powValueWithNum(Value *self, double other) {
    char fs[32];
    snprintf(fs, sizeof(fs), "**%f", other);
    Value *out = malloc(sizeof(Value));
    *out = (Value){pow(self->data, other), 0.0, _backward_pow, {self, createValue(other)}, fs, ""};
    return out;
}

Value *expValue(Value *self) {
    Value *out = malloc(sizeof(Value));
    *out = (Value){exp(self->data), 0.0, _backward_exp, {self, NULL}, "exp", ""};
    return out;
}

void backward(Value *self) {
    int arraySize = N;
    Value **topo = calloc(arraySize, sizeof(Value *));
    Value **visited = calloc(arraySize, sizeof(Value *));
    int topo_count = 0, visited_count = 0;

    _build_topo(self, &topo, &topo_count, &visited, &visited_count, &arraySize);

    self->grad = 1.0;
    for (int i = topo_count - 1; i >= 0; i--) {
        topo[i]->_backward(topo[i]);
    }

    free(topo);
    free(visited);
}

void _backward_add(Value *self) {
    Value *a = self->_prev[0];
    Value *b = self->_prev[1];

    a->grad += 1.0 * self->grad;
    b->grad += 1.0 * self->grad;
}

void _backward_mul(Value *self) {
    Value *a = self->_prev[0];
    Value *b = self->_prev[1];

    a->grad += b->data * self->grad;
    b->grad += a->data * self->grad;
}

void _backward_pow(Value *self) {
    Value *a = self->_prev[0];
    Value *b = self->_prev[1];

    a->grad += b->data * pow(a->data, b->data - 1) * self->grad;
    b->grad += pow(a->data, b->data) * log(a->data) * self->grad;
}

void _backward_exp(Value *self) {
    Value *a = self->_prev[0];

    a->grad += self->data * self->grad;
}

void _backward_tanh(Value *self) {
    Value *a = self->_prev[0];
    a->grad += (1 - pow(self->data, 2.0)) * self->grad;
}

void _backward_reLU(Value *self) {
    Value *a = self->_prev[0];
    if (a->data <= 0) {
        a->grad += 0;
    } else {
        a->grad += 1.0 * self->grad;
    }
}

/*
Why triple pointers? 
because we need to dynamically update the size of topo
and visited and for that we need to update their pointers 
when we realloc a bigger array.
*/
void _build_topo(Value *x, Value ***topo, int *topo_count, Value ***visited, int *visited_count, int *arraySize) {
    if (x == NULL) {
        return;
    }

    // if visited, exit
    for (int i = 0; i < *visited_count; i++) {
        if ((*visited)[i] == x) {
            return;
        }
    }

    // Add to visited
    (*visited)[(*visited_count)++] = x;

    // Increase size of array if allocated length exceeded
    if (*visited_count >= *arraySize) {
        *arraySize *= 2;

        /* 
        Don't directly realloc to *topo. Instead reallocate to another
        variable so as to not cause a memory leak if the allocation fails.
        If allocation fails, the old *topo will still be there which is then
        free'd later. If we overwrite it directly like so -
        `*topo = realloc(*topo, ...);`
        Then if the allocation fails *topo will be set to NULL and we won't be able
        to free the old memory block since we lost its address. 
        */
        Value **tmpTopo = realloc(*topo, sizeof(Value *) * (*arraySize));
        Value **tmpVisited = realloc(*visited, sizeof(Value *) * (*arraySize));

        if (!tmpTopo || !tmpVisited) {
            printf("topo: reallocation failed");
        }

        *topo = tmpTopo;
        *visited = tmpVisited;
    }

    // Build for children
    for (int i = 0; i < sizeof(x->_prev)/sizeof(Value *); i++) {
        _build_topo(x->_prev[i], topo, topo_count, visited, visited_count, arraySize);
    }

    // Add to final output
    (*topo)[(*topo_count)++] = x;
}

void printValue(Value *self) {
    printf("Value{data=%.4f, grad=%.4f}\n", self->data, self->grad);
}