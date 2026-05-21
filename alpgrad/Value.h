#ifndef VALUE_H
#define VALUE_H

typedef struct Value Value;

struct Value {
    double data;
    double grad;
    void (*_backward)(Value *);
    Value *_prev[2];
    char *_op;
    char *label;
};

Value *createValue(double x);
Value *createValueWithLabel(double x, char *label);

// Destruction
void destroyValue(Value *self);

// Operations
Value *addValues(Value *self, Value *other);
Value *mulValues(Value *self, Value *other);
Value *subValues(Value *self, Value *other);

Value *addValueWithNum(Value *self, double other);
Value *mulValueWithNum(Value *self, double other);
Value *subValueWithNum(Value *self, double other);

Value *powValueWithNum(Value *self, double other);

Value *divValues(Value *dividend, Value *divisor);
Value *divValueWithNum(Value *dividend, double divisor);
Value *divNumWithValue(double dividend, Value *divisor);

Value *expValue(Value *self);

Value *negValue(Value *self);

Value *tanhValue(Value *self);

Value *reLUValue(Value *self);

void printValue(Value *self);

// Backpropagation
void backward(Value *self);

#endif