#include <stdio.h>
#include <math.h>
#include <assert.h>
#include "../alpgrad/engine.h"
#include "../alpgrad/nn.h"

// The same tests that Karpathy has created for micrograd

void test_sanity_check() {

    // Expected outputs
    double xd = -4.0, yd = -20.0;
    double xg = 46.0, yg = 1.0;
    double eps = 1e-9;

    Value *x = createValue(-4.0);
    // z = x * 2 + 2 + x
    Value *z = addValues(addValueWithNum(mulValueWithNum(x, 2), 2), x);
    // q = relu(z) + z * x
    Value *q = addValues(reLUValue(z), mulValues(z, x));
    // h = relu(z * z)
    Value *h = reLUValue(mulValues(z, z));
    // y = h + q + q * x
    Value *y = addValues(addValues(h, q), mulValues(q, x));
    backward(y);

    assert(fabs(x->data - xd) < eps);
    assert(fabs(x->grad - xg) < eps);
    assert(fabs(y->data - yd) < eps);
    assert(fabs(y->grad - yg) < eps);

    printf("Sanity check successful!\n");
}

void test_more_ops() {

    // Expected Values
    double ad = -4.0, bd = 2.0, gd = 24.70408163265306;
    double ag = 138.83381924198252, bg = 645.5772594752186, gg = 1.0;
    double eps = 1e-9;

    Value *a = createValue(-4.0);
    Value *b = createValue(2.0);
    // c = a + b
    Value *c = addValues(a, b);
    // d = a * b + b**3
    Value *d = addValues(mulValues(a, b), powValueWithNum(b, 3));
    // c += c + 1
    c = addValues(c, addValueWithNum(c, 1));
    // c += 1 + c + (-a)
    c = addValues(c, addValues(addValueWithNum(c, 1), negValue(a)));
    // d += d * 2 + relu(b + a)
    d = addValues(d, addValues(mulValueWithNum(d, 2), reLUValue(addValues(b, a))));
    // d += 3 * d + relu(b - a)
    d = addValues(d, addValues(mulValueWithNum(d, 3), reLUValue(subValues(b, a))));
    // e = c - d
    Value *e = subValues(c, d);
    Value *f = powValueWithNum(e, 2);
    Value *g = divValueWithNum(f, 2.0);
    g = addValues(g, divNumWithValue(10.0, f));
    backward(g);

    assert(fabs(a->data - ad) < eps);
    assert(fabs(a->grad - ag) < eps);
    assert(fabs(b->data - bd) < eps);
    assert(fabs(b->grad - bg) < eps);
    assert(fabs(g->data - gd) < eps);
    assert(fabs(g->grad - gg) < eps);

    printf("More ops test successful!\n");
}

int main() {

    test_sanity_check();
    test_more_ops();

    return 0;
}