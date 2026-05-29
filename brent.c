
#include <math.h>

#include "brent.h"


// A clean, standalone implementation of Brent's Method for Optimization
double brent_minimize(double ax, double bx, double cx, cost_func_t f, void* user_data, double tol, double *xmin) {
    double a, b, d, etemp, fu, fv, fw, fx, p, q, r, tol1, tol2, u, v, w, x, xm;
    double e = 0.0; // Distance moved on the step before last

    double initial_mag;
    double initial_x;
    
    // Golden ratio constants
    const double CGOLD = 0.3819660;
    const double ZEPS  = 1.0e-10; 

    // Ensure brackets are in order
    a = (ax < cx) ? ax : cx;
    b = (ax > cx) ? ax : cx;
    initial_x = x = w = v = bx;
    
    // Evaluate the function at the initial guess
    // user_data lets you pass state (like your raw signals or time offset) without globals
    initial_mag = fw = fv = fx = f(x, user_data); 

    for (int iter = 1; iter <= 100; iter++) { // 100 iterations max safety limit
        xm = 0.5 * (a + b);
        tol1 = tol * fabs(x) + ZEPS;
        tol2 = 2.0 * tol1;

        // Check if stopping criterion is met
        if (fabs(x - xm) <= (tol2 - 0.5 * (b - a))) {
            *xmin = x;
            return fx;
        }

        if (fabs(e) > tol1) {
            // Construct a trial parabolic fit
            r = (x - w) * (fx - fv);
            q = (x - v) * (fx - fw);
            p = (x - v) * q - (x - w) * r;
            q = 2.0 * (q - r);
            if (q > 0.0) p = -p;
            q = fabs(q);
            etemp = e;
            e = d;

            // Is the parabolic step acceptable?
            if (fabs(p) >= fabs(0.5 * q * etemp) || p <= q * (a - x) || p >= q * (b - x)) {
                // No, take a golden section step instead
                e = (x >= xm ? a - x : b - x);
                d = CGOLD * e;
            } else {
                // Yes, take the parabolic step
                d = p / q;
                u = x + d;
                if (u - a < tol2 || b - u < tol2) {
                    d = (xm - x >= 0.0) ? fabs(tol1) : -fabs(tol1);
                }
            }
        } else {
            // Take a golden section step
            e = (x >= xm ? a - x : b - x);
            d = CGOLD * e;
        }

        // Evaluate the new point
        u = (fabs(d) >= tol1 ? x + d : x + ((d >= 0.0) ? fabs(tol1) : -fabs(tol1)));
        fu = f(u, user_data);

        // Update the brackets and best points
        if (fu <= fx) {
            if (u >= x) a = x; else b = x;
            v = w; fv = fw;
            w = x; fw = fx;
            x = u; fx = fu;
        } else {
            if (u < x) a = u; else b = u;
            if (fu <= fw || w == x) {
                v = w; fv = fw;
                w = u; fw = fu;
            } else if (fu <= fv || v == x || v == w) {
                v = u; fv = fu;
            }
        }
    }
    *xmin = x;
    return fx;
}