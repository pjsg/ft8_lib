// Define a function pointer type for your expensive frequency evaluation
typedef double (*cost_func_t)(double freq, void* user_data);

extern double brent_minimize(double ax, double bx, double cx, cost_func_t f, void* user_data, double tol, double *xmin);
