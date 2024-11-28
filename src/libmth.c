// libmth.c

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Complex number struct
typedef struct {
    float real;
    float imag;
} Complex;

// Vector struct
typedef struct {
    float x;
    float y;
    float z;
} Vector3;

// Matrix struct
typedef struct {
    float m[3][3];
} Matrix3x3;

// Polynomial struct
typedef struct {
    float* coefficients;
    int degree;
} Polynomial;

// Basic math functions
__declspec(dllexport) int add(int a, int b) {
    return a + b;
}

__declspec(dllexport) int subtract(int a, int b) {
    return a - b;
}

__declspec(dllexport) int multiply(int a, int b) {
    return a * b;
}

__declspec(dllexport) float divide(int a, int b) {
    if (b == 0) {
        printf("Error: Division by zero!\n");
        return 0.0f;
    }
    return (float)a / (float)b;
}

// Complex math functions
__declspec(dllexport) Complex add_complex(Complex a, Complex b) {
    Complex result = {a.real + b.real, a.imag + b.imag};
    return result;
}

__declspec(dllexport) Complex subtract_complex(Complex a, Complex b) {
    Complex result = {a.real - b.real, a.imag - b.imag};
    return result;
}

__declspec(dllexport) Complex multiply_complex(Complex a, Complex b) {
    Complex result;
    result.real = a.real * b.real - a.imag * b.imag;
    result.imag = a.real * b.imag + a.imag * b.real;
    return result;
}

__declspec(dllexport) Complex divide_complex(Complex a, Complex b) {
    Complex result;
    float denominator = b.real * b.real + b.imag * b.imag;  // Denominator: |b|^2
    if (denominator == 0) {
        printf("Error: Division by zero!\n");
        result.real = 0.0f;
        result.imag = 0.0f;
        return result;  // Return a complex number with zero
    }
    result.real = (a.real * b.real + a.imag * b.imag) / denominator;
    result.imag = (a.imag * b.real - a.real * b.imag) / denominator;
    return result;
}

// Algebraic functions
__declspec(dllexport) float quadratic_root1(float a, float b, float c) {
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        printf("Error: No real roots!\n");
        return 0.0f;
    }
    return (-b + sqrt(discriminant)) / (2 * a);
}

__declspec(dllexport) float quadratic_root2(float a, float b, float c) {
    float discriminant = b * b - 4 * a * c;
    if (discriminant < 0) {
        printf("Error: No real roots!\n");
        return 0.0f;
    }
    return (-b - sqrt(discriminant)) / (2 * a);
}

// Vector operations
__declspec(dllexport) Vector3 add_vector(Vector3 a, Vector3 b) {
    Vector3 result = {a.x + b.x, a.y + b.y, a.z + b.z};
    return result;
}

__declspec(dllexport) Vector3 subtract_vector(Vector3 a, Vector3 b) {
    Vector3 result = {a.x - b.x, a.y - b.y, a.z - b.z};
    return result;
}

__declspec(dllexport) Vector3 cross_product(Vector3 a, Vector3 b) {
    Vector3 result;
    result.x = a.y * b.z - a.z * b.y;
    result.y = a.z * b.x - a.x * b.z;
    result.z = a.x * b.y - a.y * b.x;
    return result;
}

__declspec(dllexport) float dot_product(Vector3 a, Vector3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__declspec(dllexport) float magnitude(Vector3 v) {
    return sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

__declspec(dllexport) Vector3 normalize(Vector3 v) {
    float mag = magnitude(v);
    Vector3 result = {v.x / mag, v.y / mag, v.z / mag};
    return result;
}

__declspec(dllexport) float angle_between_vectors(Vector3 a, Vector3 b) {
    float dot = dot_product(a, b);
    float mag_a = magnitude(a);
    float mag_b = magnitude(b);
    return acos(dot / (mag_a * mag_b));  // Angle in radians
}

// Matrix operations
__declspec(dllexport) Matrix3x3 multiply_matrix(Matrix3x3 a, Matrix3x3 b) {
    Matrix3x3 result = {{0}};
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                result.m[i][j] += a.m[i][k] * b.m[k][j];
            }
        }
    }
    return result;
}

__declspec(dllexport) Matrix3x3 transpose_matrix(Matrix3x3 m) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = m.m[j][i];
        }
    }
    return result;
}

__declspec(dllexport) Matrix3x3 scalar_multiply_matrix(float scalar, Matrix3x3 m) {
    Matrix3x3 result;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result.m[i][j] = scalar * m.m[i][j];
        }
    }
    return result;
}

__declspec(dllexport) float determinant(Matrix3x3 m) {
    return m.m[0][0] * (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1])
         - m.m[0][1] * (m.m[1][0] * m.m[2][2] - m.m[1][2] * m.m[2][0])
         + m.m[0][2] * (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]);
}

// Polynomial functions
__declspec(dllexport) float evaluate_polynomial(Polynomial p, float x) {
    float result = 0;
    for (int i = 0; i <= p.degree; i++) {
        result += p.coefficients[i] * pow(x, p.degree - i);
    }
    return result;
}

__declspec(dllexport) void print_polynomial(Polynomial p) {
    for (int i = 0; i <= p.degree; i++) {
        if (i == 0)
            printf("%.2f", p.coefficients[i]);
        else
            printf(" + %.2fx^%d", p.coefficients[i], p.degree - i);
    }
    printf("\n");
}

// Newton's method for root finding of a function
__declspec(dllexport) float newtons_method(Polynomial p, float (*derivative)(Polynomial, float), float initial_guess, float tolerance) {
    float x = initial_guess;
    float diff;
    
    do {
        float fx = evaluate_polynomial(p, x);
        float fpx = derivative(p, x);
        if (fpx == 0) {
            printf("Error: Derivative is zero, can't proceed with Newton's method\n");
            return x;
        }
        
        float x_new = x - fx / fpx;
        diff = fabs(x_new - x);
        x = x_new;
    } while (diff > tolerance);
    
    return x;
}

// Numerical integration (Trapezoidal rule)
__declspec(dllexport) float integrate(Polynomial p, float a, float b, int n) {
    float h = (b - a) / n;
    float sum = evaluate_polynomial(p, a) + evaluate_polynomial(p, b);
    
    for (int i = 1; i < n; i++) {
        float x = a + i * h;
        sum += 2 * evaluate_polynomial(p, x);
    }
    
    return sum * h / 2.0f;
}

// Statistical functions
__declspec(dllexport) float mean(float* data, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += data[i];
    }
    return sum / size;
}

__declspec(dllexport) float standard_deviation(float* data, int size) {
    float m = mean(data, size);
    float sum_sq_diff = 0.0f;
    
    for (int i = 0; i < size; i++) {
        sum_sq_diff += (data[i] - m) * (data[i] - m);
    }
    
    return sqrt(sum_sq_diff / size);
}

// Matrix inversion for 2x2 matrix
__declspec(dllexport) Matrix3x3 inverse_matrix(Matrix3x3 m) {
    float det = determinant(m);
    if (det == 0) {
        printf("Error: Matrix is singular and cannot be inverted\n");
        return m;  // Return original matrix if singular
    }

    Matrix3x3 result;
    result.m[0][0] = (m.m[1][1] * m.m[2][2] - m.m[1][2] * m.m[2][1]) / det;
    result.m[0][1] = (m.m[0][2] * m.m[2][1] - m.m[0][1] * m.m[2][2]) / det;
    result.m[0][2] = (m.m[0][1] * m.m[1][2] - m.m[0][2] * m.m[1][1]) / det;

    result.m[1][0] = (m.m[1][2] * m.m[2][0] - m.m[1][0] * m.m[2][2]) / det;
    result.m[1][1] = (m.m[0][0] * m.m[2][2] - m.m[0][2] * m.m[2][0]) / det;
    result.m[1][2] = (m.m[0][1] * m.m[1][0] - m.m[0][0] * m.m[1][1]) / det;

    result.m[2][0] = (m.m[1][0] * m.m[2][1] - m.m[1][1] * m.m[2][0]) / det;
    result.m[2][1] = (m.m[0][1] * m.m[2][0] - m.m[0][0] * m.m[2][1]) / det;
    result.m[2][2] = (m.m[0][0] * m.m[1][1] - m.m[0][1] * m.m[1][0]) / det;

    return result;
}

// Solving a system of linear equations (Ax = b) using Cramer's Rule
__declspec(dllexport) void solve_linear_system(Matrix3x3 A, Vector3 b) {
    // Assuming 3x3 matrix A and 3x1 vector b
    float det_A = determinant(A);
    if (det_A == 0) {
        printf("Error: The matrix is singular, no unique solution exists.\n");
        return;
    }

    // Using Cramer's Rule to find the solutions
    Matrix3x3 A1 = A, A2 = A, A3 = A;
    A1.m[0][0] = b.x; A1.m[1][0] = b.y; A1.m[2][0] = b.z; // Replace first column
    A2.m[0][1] = b.x; A2.m[1][1] = b.y; A2.m[2][1] = b.z; // Replace second column
    A3.m[0][2] = b.x; A3.m[1][2] = b.y; A3.m[2][2] = b.z; // Replace third column

    float x = determinant(A1) / det_A;
    float y = determinant(A2) / det_A;
    float z = determinant(A3) / det_A;

    printf("Solution: x = %.2f, y = %.2f, z = %.2f\n", x, y, z);
}

// Bisection method to find root of f(x) = 0 in the interval [a, b]
__declspec(dllexport) float bisection_method(float (*f)(float), float a, float b, float tol) {
    if (f(a) * f(b) >= 0) {
        printf("Error: The function has the same sign at both endpoints\n");
        return -1;
    }

    float c = a;
    while ((b - a) >= tol) {
        c = (a + b) / 2;
        if (f(c) == 0.0)
            break;
        else if (f(c) * f(a) < 0)
            b = c;
        else
            a = c;
    }

    return c;
}

// Exponentiation by squaring to calculate base^exponent
__declspec(dllexport) float power(float base, int exponent) {
    float result = 1.0f;
    while (exponent > 0) {
        if (exponent % 2 == 1) {
            result *= base;
        }
        base *= base;
        exponent /= 2;
    }
    return result;
}

// Sine approximation using Taylor series
__declspec(dllexport) float sine_taylor(float x) {
    float result = 0;
    float term = x;  // First term of Taylor series
    int n = 1;
    while (fabs(term) > 0.0001) {
        result += term;
        term *= -x * x / ((2 * n) * (2 * n + 1));  // Next term in series
        n++;
    }
    return result;
}

// Cosine approximation using Taylor series
__declspec(dllexport) float cosine_taylor(float x) {
    float result = 1;
    float term = 1;
    int n = 1;
    while (fabs(term) > 0.0001) {
        term *= -x * x / ((2 * n - 1) * (2 * n));  // Next term in series
        result += term;
        n++;
    }
    return result;
}

// Calculate trace of a matrix (sum of diagonal elements)
__declspec(dllexport) float trace(Matrix3x3 m) {
    return m.m[0][0] + m.m[1][1] + m.m[2][2];
}

// Natural logarithm approximation using Taylor series
__declspec(dllexport) float ln_taylor(float x) {
    if (x <= 0) {
        printf("Error: ln(x) is undefined for x <= 0\n");
        return -1;
    }

    float result = 0;
    float term = (x - 1) / (x + 1);
    float term_squared = term * term;
    float num = term;
    int n = 1;

    while (fabs(num) > 0.0001) {
        result += num / (2 * n - 1);
        num *= term_squared;
        n++;
    }

    return 2 * result;
}

// Base-10 logarithm approximation
__declspec(dllexport) float log10_taylor(float x) {
    return ln_taylor(x) / ln_taylor(10.0f);
}

// Polynomial differentiation (returning the derivative polynomial)
__declspec(dllexport) Polynomial differentiate_polynomial(Polynomial p) {
    Polynomial derivative;
    derivative.degree = p.degree - 1;
    derivative.coefficients = (float*)malloc(derivative.degree * sizeof(float));

    for (int i = 0; i < derivative.degree; i++) {
        derivative.coefficients[i] = p.coefficients[i] * (p.degree - i);
    }

    return derivative;
}

// Factorial (for non-negative integers)
__declspec(dllexport) long long factorial(int n) {
    if (n < 0) {
        printf("Error: Factorial is undefined for negative integers\n");
        return -1;
    }
    long long result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

// Binomial Coefficient (n choose k)
__declspec(dllexport) long long binomial_coefficient(int n, int k) {
    if (k > n) {
        printf("Error: k cannot be greater than n\n");
        return 0;
    }
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// Fibonacci sequence (iterative approach)
__declspec(dllexport) long long fibonacci(int n) {
    if (n < 0) {
        printf("Error: Fibonacci sequence is undefined for negative indices\n");
        return -1;
    }
    long long a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        long long temp = a + b;
        a = b;
        b = temp;
    }
    return (n == 0) ? a : b;
}

// Check if a number is prime
__declspec(dllexport) int is_prime(int n) {
    if (n <= 1) return 0;  // Numbers less than 2 are not prime
    for (int i = 2; i <= sqrt(n); i++) {
        if (n % i == 0) return 0;  // Not a prime
    }
    return 1;  // Prime number
}

// Sieve of Eratosthenes to find all primes up to n
__declspec(dllexport) void sieve_of_eratosthenes(int n) {
    if (n < 2) {
        printf("Error: No primes less than 2\n");
        return;
    }

    int *is_prime = (int*)malloc((n + 1) * sizeof(int));
    for (int i = 0; i <= n; i++) {
        is_prime[i] = 1;
    }

    is_prime[0] = is_prime[1] = 0;  // 0 and 1 are not prime

    for (int p = 2; p * p <= n; p++) {
        if (is_prime[p]) {
            for (int i = p * p; i <= n; i += p) {
                is_prime[i] = 0;
            }
        }
    }

    printf("Primes up to %d: ", n);
    for (int i = 2; i <= n; i++) {
        if (is_prime[i]) {
            printf("%d ", i);
        }
    }
    printf("\n");

    free(is_prime);
}

// Calculate GCD (Greatest Common Divisor) using Euclidean algorithm
__declspec(dllexport) int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// Calculate LCM (Least Common Multiple)
__declspec(dllexport) int lcm(int a, int b) {
    return abs(a * b) / gcd(a, b);
}

// Euler's Totient function (number of integers ≤ n that are coprime with n)
__declspec(dllexport) int euler_totient(int n) {
    int result = n;
    for (int i = 2; i * i <= n; i++) {
        if (n % i == 0) {
            while (n % i == 0) {
                n /= i;
            }
            result -= result / i;
        }
    }
    if (n > 1) {
        result -= result / n;
    }
    return result;
}

// Logarithmic Integral approximation (Li(x))
__declspec(dllexport) float li_approximation(float x) {
    if (x <= 0) {
        printf("Error: Li(x) is undefined for x <= 0\n");
        return -1;
    }

    float result = 0;
    float dx = 0.1f;  // Step size for numerical integration
    for (float t = 2.0f; t < x; t += dx) {
        result += 1 / (log(t) * dx);  // Integral approximation
    }
    return result;
}

// Hyperbolic sine
__declspec(dllexport) float sinh_function(float x) {
    return (exp(x) - exp(-x)) / 2.0f;
}

// Hyperbolic cosine
__declspec(dllexport) float cosh_function(float x) {
    return (exp(x) + exp(-x)) / 2.0f;
}

// Hyperbolic tangent
__declspec(dllexport) float tanh_function(float x) {
    return sinh_function(x) / cosh_function(x);
}

// Bessel function of the first kind (J0)
__declspec(dllexport) float bessel_j0(float x) {
    float sum = 0;
    int n = 0;
    float term;
    do {
        term = pow(-1, n) * pow(x / 2, 2 * n) / (factorial(n) * factorial(n));
        sum += term;
        n++;
    } while (fabs(term) > 0.0001f);
    return sum;
}

// Uniform random number generator between 0 and 1
__declspec(dllexport) float random_float() {
    return (float)rand() / RAND_MAX;
}

// Uniform random integer generator between min and max
__declspec(dllexport) int random_int(int min, int max) {
    return min + rand() % (max - min + 1);
}

// Sum of squares of integers from 1 to n
__declspec(dllexport) long long sum_of_squares(int n) {
    long long sum = 0;
    for (int i = 1; i <= n; i++) {
        sum += i * i;
    }
    return sum;
}

// Verify Pythagorean identity: sin^2(x) + cos^2(x) = 1
__declspec(dllexport) int verify_trig_identity(float x) {
    float sin_val = sine_taylor(x);
    float cos_val = cosine_taylor(x);
    return fabs(sin_val * sin_val + cos_val * cos_val - 1) < 0.0001f;
}
