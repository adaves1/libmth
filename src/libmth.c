
// libmth.c

#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>  // For INT_MAX
#include <stdbool.h> // For bool
#include <float.h>

#ifndef M_PI
    #define M_PI 3.14159265358979323846
#endif



#define INPUT_NODES 3     // Input layer nodes
#define HIDDEN_NODES 4    // Hidden layer nodes
#define OUTPUT_NODES 1    // Output layer nodes
#define LEARNING_RATE 0.5f
#define EPOCHS 10000      // Number of iterations for training
#define BLOCK_SIZE 16 // AES block size in bytes (128 bits)
#define KEY_SIZE 16   // AES key size for AES-128 (128 bits)


float sigmoid(float x) {
    return 1.0f / (1.0f + exp(-x));
}

// Derivative of Sigmoid
float sigmoid_derivative(float x) {
    return x * (1.0f - x);
}

// Random Initialization of Weights
float random_weight() {
    return (float)rand() / (float)RAND_MAX * 2.0f - 1.0f;  // Random between -1 and 1
}

// Neural Network Structure
typedef struct {
    float input[INPUT_NODES];
    float hidden[HIDDEN_NODES];
    float output[OUTPUT_NODES];
    
    float input_to_hidden_weights[INPUT_NODES][HIDDEN_NODES];
    float hidden_to_output_weights[HIDDEN_NODES][OUTPUT_NODES];

    float hidden_bias[HIDDEN_NODES];
    float output_bias[OUTPUT_NODES];
} NeuralNetwork;

// Initialize the Neural Network
void initialize_network(NeuralNetwork* nn) {
    srand(time(NULL));

    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->input_to_hidden_weights[i][j] = random_weight();
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->hidden_to_output_weights[i][j] = random_weight();
        }
    }

    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->hidden_bias[i] = random_weight();
    }

    for (int i = 0; i < OUTPUT_NODES; i++) {
        nn->output_bias[i] = random_weight();
    }
}

// Feedforward Process
void feedforward(NeuralNetwork* nn) {
    // Hidden layer calculations
    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->hidden[i] = 0.0f;
        for (int j = 0; j < INPUT_NODES; j++) {
            nn->hidden[i] += nn->input[j] * nn->input_to_hidden_weights[j][i];
        }
        nn->hidden[i] += nn->hidden_bias[i];
        nn->hidden[i] = sigmoid(nn->hidden[i]);  // Apply activation function
    }

    // Output layer calculations
    for (int i = 0; i < OUTPUT_NODES; i++) {
        nn->output[i] = 0.0f;
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->output[i] += nn->hidden[j] * nn->hidden_to_output_weights[j][i];
        }
        nn->output[i] += nn->output_bias[i];
        nn->output[i] = sigmoid(nn->output[i]);  // Apply activation function
    }
}

// Backpropagation (Training)
void backpropagate(NeuralNetwork* nn, float* expected_output) {
    float output_error[OUTPUT_NODES];
    float hidden_error[HIDDEN_NODES];
    float output_delta[OUTPUT_NODES];
    float hidden_delta[HIDDEN_NODES];

    // Calculate error at output layer
    for (int i = 0; i < OUTPUT_NODES; i++) {
        output_error[i] = expected_output[i] - nn->output[i];
        output_delta[i] = output_error[i] * sigmoid_derivative(nn->output[i]);
    }

    // Calculate error at hidden layer
    for (int i = 0; i < HIDDEN_NODES; i++) {
        hidden_error[i] = 0.0f;
        for (int j = 0; j < OUTPUT_NODES; j++) {
            hidden_error[i] += output_delta[j] * nn->hidden_to_output_weights[i][j];
        }
        hidden_delta[i] = hidden_error[i] * sigmoid_derivative(nn->hidden[i]);
    }

    // Update weights between hidden and output layer
    for (int i = 0; i < HIDDEN_NODES; i++) {
        for (int j = 0; j < OUTPUT_NODES; j++) {
            nn->hidden_to_output_weights[i][j] += LEARNING_RATE * output_delta[j] * nn->hidden[i];
        }
    }

    // Update weights between input and hidden layer
    for (int i = 0; i < INPUT_NODES; i++) {
        for (int j = 0; j < HIDDEN_NODES; j++) {
            nn->input_to_hidden_weights[i][j] += LEARNING_RATE * hidden_delta[j] * nn->input[i];
        }
    }

    // Update biases
    for (int i = 0; i < OUTPUT_NODES; i++) {
        nn->output_bias[i] += LEARNING_RATE * output_delta[i];
    }
    for (int i = 0; i < HIDDEN_NODES; i++) {
        nn->hidden_bias[i] += LEARNING_RATE * hidden_delta[i];
    }
}

// Training Loop
void train(NeuralNetwork* nn, float* inputs, float* outputs, int num_samples) {
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        for (int i = 0; i < num_samples; i++) {
            // Set inputs
            for (int j = 0; j < INPUT_NODES; j++) {
                nn->input[j] = inputs[i * INPUT_NODES + j];
            }

            // Feedforward
            feedforward(nn);

            // Backpropagate
            backpropagate(nn, &outputs[i]);

            // Optionally print out progress
            if (epoch % 1000 == 0) {
                printf("Epoch %d, Sample %d, Output: %f, Expected: %f\n", epoch, i, nn->output[0], outputs[i]);
            }
        }
    }
}

// Example usage of the neural network
int main() {
    NeuralNetwork nn;
    initialize_network(&nn);

    // Example training data: XOR problem (2 inputs, 1 output)
    float inputs[4][INPUT_NODES] = {
        {0.0f, 0.0f, 1.0f},
        {0.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 1.0f},
        {1.0f, 1.0f, 1.0f}
    };

    float outputs[4] = {0.0f, 1.0f, 1.0f, 0.0f};

    // Flatten the 2D input array to 1D array for training loop
    train(&nn, (float*)inputs, outputs, 4);

    return 0;
}

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

// Matrix multiplication (NxN matrix)
__declspec(dllexport) void multiply_matrices(Matrix3x3 A, Matrix3x3 B, Matrix3x3* result) {
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            result->m[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                result->m[i][j] += A.m[i][k] * B.m[k][j];
            }
        }
    }
}

// Matrix transposition (transpose a 3x3 matrix)
__declspec(dllexport) void transpose_matrix(Matrix3x3* m) {
    for (int i = 0; i < 3; i++) {
        for (int j = i + 1; j < 3; j++) {
            float temp = m->m[i][j];
            m->m[i][j] = m->m[j][i];
            m->m[j][i] = temp;
        }
    }
}

// QR Decomposition (using Gram-Schmidt process)
__declspec(dllexport) void qr_decomposition(Matrix3x3* A, Matrix3x3* Q, Matrix3x3* R) {
    // This is a simplified QR decomposition using the Gram-Schmidt process
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            Q->m[i][j] = A->m[i][j];  // Copy A into Q
        }
    }

    for (int i = 0; i < 3; i++) {
        // Normalize each column of Q
        float norm = 0;
        for (int j = 0; j < 3; j++) {
            norm += Q->m[j][i] * Q->m[j][i];
        }
        norm = sqrt(norm);
        for (int j = 0; j < 3; j++) {
            Q->m[j][i] /= norm;
        }

        // Calculate R
        for (int j = 0; j < 3; j++) {
            R->m[i][j] = 0;
            for (int k = 0; k < 3; k++) {
                R->m[i][j] += Q->m[k][i] * A->m[k][j];
            }
        }
    }
}

// Eigenvalue calculation using the Power Method for the largest eigenvalue
__declspec(dllexport) float power_method_eigenvalue(Matrix3x3* A, float tolerance) {
    // Using random vector to start
    float eigenvalue = 0;
    float vector[3] = {1.0f, 1.0f, 1.0f};
    float prev_eigenvalue = 0;

    while (1) {
        // Multiply matrix A by vector
        float result[3] = {0};
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                result[i] += A->m[i][j] * vector[j];
            }
        }

        // Normalize the result
        float norm = 0;
        for (int i = 0; i < 3; i++) {
            norm += result[i] * result[i];
        }
        norm = sqrt(norm);

        for (int i = 0; i < 3; i++) {
            vector[i] = result[i] / norm;
        }

        // Compute the eigenvalue approximation (Rayleigh quotient)
        eigenvalue = 0;
        for (int i = 0; i < 3; i++) {
            eigenvalue += vector[i] * result[i];
        }

        if (fabs(eigenvalue - prev_eigenvalue) < tolerance) {
            break;
        }

        prev_eigenvalue = eigenvalue;
    }

    return eigenvalue;
}

// Monte Carlo Integration for a function f(x) over the interval [a, b]
__declspec(dllexport) float monte_carlo_integration(float (*f)(float), float a, float b, int num_samples) {
    float sum = 0.0f;
    for (int i = 0; i < num_samples; i++) {
        float x = a + (b - a) * random_float();
        sum += f(x);
    }
    return (b - a) * sum / num_samples;
}

// 4th-Order Runge-Kutta method for solving ODEs: dy/dx = f(x, y)
__declspec(dllexport) float runge_kutta(float (*f)(float, float), float y0, float x0, float xn, float h) {
    int steps = (int)((xn - x0) / h);
    float y = y0;
    float x = x0;

    for (int i = 0; i < steps; i++) {
        float k1 = h * f(x, y);
        float k2 = h * f(x + h / 2, y + k1 / 2);
        float k3 = h * f(x + h / 2, y + k2 / 2);
        float k4 = h * f(x + h, y + k3);

        y += (k1 + 2*k2 + 2*k3 + k4) / 6;
        x += h;
    }

    return y;
}

// 1D Fast Fourier Transform (FFT) using Cooley-Tukey algorithm
__declspec(dllexport) void fft(float* real, float* imag, int n) {
    int j = 0;
    for (int i = 1; i < n; i++) {
        int bit = n >> 1;
        for (; j >= bit; bit >>= 1) j -= bit;
        j += bit;
        if (i < j) {
            float temp_real = real[i];
            float temp_imag = imag[i];
            real[i] = real[j];
            imag[i] = imag[j];
            real[j] = temp_real;
            imag[j] = temp_imag;
        }
    }

    for (int size = 2; size <= n; size <<= 1) {
        float angle = -2.0f * M_PI / size;
        for (int i = 0; i < n; i += size) {
            for (int j = 0; j < size / 2; j++) {
                float real_part = cos(j * angle);
                float imag_part = sin(j * angle);
                float temp_real = real[i + j + size / 2] * real_part - imag[i + j + size / 2] * imag_part;
                float temp_imag = real[i + j + size / 2] * imag_part + imag[i + j + size / 2] * real_part;
                real[i + j + size / 2] = real[i + j] - temp_real;
                imag[i + j + size / 2] = imag[i + j] - temp_imag;
                real[i + j] += temp_real;
                imag[i + j] += temp_imag;
            }
        }
    }
}

// 1D Convolution (Apply a kernel to a signal)
__declspec(dllexport) void convolve_1d(float* signal, float* kernel, float* output, int signal_len, int kernel_len) {
    int pad = kernel_len / 2;
    for (int i = 0; i < signal_len; i++) {
        output[i] = 0;
        for (int j = 0; j < kernel_len; j++) {
            int idx = i + j - pad;
            if (idx >= 0 && idx < signal_len) {
                output[i] += signal[idx] * kernel[j];
            }
        }
    }
}

// 2D Convolution (Apply a kernel to a 2D matrix/image)
__declspec(dllexport) void convolve_2d(float** image, float** kernel, float** output, int image_rows, int image_cols, int kernel_size) {
    int pad = kernel_size / 2;

    // Loop through each pixel in the image
    for (int i = 0; i < image_rows; i++) {
        for (int j = 0; j < image_cols; j++) {
            float sum = 0.0f;

            // Loop through each element in the kernel
            for (int m = -pad; m <= pad; m++) {
                for (int n = -pad; n <= pad; n++) {
                    // Check if the pixel is within image bounds (ignoring out-of-bounds regions)
                    if (i + m >= 0 && i + m < image_rows && j + n >= 0 && j + n < image_cols) {
                        sum += image[i + m][j + n] * kernel[pad + m][pad + n];
                    }
                }
            }

            // Assign the computed value to the output pixel
            output[i][j] = sum;
        }
    }
}

// Linear regression using least squares method (y = mx + b)
__declspec(dllexport) void linear_regression(float* x, float* y, float* m, float* b, int n) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

    for (int i = 0; i < n; i++) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    // Calculate slope (m) and intercept (b)
    *m = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    *b = (sum_y - (*m) * sum_x) / n;
}

// Binary Search (Find the index of a number in a sorted array)
__declspec(dllexport) int binary_search(int* arr, int left, int right, int target) {
    while (left <= right) {
        int mid = left + (right - left) / 2;

        // Check if target is at mid
        if (arr[mid] == target)
            return mid;

        // If target is greater, ignore the left half
        if (arr[mid] < target)
            left = mid + 1;
        // If target is smaller, ignore the right half
        else
            right = mid - 1;
    }

    return -1; // Target not found
}

// Calculate the determinant of a 3x3 matrix
__declspec(dllexport) float determinant_3x3(float matrix[3][3]) {
    return matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[2][1] * matrix[1][2]) -
           matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[2][0] * matrix[1][2]) +
           matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[2][0] * matrix[1][1]);
}

// Fast inverse square root (popularized by Quake III)
__declspec(dllexport) float fast_inverse_sqrt(float x) {
    long i = *(long*)&x;
    i = 0x5f3759df - (i >> 1);
    x = *(float*)&i;
    x = x * (1.5f - 0.5f * x * x);  // One iteration for accuracy
    return x;
}

// K-means clustering algorithm for clustering points into k clusters
__declspec(dllexport) void kmeans_clustering(float* points, float* centroids, int* labels, int num_points, int num_clusters, int num_iterations) {
    for (int iter = 0; iter < num_iterations; iter++) {
        // Assign each point to the nearest centroid
        for (int i = 0; i < num_points; i++) {
            float min_dist = FLT_MAX;
            int cluster = 0;
            for (int j = 0; j < num_clusters; j++) {
                float dist = 0;
                for (int k = 0; k < 2; k++) {  // Assuming 2D points
                    dist += (points[i * 2 + k] - centroids[j * 2 + k]) * (points[i * 2 + k] - centroids[j * 2 + k]);
                }
                if (dist < min_dist) {
                    min_dist = dist;
                    cluster = j;
                }
            }
            labels[i] = cluster;
        }

        // Recompute centroids
        for (int j = 0; j < num_clusters; j++) {
            float sum_x = 0, sum_y = 0;
            int count = 0;
            for (int i = 0; i < num_points; i++) {
                if (labels[i] == j) {
                    sum_x += points[i * 2];
                    sum_y += points[i * 2 + 1];
                    count++;
                }
            }
            centroids[j * 2] = sum_x / count;
            centroids[j * 2 + 1] = sum_y / count;
        }
    }
}

// Dijkstra's Algorithm to find the shortest path in a graph (represented by adjacency matrix)
__declspec(dllexport) void dijkstra(int graph[5][5], int start, int* dist, int* prev, int num_vertices) {
    bool visited[5] = { false };
    for (int i = 0; i < num_vertices; i++) {
        dist[i] = INT_MAX;
        prev[i] = -1;
    }
    dist[start] = 0;

    for (int i = 0; i < num_vertices; i++) {
        int u = -1;
        for (int j = 0; j < num_vertices; j++) {
            if (!visited[j] && (u == -1 || dist[j] < dist[u])) {
                u = j;
            }
        }
        visited[u] = true;

        for (int v = 0; v < num_vertices; v++) {
            if (graph[u][v] && dist[u] + graph[u][v] < dist[v]) {
                dist[v] = dist[u] + graph[u][v];
                prev[v] = u;
            }
        }
    }
}

// Numerical integration using Simpson's Rule
__declspec(dllexport) float simpsons_rule(float (*f)(float), float a, float b, int n) {
    if (n % 2 == 1) n++;  // Ensure n is even
    float h = (b - a) / n;
    float sum = f(a) + f(b);

    for (int i = 1; i < n; i += 2) {
        sum += 4 * f(a + i * h);
    }

    for (int i = 2; i < n - 1; i += 2) {
        sum += 2 * f(a + i * h);
    }

    return sum * h / 3;
}

// Basic Turing Machine Simulator
__declspec(dllexport) void turing_machine_simulator(int* tape, int tape_size, int* instructions, int num_instructions, int start_state) {
    int current_state = start_state;
    int head_position = tape_size / 2; // Start in the middle

    while (1) {
        int instruction = instructions[current_state * 2 + tape[head_position]];
        if (instruction == -1) break; // Halt condition

        // Execute the instruction
        tape[head_position] = instruction & 0x1;
        current_state = (instruction >> 1) & 0xFF;
        head_position += (instruction >> 9) & 0x1 ? 1 : -1;  // Move head
    }
}

// Binary Matrix Multiplication (Assuming matrices are 0s and 1s)
__declspec(dllexport) void bin_matrix_multiply(int* A, int* B, int* C, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            C[i * n + j] = 0;
            for (int k = 0; k < n; k++) {
                C[i * n + j] ^= (A[i * n + k] & B[k * n + j]);
            }
        }
    }
}

// Cholesky Decomposition: A = L * L^T
__declspec(dllexport) int cholesky_decomposition(float* matrix, float* L, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < i + 1; j++) {
            float sum = matrix[i * n + j];
            for (int k = 0; k < j; k++) {
                sum -= L[i * n + k] * L[j * n + k];
            }

            if (i == j) {
                if (sum <= 0) return -1;  // Matrix is not positive definite
                L[i * n + j] = sqrt(sum);
            } else {
                L[i * n + j] = sum / L[j * n + j];
            }
        }
    }
    return 0;
}

// Monte Carlo method for estimating the value of Pi
__declspec(dllexport) float monte_carlo_pi(int num_samples) {
    int inside_circle = 0;
    for (int i = 0; i < num_samples; i++) {
        float x = (float)rand() / RAND_MAX;
        float y = (float)rand() / RAND_MAX;
        
        if (x * x + y * y <= 1) {
            inside_circle++;
        }
    }
    return 4.0f * inside_circle / num_samples;
}

// Lagrange interpolation (finds a polynomial that fits the given data points)
__declspec(dllexport) float lagrange_interpolation(float* x_points, float* y_points, int n, float x) {
    float result = 0.0f;
    for (int i = 0; i < n; i++) {
        float term = y_points[i];
        for (int j = 0; j < n; j++) {
            if (j != i) {
                term *= (x - x_points[j]) / (x_points[i] - x_points[j]);
            }
        }
        result += term;
    }
    return result;
}

// Floyd-Warshall Algorithm: Finds shortest paths between all pairs of vertices
__declspec(dllexport) void floyd_warshall(int* graph, int* dist, int num_vertices) {
    // Initialize the distance matrix with the graph's weights
    for (int i = 0; i < num_vertices; i++) {
        for (int j = 0; j < num_vertices; j++) {
            dist[i * num_vertices + j] = graph[i * num_vertices + j];
        }
    }

    // Apply Floyd-Warshall dynamic programming algorithm
    for (int k = 0; k < num_vertices; k++) {
        for (int i = 0; i < num_vertices; i++) {
            for (int j = 0; j < num_vertices; j++) {
                if (dist[i * num_vertices + j] > dist[i * num_vertices + k] + dist[k * num_vertices + j]) {
                    dist[i * num_vertices + j] = dist[i * num_vertices + k] + dist[k * num_vertices + j];
                }
            }
        }
    }
}

// Example Usage:
// Graph represented as an adjacency matrix (directed graph)
// -1 means no path exists between nodes
// dist will contain the shortest distances between all pairs of nodes after execution

static const uint8_t SBOX[256] = {
    // The S-box values go here, which are pre-calculated
    // This is a truncated version for brevity
    0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76, 0xca,
    0x82, 0xc9, 0x7d, 0x9d, 0x2d, 0x8f, 0x98, 0x11, 0x69, 0xd9, 0x8d, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce,
    // The rest of the S-box values would be added here...
};

// AES round constant
static const uint8_t RCON[10] = {
    0x8d, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, 0x80, 0x1b
};

// SubBytes - Substitution step using S-box
void subBytes(uint8_t *state) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        state[i] = SBOX[state[i]];
    }
}

// ShiftRows - Rows in the state are shifted
void shiftRows(uint8_t *state) {
    uint8_t temp[BLOCK_SIZE];

    // Row 1 is shifted 1 byte to the left
    temp[0] = state[4]; temp[1] = state[5]; temp[2] = state[6]; temp[3] = state[7];
    state[4] = temp[0]; state[5] = temp[1]; state[6] = temp[2]; state[7] = temp[3];

    // Row 2 is shifted 2 bytes to the left
    temp[0] = state[8]; temp[1] = state[9]; temp[2] = state[10]; temp[3] = state[11];
    state[8] = temp[0]; state[9] = temp[1]; state[10] = temp[2]; state[11] = temp[3];

    // Row 3 is shifted 3 bytes to the left
    temp[0] = state[12]; temp[1] = state[13]; temp[2] = state[14]; temp[3] = state[15];
    state[12] = temp[0]; state[13] = temp[1]; state[14] = temp[2]; state[15] = temp[3];
}

// MixColumns - Mixes the columns of the state matrix
void mixColumns(uint8_t *state) {
    uint8_t temp[BLOCK_SIZE];
    for (int i = 0; i < 4; i++) {
        temp[0] = state[i];
        temp[1] = state[i + 4];
        temp[2] = state[i + 8];
        temp[3] = state[i + 12];

        state[i] = (temp[0] ^ temp[1] ^ temp[2] ^ temp[3]);
        state[i + 4] = (temp[1] ^ temp[2] ^ temp[3] ^ temp[0]);
        state[i + 8] = (temp[2] ^ temp[3] ^ temp[0] ^ temp[1]);
        state[i + 12] = (temp[3] ^ temp[0] ^ temp[1] ^ temp[2]);
    }
}

// AddRoundKey - XORs the round key with the state
void addRoundKey(uint8_t *state, uint8_t *roundKey) {
    for (int i = 0; i < BLOCK_SIZE; i++) {
        state[i] ^= roundKey[i];
    }
}

// AES Encryption (AES-128)
void aes_encrypt(uint8_t *state, uint8_t *roundKeys) {
    addRoundKey(state, roundKeys);

    for (int round = 1; round < 10; round++) {
        subBytes(state);
        shiftRows(state);
        mixColumns(state);
        addRoundKey(state, roundKeys + round * BLOCK_SIZE);
    }

    subBytes(state);
    shiftRows(state);
    addRoundKey(state, roundKeys + 10 * BLOCK_SIZE); // Final round key
}

// AES Decryption (AES-128)
void aes_decrypt(uint8_t *state, uint8_t *roundKeys) {
    addRoundKey(state, roundKeys + 10 * BLOCK_SIZE);

    for (int round = 9; round > 0; round--) {
        // Reverse operations
        shiftRows(state);
        subBytes(state);
        addRoundKey(state, roundKeys + round * BLOCK_SIZE);
        mixColumns(state);
    }

    shiftRows(state);
    subBytes(state);
    addRoundKey(state, roundKeys); // Initial round key
}

__declspec(dllexport) void rotate_point(float x, float y, float angle, float* x_out, float* y_out) {
    float radians = angle * M_PI / 180.0f;  // Convert angle to radians
    *x_out = x * cos(radians) - y * sin(radians);
    *y_out = x * sin(radians) + y * cos(radians);
}

__declspec(dllexport) void dfs(int graph[5][5], int start, bool* visited, int num_vertices) {
    visited[start] = true;
    for (int v = 0; v < num_vertices; v++) {
        if (graph[start][v] != 0 && !visited[v]) {
            dfs(graph, v, visited, num_vertices);
        }
    }
}

__declspec(dllexport) void moving_average(float* data, int data_size, int window_size, float* output) {
    for (int i = 0; i <= data_size - window_size; i++) {
        float sum = 0;
        for (int j = 0; j < window_size; j++) {
            sum += data[i + j];
        }
        output[i] = sum / window_size;
    }
}

static int compare(const void* a, const void* b) {
    float fa = *(const float*)a;
    float fb = *(const float*)b;
    return (fa > fb) - (fa < fb);
}

__declspec(dllexport) float find_median(float* arr, int size) {
    qsort(arr, size, sizeof(float), compare);
    if (size % 2 == 0) {
        return (arr[size / 2 - 1] + arr[size / 2]) / 2.0f;
    } else {
        return arr[size / 2];
    }
}

__declspec(dllexport) void prims_mst(int graph[5][5], int num_vertices, int* parent) {
    int key[5];
    bool mst_set[5] = { false };

    for (int i = 0; i < num_vertices; i++) {
        key[i] = INT_MAX;
        parent[i] = -1;
    }
    key[0] = 0;

    for (int count = 0; count < num_vertices - 1; count++) {
        int min = INT_MAX, u = -1;

        for (int v = 0; v < num_vertices; v++) {
            if (!mst_set[v] && key[v] < min) {
                min = key[v];
                u = v;
            }
        }

        mst_set[u] = true;

        for (int v = 0; v < num_vertices; v++) {
            if (graph[u][v] && !mst_set[v] && graph[u][v] < key[v]) {
                parent[v] = u;
                key[v] = graph[u][v];
            }
        }
    }
}

__declspec(dllexport) float chi_squared(float* observed, float* expected, int size) {
    float chi_sq = 0;
    for (int i = 0; i < size; i++) {
        if (expected[i] != 0) {
            float diff = observed[i] - expected[i];
            chi_sq += (diff * diff) / expected[i];
        }
    }
    return chi_sq;
}

__declspec(dllexport) void reflect_point(float x, float y, float x_ref, float y_ref, float* x_out, float* y_out) {
    *x_out = 2 * x_ref - x;
    *y_out = 2 * y_ref - y;
}

__declspec(dllexport) void vector_cross_product(float* vec1, float* vec2, float* result) {
    result[0] = vec1[1] * vec2[2] - vec1[2] * vec2[1];
    result[1] = vec1[2] * vec2[0] - vec1[0] * vec2[2];
    result[2] = vec1[0] * vec2[1] - vec1[1] * vec2[0];
}


__declspec(dllexport) int invert_matrix_2x2(float matrix[2][2], float inverse[2][2]) {
    float det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    if (det == 0) {
        return 0;  // Not invertible
    }

    float inv_det = 1.0f / det;
    inverse[0][0] = matrix[1][1] * inv_det;
    inverse[0][1] = -matrix[0][1] * inv_det;
    inverse[1][0] = -matrix[1][0] * inv_det;
    inverse[1][1] = matrix[0][0] * inv_det;

    return 1;  // Success
}

__declspec(dllexport) int gaussian_elimination(float** matrix, float* results, int n) {
    for (int i = 0; i < n; i++) {
        // Make the diagonal element 1
        float pivot = matrix[i][i];
        if (pivot == 0) return 0;  // Singular matrix

        for (int j = i; j <= n; j++) {
            matrix[i][j] /= pivot;
        }

        // Make the other rows 0 in the current column
        for (int k = 0; k < n; k++) {
            if (k != i) {
                float factor = matrix[k][i];
                for (int j = i; j <= n; j++) {
                    matrix[k][j] -= factor * matrix[i][j];
                }
            }
        }
    }

    for (int i = 0; i < n; i++) {
        results[i] = matrix[i][n];
    }

    return 1;  // Success
}

__declspec(dllexport) float distance_3d(float x1, float y1, float z1, float x2, float y2, float z2) {
    float dx = x2 - x1;
    float dy = y2 - y1;
    float dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}

__declspec(dllexport) float bilinear_interpolation(float x, float y, float* grid, int width) {
    int x0 = (int)x, y0 = (int)y;
    int x1 = x0 + 1, y1 = y0 + 1;

    float dx = x - x0;
    float dy = y - y0;

    float f00 = grid[y0 * width + x0];
    float f01 = grid[y1 * width + x0];
    float f10 = grid[y0 * width + x1];
    float f11 = grid[y1 * width + x1];

    return (1 - dx) * (1 - dy) * f00 +
           (1 - dx) * dy * f01 +
           dx * (1 - dy) * f10 +
           dx * dy * f11;
}

__declspec(dllexport) int lu_decomposition(float** matrix, float** lower, float** upper, int n) {
    for (int i = 0; i < n; i++) {
        // Upper Triangular
        for (int k = i; k < n; k++) {
            float sum = 0;
            for (int j = 0; j < i; j++) {
                sum += (lower[i][j] * upper[j][k]);
            }
            upper[i][k] = matrix[i][k] - sum;
        }

        // Lower Triangular
        for (int k = i; k < n; k++) {
            if (i == k) {
                lower[i][i] = 1;  // Diagonal as 1
            } else {
                float sum = 0;
                for (int j = 0; j < i; j++) {
                    sum += (lower[k][j] * upper[j][i]);
                }
                lower[k][i] = (matrix[k][i] - sum) / upper[i][i];
            }
        }
    }
    return 1;  // Success
}

__declspec(dllexport) int hamming_distance(int x, int y) {
    int xor_result = x ^ y;
    int count = 0;
    while (xor_result) {
        count += xor_result & 1;
        xor_result >>= 1;
    }
    return count;
}


__declspec(dllexport) int levenshtein_distance(const char* s1, const char* s2) {
    int len1 = strlen(s1);
    int len2 = strlen(s2);
    int dp[len1 + 1][len2 + 1];

    for (int i = 0; i <= len1; i++) dp[i][0] = i;
    for (int j = 0; j <= len2; j++) dp[0][j] = j;

    for (int i = 1; i <= len1; i++) {
        for (int j = 1; j <= len2; j++) {
            int cost = (s1[i - 1] == s2[j - 1]) ? 0 : 1;
            dp[i][j] = fmin(dp[i - 1][j] + 1, fmin(dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost));
        }
    }
    return dp[len1][len2];
}

__declspec(dllexport) float gradient_descent(float a, float b, float c, float alpha, int iterations) {
    float x = 0;  // Start at x = 0
    for (int i = 0; i < iterations; i++) {
        float gradient = 2 * a * x + b;  // Derivative of f(x)
        x -= alpha * gradient;          // Update x using learning rate
    }
    return x;
}

__declspec(dllexport) void calculate_statistics(float* data, int n, float* mean, float* median, float* mode) {
    // Calculate mean
    float sum = 0;
    for (int i = 0; i < n; i++) {
        sum += data[i];
    }
    *mean = sum / n;

    // Calculate median
    qsort(data, n, sizeof(float), (int(*)(const void*, const void*))strcmp);
    if (n % 2 == 0) {
        *median = (data[n / 2 - 1] + data[n / 2]) / 2.0f;
    } else {
        *median = data[n / 2];
    }

    // Calculate mode
    int max_count = 0;
    float current_mode = data[0];
    int count = 1;
    for (int i = 1; i < n; i++) {
        if (data[i] == data[i - 1]) {
            count++;
        } else {
            count = 1;
        }
        if (count > max_count) {
            max_count = count;
            current_mode = data[i];
        }
    }
    *mode = current_mode;
}

__declspec(dllexport) float matrix_determinant(float** matrix, int n) {
    if (n == 1) return matrix[0][0];

    float det = 0;
    float** submatrix = (float**)malloc((n - 1) * sizeof(float*));
    for (int i = 0; i < n - 1; i++) {
        submatrix[i] = (float*)malloc((n - 1) * sizeof(float));
    }

    for (int x = 0; x < n; x++) {
        int subi = 0;
        for (int i = 1; i < n; i++) {
            int subj = 0;
            for (int j = 0; j < n; j++) {
                if (j == x) continue;
                submatrix[subi][subj] = matrix[i][j];
                subj++;
            }
            subi++;
        }
        det += (x % 2 == 0 ? 1 : -1) * matrix[0][x] * matrix_determinant(submatrix, n - 1);
    }

    for (int i = 0; i < n - 1; i++) free(submatrix[i]);
    free(submatrix);

    return det;
}

__declspec(dllexport) void cross_product_3d(float* v1, float* v2, float* result) {
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

__declspec(dllexport) int fast_modular_exponentiation(int base, int exp, int mod) {
    int result = 1;
    base %= mod;
    while (exp > 0) {
        if (exp % 2 == 1) {
            result = (result * base) % mod;
        }
        exp /= 2;
        base = (base * base) % mod;
    }
    return result;
}

__declspec(dllexport) void bresenham_line(int x1, int y1, int x2, int y2, int* line_x, int* line_y, int* count) {
    int dx = abs(x2 - x1);
    int dy = abs(y2 - y1);
    int sx = (x1 < x2) ? 1 : -1;
    int sy = (y1 < y2) ? 1 : -1;
    int err = dx - dy;

    *count = 0;
    while (1) {
        line_x[*count] = x1;
        line_y[*count] = y1;
        (*count)++;
        if (x1 == x2 && y1 == y2) break;
        int e2 = 2 * err;
        if (e2 > -dy) {
            err -= dy;
            x1 += sx;
        }
        if (e2 < dx) {
            err += dx;
            y1 += sy;
        }
    }
}

__declspec(dllexport) int matrix_inverse(float** matrix, float** inverse, int n) {
    float** augmented = (float**)malloc(n * sizeof(float*));
    for (int i = 0; i < n; i++) {
        augmented[i] = (float*)malloc(2 * n * sizeof(float));
        for (int j = 0; j < n; j++) {
            augmented[i][j] = matrix[i][j];
            augmented[i][j + n] = (i == j) ? 1 : 0;
        }
    }

    for (int i = 0; i < n; i++) {
        float pivot = augmented[i][i];
        if (pivot == 0) return 0;  // Singular matrix

        for (int j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        for (int k = 0; k < n; k++) {
            if (k == i) continue;
            float factor = augmented[k][i];
            for (int j = 0; j < 2 * n; j++) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            inverse[i][j] = augmented[i][j + n];
        }
        free(augmented[i]);
    }
    free(augmented);
    return 1;  // Success
}

__declspec(dllexport) float hermite_polynomial(int n, float x) {
    if (n == 0) return 1;
    if (n == 1) return 2 * x;
    return 2 * x * hermite_polynomial(n - 1, x) - 2 * (n - 1) * hermite_polynomial(n - 2, x);
}

__declspec(dllexport) float newton_raphson(float (*func)(float), float (*deriv)(float), float guess, float tolerance, int max_iter) {
    int iter = 0;
    while (iter < max_iter) {
        float value = func(guess);
        float slope = deriv(guess);
        if (fabs(slope) < 1e-7) break;  // Prevent division by zero
        float next_guess = guess - value / slope;
        if (fabs(next_guess - guess) < tolerance) return next_guess;
        guess = next_guess;
        iter++;
    }
    return guess;  // Return last guess if max iterations reached
}

__declspec(dllexport) void string_permutations(char* str, int l, int r, void (*callback)(const char*)) {
    if (l == r) {
        callback(str);
    } else {
        for (int i = l; i <= r; i++) {
            char temp = str[l];
            str[l] = str[i];
            str[i] = temp;

            string_permutations(str, l + 1, r, callback);

            temp = str[l];
            str[l] = str[i];
            str[i] = temp;
        }
    }
}

__declspec(dllexport) void discrete_fourier_transform(float* real, float* imag, int n) {
    float* temp_real = (float*)malloc(n * sizeof(float));
    float* temp_imag = (float*)malloc(n * sizeof(float));

    for (int k = 0; k < n; k++) {
        temp_real[k] = 0;
        temp_imag[k] = 0;
        for (int t = 0; t < n; t++) {
            float angle = -2.0f * M_PI * k * t / n;
            temp_real[k] += real[t] * cos(angle) - imag[t] * sin(angle);
            temp_imag[k] += real[t] * sin(angle) + imag[t] * cos(angle);
        }
    }

    for (int i = 0; i < n; i++) {
        real[i] = temp_real[i];
        imag[i] = temp_imag[i];
    }

    free(temp_real);
    free(temp_imag);
}

__declspec(dllexport) float logarithm_base(float value, float base) {
    return log(value) / log(base);
}

__declspec(dllexport) void quicksort(int* array, int low, int high) {
    if (low < high) {
        int pivot = array[high];
        int i = low - 1;
        for (int j = low; j < high; j++) {
            if (array[j] < pivot) {
                i++;
                int temp = array[i];
                array[i] = array[j];
                array[j] = temp;
            }
        }
        int temp = array[i + 1];
        array[i + 1] = array[high];
        array[high] = temp;

        int pi = i + 1;
        quicksort(array, low, pi - 1);
        quicksort(array, pi + 1, high);
    }
}

__declspec(dllexport) float square(float x) {
    return x * x;
}

__declspec(dllexport) float radians_to_degrees(float radians) {
    return radians * (180.0f / M_PI);
}

__declspec(dllexport) void reverse_string(char* str) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        char temp = str[i];
        str[i] = str[n - i - 1];
        str[n - i - 1] = temp;
    }
}

__declspec(dllexport) int word_count(const char* str) {
    int count = 0;
    int in_word = 0;
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] == ' ' || str[i] == '\t' || str[i] == '\n') {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            count++;
        }
    }
    return count;
}

__declspec(dllexport) void to_uppercase(char* str) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= 'a' && str[i] <= 'z') {
            str[i] -= 32;
        }
    }
}

__declspec(dllexport) void sort_strings(char** strings, int n) {
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (strcmp(strings[i], strings[j]) > 0) {
                char* temp = strings[i];
                strings[i] = strings[j];
                strings[j] = temp;
            }
        }
    }
}

__declspec(dllexport) int read_file(const char* filename, char* buffer, int buffer_size) {
    FILE* file = fopen(filename, "r");
    if (!file) return 0;  // Error opening file

    int i = 0;
    char c;
    while ((c = fgetc(file)) != EOF && i < buffer_size - 1) {
        buffer[i++] = c;
    }
    buffer[i] = '\0';
    fclose(file);
    return 1;  // Success
}

__declspec(dllexport) unsigned int crc32(const unsigned char* data, size_t length) {
    unsigned int crc = 0xFFFFFFFF;
    for (size_t i = 0; i < length; i++) {
        unsigned char byte = data[i];
        crc ^= byte;
        for (int j = 0; j < 8; j++) {
            if (crc & 1) {
                crc = (crc >> 1) ^ 0xEDB88320;
            } else {
                crc >>= 1;
            }
        }
    }
    return ~crc;
}

__declspec(dllexport) int find_substring(const char* str, const char* substr) {
    const char* pos = strstr(str, substr);
    return pos ? (int)(pos - str) : -1;
}

__declspec(dllexport) int is_palindrome(const char* str) {
    int left = 0, right = strlen(str) - 1;
    while (left < right) {
        if (str[left++] != str[right--]) return 0;
    }
    return 1;  // True
}

__declspec(dllexport) void caesar_cipher(char* str, int shift) {
    for (int i = 0; str[i] != '\0'; i++) {
        if (str[i] >= 'a' && str[i] <= 'z') {
            str[i] = ((str[i] - 'a' + shift) % 26) + 'a';
        } else if (str[i] >= 'A' && str[i] <= 'Z') {
            str[i] = ((str[i] - 'A' + shift) % 26) + 'A';
        }
    }
}

__declspec(dllexport) int linear_search(const int* array, int size, int value) {
    for (int i = 0; i < size; i++) {
        if (array[i] == value) return i;
    }
    return -1;  // Not found
}

__declspec(dllexport) void rotate_array(int* array, int size, int k) {
    k %= size;
    int* temp = (int*)malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) temp[i] = array[i];
    for (int i = 0; i < size - k; i++) array[i] = array[i + k];
    for (int i = 0; i < k; i++) array[size - k + i] = temp[i];
    free(temp);
}

__declspec(dllexport) void generate_fibonacci(int* array, int n) {
    if (n < 1) return;
    array[0] = 0;
    if (n == 1) return;
    array[1] = 1;
    for (int i = 2; i < n; i++) {
        array[i] = array[i - 1] + array[i - 2];
    }
}

__declspec(dllexport) int count_lines(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return -1;  // Error
    int count = 0;
    char c;
    while ((c = fgetc(file)) != EOF) {
        if (c == '\n') count++;
    }
    fclose(file);
    return count;
}





