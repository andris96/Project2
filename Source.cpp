#include <fstream>
#include <cmath>
#include <string>
#include <armadillo>
#include <iomanip>
#include "time.h"

using namespace std;
using namespace arma;

/* 
This is a function for defining a tridiagonal matrix, where a is the diagonal elements,
b is the element directly below the diagonal, and c is directly above the diagonal.
 
*/
mat tridi_matrix(double a, double b, double c, int n) {

		mat A = mat(n, n).zeros();

		for (int i = 0; i < n; i++) {
			A(i, i) = a;

			if (i > 0) {
				A(i, i - 1) = b;
			}

			if (i < n - 1) {
				A(i, i + 1) = c;
			}
		}
		return A;
	}



double maxoffdiag(mat A, int * k, int * l, int n) {
	double max = 0.0;
	//we only iterate over upper diagonal elements, since the matrix is symmetric
	for (int i = 0; i < n; i++) {
			for (int j = i + 1; j < n; j++) {
				if (abs(A(i, j)) > max) {	
					//when we find a value that is greater than max, we set the new max to that value
					max = fabs(A(i, j));
					*k = i; //need to make note of where in the matrix the element is located
					*l = j;
				}
			}
		}
	return max;
	}	

void jacobi_rotate(mat& A, mat& R, int k, int l, int n)
{
	double s, c;
//if A(k,l) != 0, we need to decide c and s such that we can set it to zero
if (A(k, l) != 0.0) {
	double t, tau;
	
	tau = (A(l, l) - A(k, k)) / (2 * A(k, l));
	
	if (tau >= 0) {
		t = 1.0 / (tau + sqrt(1.0 + tau * tau)); 
	}
	else {
		t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
	}
	c = 1 / sqrt(1 + t * t);
	s = c * t;
	}
		//if A(k,l) is already 0, then we know that sine is 0 and cosine is 1
	else {
	c = 1.0;
	s = 0.0;
	}
	double a_kk, a_ll, a_ik, a_il, r_ik, r_il;
	//these values will be changed during the iteration, so we need to save them to avoid using the wrong value
	a_kk = A(k, k); 
	a_ll = A(l, l);

	A(k, k) = c * c * a_kk - 2.0 * c * s * A(k, l) + s * s * a_ll;
	A(l, l) = s * s * a_kk + 2.0 * c * s * A(k, l) + c * c * a_ll;
	A(k, l) = 0.0; 
	A(l, k) = 0.0; 

	for (int i = 0; i < n; i++) {
		if (i != k && i != l) {
			a_ik = A(i, k);
			a_il = A(i, l);
			A(i, k) = c * a_ik - s * a_il;
			A(k, i) = A(i, k);
			A(i, l) = c * a_il + s * a_ik;
			A(l, i) = A(i, l);
		}
			
			r_ik = R(i, k);
			r_il = R(i, l);
			R(i, k) = c * r_ik - s * r_il;
			R(i, l) = c * r_il + s * r_ik;
		}
		return;
	} // end of function jacobi_rotate

void jacobi_method(mat& A, mat& R, int n) {
	{
		int k, l;
		double epsilon = 1.0e-10;
		double max_number_iterations = (double)n * (double)n * (double)n;
		int iterations = 0;
		double max_offdiag = maxoffdiag(A, &k, &l, n);

		while (fabs(max_offdiag) > epsilon && (double)iterations < max_number_iterations) {
			max_offdiag = maxoffdiag(A, &k, &l, n);
			jacobi_rotate(A, R, k, l, n);
			iterations++;
		}
		cout << "Number of iterations: " << iterations << endl;
		return;
	}
}



void maxoffdiag_test() {
	//defining a random 10 by 10 matrix with only 0's in the diagonal and random elements elsewhere
	mat random = randu<mat>(10, 10);
	mat random_symmetric = symmatu(random); //need to define a symmetrical matrix, since maxoffdiag only check the upper triangle
	int k, l;
	for (int i = 0; i < 10; i++) {
		random_symmetric(i, i) = 0.0;	//making sure no elements in the diagonal is larger than nondiagonal ones
	}

	double expected_value = random_symmetric.max(); //finding max value with the armadillo .max() function

	if (maxoffdiag(random_symmetric, &k, &l, 10) == expected_value) { //comparing the values
		cout << "maxoffdiag is working as expected" << endl;
	}
	else {
		cout << "maxoffdiag returns wrong value" << endl;
	}

}

//function to normalise every column vector in a matrix
void normal_vectors(mat& A,int n) {
	for (int i = 0; i < n; i++) {
		A.col(i) = normalise(A.col(i), 1);
	}
}

//function to obtain the smallest eigenvalue
int obtain_eigval_min(mat A,int n) {
	double eigval_min = A(0, 0);
	int i_number{};
	for (int i = 0; i < n; i++) {
		if (fabs(eigval_min) > fabs(A(i, i))) {
			eigval_min = A(i, i);
			i_number = i;
		}
	}
	return i_number; //returns i, which can be used to find the eigen value and corresponding eigenvector
}



/*
We know that two vectors are parallel if a_i = c*b_i where c is some constant.
This test is checking that for every element in a and b the constant is the same +- epsilon
if the constant c varies by more than epsilon, the test will return a message, saying that they are not parallel.
*/
void compare_eigenvector_test(vec a, vec b) {
	double constant;
	int n = a.size();
	double epsilon = 1e-4;
	constant = a(0) / b(0);
	for (int i = 1; i < n; i++) {
		if (constant > a(i) / b(i) + epsilon && constant < a(i)/b(i) - epsilon ) {
			cout << constant << endl;
			cout << a(i) / b(i) << endl;
			cout << "the vectors are not parallel" << endl;
			break;
		}
	}
}


int main(int argc, char* argv[]){
	int n = 100; 
	double h = 1 / (double)n;
	
	double a = -1 / (pow(h, 2));
	double b = 2 / (pow(h,2));


	mat mat_A = tridi_matrix(b, a, a, n);	//making tridiagonal matrix
	mat mat_eig_test = mat_A;
	
	

	mat mat_R = eye(n, n);	//making identity matrix

	jacobi_method(mat_A, mat_R,n);


	//normal_vectors(mat_R, n);
	//cout << mat_R << endl;

	int i_number = obtain_eigval_min(mat_A, n); //obtaining the position of the smallest eigenvalue

	vec eigvec_min = mat_R.col(i_number); //eigenvector corresponding til smallest eigenvalue

	//saving the elements of the vector for plotting
	ofstream myfile;
	myfile.open("Eigenvector.txt");
	myfile << eigvec_min;
	myfile.close();


	//cout << mat_A << endl;



	//cout << mat_R << endl;

	vec eigval;
	mat eigvec;

	eig_sym(eigval, eigvec, mat_eig_test);

	//cout << eigval << endl;

	

	compare_eigenvector_test(eigvec.col(0), eigvec_min); //if this test fails a message will come up
	
	maxoffdiag_test();

	double rho_max = 100;
	double h_2 = rho_max / n;

	double d = -1 / (pow(h_2, 2));

	//defining matrix for single electron problem
	mat mat_d = tridi_matrix(0,d,d,n);

	for (int i = 0; i < n; i++) {
		mat_d(i, i) = 2 / (pow(h_2, 2)) + i * h_2;
	}

	mat mat_R2 = eye(n, n);
	
	jacobi_method(mat_d, mat_R2, n);

	int i_number_2 = obtain_eigval_min(mat_d, n); //obtaining the position of the smallest eigenvalue

	cout << i_number_2 << endl;

	vec eigvec_min_2 = mat_R2.col(i_number_2); //eigenvector corresponding til smallest eigenvalue

	cout << mat_d(i_number_2, i_number_2) << endl;


	return 0;
}

