/*
*	Author: Lukas Gosch
*	Date: 05.05.2018
*	Description:
*		Implementation of a blocked right-looking LU factorization 
*		with partial pivoting using CBLAS.
*
*		Implementation of a random matrix generator for experimental
*		evaluation.
*/

#include <iostream>
#include <ctime>
#include <random>
#include <fstream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

//BLAS LVL 1
extern "C" void cblas_dscal(const int __N, const double __alpha, double *__X, const int __incX);
extern "C" void cblas_dswap(const int __N, double *__X, const int __incX, double *__Y, const int __incY);
extern "C" int cblas_idamax(const int __N, const double *__X, const int __incX);
extern "C" void cblas_dcopy(const int __N, const double *__X, const int __incX, double *__Y, const int __incY);
//C++ does not allow forward declaration of enums
enum CBLAS_ORDER 	{CblasRowMajor=101, CblasColMajor=102};
extern "C" void cblas_dger(const enum CBLAS_ORDER __Order, const int __M, const int __N, const double __alpha, const double *__X, const int __incX, const double *__Y, const int __incY, double *__A, const int __lda);

//Blas LVL 2
extern "C" void cblas_daxpy(const int __N, const double __alpha, const double *__X, const int __incX, double *__Y, const int __incY);

//Blas LVL 3
enum CBLAS_TRANSPOSE 	{CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
enum CBLAS_UPLO		{CblasUpper=121, CblasLower=122};
enum CBLAS_DIAG		{CblasNonUnit=131, CblasUnit=132};
enum CBLAS_SIDE		{CblasLeft=141, CblasRight=142};
extern "C" void cblas_dtrsm(const enum CBLAS_ORDER __Order, const enum CBLAS_SIDE __Side, const enum CBLAS_UPLO __Uplo, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_DIAG __Diag, const int __M, const int __N, const double __alpha, const double *__A, const int __lda, double *__B, const int __ldb);
extern "C" void cblas_dgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N, const int __K, const double __alpha, const double *__A, const int __lda, const double *__B, const int __ldb, const double __beta, double *__C, const int __ldc);

struct solution{
	int* p; //permutation vector
	double time; //time in seconds used for calculating solution vector
	int error; //0 successfull solution calculation, 1 error in calculating solution vector (integrity of struct not given)
};

double* createMatrix(int);
solution LUfact(double*, int, int);

int main(){
	//Configuration 
	int limit {5000}; //stops at given problem size
	int start_n {100}; //starting problem size
	int stepsize {100}; //problem size increase each step
	double flops {27.2 * 1e9}; //theoretical peak performance per second one processor core achieves for double precision

	//Create Log-Files
	ofstream file_log;
	file_log.open("performance_lublocked.txt");
	//n ... Problem size; time ... processor time;
	//relp ... achieved relative performance (compared to peak performance)
	//rfacte ... relative factorization error
	file_log<< "n,time,relp,rfacte\n";
	
	//LU-Factorization + Diagnostics for different Problem Sizes
	solution s;
	int n {start_n};
	while(n <= limit){
		//Generate test data for problem size n (using mersenne_twister_engine)
		double *A = createMatrix(n);
		double *R = new double[n*n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				R[i*n + j] = A[j*n+i];
			}
		}

		//calculate solution (LU-Factorization) for given problem size
		s = LUfact(A, n, 40);
		if(s.error == 1){
			cerr << "Matrix does not have an LU Factorization!" << endl;
			return 1;
		}

		//Efficiency
		double flops_needed {2.0 * n * n * n / 3}; //O(2/3 * n^3)
		double theoretical_flops {flops * s.time};
		double relp {flops_needed / theoretical_flops};

		//Relative Factorization Error (rfacte)
		double A_norm {0};
		double rfacte {0};
		vector<double> colsum(n, 0);

		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				colsum[j] += abs(R[i*n+j]);
			}
		}

		A_norm = *max_element(colsum.begin(), colsum.end());

		//Extract L & U Matrix from computed LU Factorization and
		//prepare them for usage in the cblas routine
		double *L = new double[n*n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				if(i == j){
					L[i*n + j] = 1;
				}else if(i > j){
					L[i*n + j] = A[j*n+i];
				}else{
					L[i*n + j] = 0;
				}

			}
		}

		double *U = new double[n*n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				if(i <= j){
					U[i*n + j] = A[j*n+i];
				}else{
					U[i*n + j] = 0;
				}
			}
		}

		//Calc of PA
		double *PA = new double[n*n];
		for(int i = 0; i < n; i++){
			cblas_dcopy(n, &R[s.p[i]*n], 1, &PA[i*n], 1);
		}

		//Calculate: LU - PA
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, L, n, U, n, -1, PA, n);
		delete [] L;
		delete [] U;

		double res_norm {0};
		vector<double> res_colsum(n, 0);

		//Calculate Column Sum for LU-PA
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				res_colsum[j] += abs(PA[i*n + j]);
			}
		}
		delete [] R;
		delete [] A;

		//Choose biggest column sum for LU-PA => 1 Norm of LU-PA
		res_norm = *max_element(res_colsum.begin(), res_colsum.end());
		rfacte = res_norm / A_norm;

		cout << "N: " << n << " time: "<< s.time << " Rel.P.: " << relp << " Rfacte:" << rfacte << endl;
		
		//Write Log File
		file_log << n << "," << s.time << "," << relp << "," << rfacte << "\n"; 

		n += stepsize;
	}

	
	file_log.close();
	
	return 0;
}

/**
*	Generates a matrix of dimensions nxn in column major order. 
*	Fills it with data pulled from the interval [0, 1] and random sign using a uniform 
* 	real distribution and the mersenne_twister_engine for random number generation.
*	Diagonal elements choosen to be very small by the same factor as in part three
*	to compare performance with part three.
*	n ... number of rows and columns
**/
double* createMatrix(int n){
	mt19937_64 gen;
	gen.seed(100);
	uniform_real_distribution<double> dis(1, 1.5);

	double* A = new double[n*n];
	int sign = 1;
	//Creates Column Major Order Matrix
	//fill row
	for(int j = 0; j < n; j++){
		//fill column
		for(int i = 0; i < n; i++){
			if(dis(gen) > 1.25) sign = -1;
			else sign = 1;
			if(i == j){
				A[j*n + i] = sign * dis(gen) * 1e-14;
			}else{
				A[j*n + i] = sign * dis(gen);
			}
		}
	}

	return A;
}

/**
* Calculates the blocked LU Factorization with partial pivoting of a given square matrix A stored in
* column-major order using CBLAS.
* n... dimensions of A
**/
solution LUfact(double* A, int n, int b){
	solution s;

	//Start Timer
	clock_t c_start = clock();

	//built permutation vector
	int* p = new int[n];
	for(int i = 0; i < n; i++){
		p[i] = i;
	}

	//Blocked LU-Factorization Algorithm implemented for a Matrix stored in row major order
	int size = 0;
	double invElement = 0;
	int diag = 0;
	int x = 0;
	int y = 0;
	int k = 0;

	for(; k< n-(n%b); k+=b){
		for(int i = k; i<k+b; i++){
			diag = i*n + i;
			x = diag + 1;
			y = diag + n;

			//Find index of max value A(k:n, k)
			int max_i = cblas_idamax(n-i, &A[diag], 1);

			//Swap rows in A & p
			if(max_i != 0){
				cblas_dswap(n, &A[i], n, &A[i+max_i], n);
				swap(p[i], p[i+max_i]);
			}

			if(A[diag] == 0){
				s.error = 1;
				return s;
			}

			//Dimension of Submatrix
			size = n-i-1;

			//Update row elements
			invElement = 1 / A[diag];
			cblas_dscal(size, invElement, &A[x], 1);
			
			//Rank-1 Update to Submatrix
			cblas_dger(CblasColMajor, size, b-i+k-1, -1, &A[x], 1, &A[y], n, &A[y+1], n);
		}
		

		//Solve L11*A12_new = A12
		cblas_dtrsm(CblasColMajor, CblasLeft, CblasLower, CblasNoTrans, CblasUnit, b, n-b-k, 1, &A[k*n+k], n, &A[(k+b)*n+k], n);

		//A22_new = A22 - L21*A12 
		cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, n-b-k, n-b-k, b, -1, &A[(k*n)+k+b], n, &A[(k+b)*n+k], n, 1, &A[(k+b)*n+k+b], n);
	}
	
	if(k < n){
		for(int i = k; i<n; i++){
			diag = i*n + i;
			x = diag + 1;
			y = diag + n;

			//Find index of max value A(k:n, k)
			int max_i = cblas_idamax(n-i, &A[diag], 1);

			//Swap rows in A & p
			if(max_i != 0){
				cblas_dswap(n, &A[i], n, &A[i+max_i], n);
				swap(p[i], p[i+max_i]);
			}

			if(A[diag] == 0){
				s.error = 1;
				return s;
			}

			//Dimension of Submatrix
			size = n-i-1;

			//Update row elements
			invElement = 1 / A[diag];
			cblas_dscal(size, invElement, &A[x], 1);
			
			//Rank-1 Update to Submatrix
			cblas_dger(CblasColMajor, size, size, -1, &A[x], 1, &A[y], n, &A[y+1], n);
		}
		
	}

	//End Timer
	clock_t c_end = clock();
	
	//Calculate CPU Time in Sec
	double t_int = c_end - c_start;
	double clocks = CLOCKS_PER_SEC;
	double time = t_int / clocks;

	//Build solution struct
	s.p = p;
	s.time = time;
	s.error = 0;
	
	return s;
}