/*
*	Author: Lukas Gosch
*	Date: 05.05.2018
*	Description:
*		Implementation of a unblocked right-looking LU factorization 
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

//C++ does not allow forward declaration of enums, therefore they are defined here
enum CBLAS_ORDER 	{CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE 	{CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};
extern "C" void cblas_daxpy(const int __N, const double __alpha, const double *__X, const int __incX, double *__Y, const int __incY);
extern "C" void cblas_dgemm(const enum CBLAS_ORDER __Order, const enum CBLAS_TRANSPOSE __TransA, const enum CBLAS_TRANSPOSE __TransB, const int __M, const int __N, const int __K, const double __alpha, const double *__A, const int __lda, const double *__B, const int __ldb, const double __beta, double *__C, const int __ldc);

struct solution{
	vector<vector<double>> A; //solution Matrix
	vector<int> p; //permutation vector
	double time; //time in seconds used for calculating solution vector
	int error; //0 successfull solution calculation, 1 error in calculating solution vector (integrity of struct not given)
};

vector<vector<double>> createMatrix(int);
solution LUfact(vector<vector<double>>, int);

int main(){
	//Configuration 
	int limit {5000}; //stops at given problem size
	int start_n {100}; //starting problem size
	int stepsize {100}; //problem size increase each step
	double flops {27.2 * 1e9}; //theoretical peak performance per second one processor core achieves for double precision

	//Create Log-Files
	ofstream file_log;
	file_log.open("performance_lupp.txt");
	//n ... Problem size; time ... processor time;
	//relp ... achieved relative performance (compared to peak performance)
	//rfacte ... relative factorization error
	file_log<< "n,time,relp,rfacte\n";
	
	//LU-Factorization + Diagnostics for different Problem Sizes
	solution s;
	int n {start_n};
	while(n <= limit){
		//Generate test data for problem size n (using mersenne_twister_engine)
		vector<vector<double>> A = createMatrix(n);

		//calculate solution (LU-Factorization) for given problem size
		s = LUfact(A, n);
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
				colsum[j] += abs(A[i][j]);
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
					L[i*n + j] = s.A[i][j];
				}else{
					L[i*n + j] = 0;
				}

			}
		}

		double *U = new double[n*n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				if(i <= j){
					U[i*n + j] = s.A[i][j];
				}else{
					U[i*n + j] = 0;
				}
			}
		}

		//Calc of PA in O(n)
		vector<vector<double>> PA(n);
		for(int i = 0; i < n; i++){
			PA[i].swap(A[s.p[i]]);
		}

		double *R = new double[n*n];
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				R[i*n + j] = PA[i][j];
			}
		}
		//Calculate: LU - PA
		cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1, L, n, U, n, -1, R, n);
		delete [] L;
		delete [] U;

		double res_norm {0};
		vector<double> res_colsum(n, 0);

		//Calculate Column Sum for LU-PA
		for(int i = 0; i < n; i++){
			for(int j = 0; j < n; j++){
				res_colsum[j] += abs(R[i*n + j]);
			}
		}
		delete [] R;

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
*	Generates a matrix of dimensions nxn in row major order. 
*	Fills it with data pulled from the interval [0, 1] and random sign using a uniform 
* 	real distribution and the mersenne_twister_engine for random number generation.
*	Diagonal elements choosen to be very small to highlight the effect of pivoting.
*	n ... number of rows and columns
**/
vector<vector<double>> createMatrix(int n){
	mt19937_64 gen;
	gen.seed(100);
	uniform_real_distribution<double> dis(0, 1);

	vector<vector<double>> A(n, vector<double>(n));
	int sign = 1;
	//Creates Row Major Order Matrix
	//fill column
	for(int i = 0; i < n; i++){
		//fill row
		for(int j = 0; j < n; j++){
			if(dis(gen) > 0.5) sign = -1;
			else sign = 1;
			if(i == j){
				A[i][j] = sign * dis(gen) * 1e-14;
			}else{
				A[i][j] = sign * dis(gen);
			}
		}
	}

	return A;
}

/**
* Calculates the LU Factorization with partial pivoting of a given square matrix A stored in
* row-major order using CBLAS.
* n... dimensions of A
**/
solution LUfact(vector<vector<double>> A, int n){
	solution s;

	//Start Timer
	clock_t c_start = clock();

	//build permutation vector
	vector<int> p(n, 0);
	for(int i = 1; i < n; i++){
		p[i] = i;
	}

	//Unblocked LU-Factorization Algorithm with pivoting implemented for a Matrix stored in row major order
	for(int k = 0; k < n; k++){
		//Find max value and index of A(k:n, k)
		double max_v = abs(A[k][k]);
		int max_i = k;
		for(int j = k+1; j < n; j++){
			if(abs(A[j][k]) > max_v){
				max_v = abs(A[j][k]);
				max_i = j;
			}
		}

		//Swap rows in A in constant time! (swap swaps pointers, not element-wise swaps)
		//Swap elements in p-vector
		if(k != max_i){
			swap(A[k], A[max_i]);
			swap(p[k], p[max_i]);
		}

		//Matrix not regular
		if(A[k][k] == 0){
			s.error = 1;
			return s;
		}

		//Standard LU-Fact.
		for(int i = k+1; i < n; i++){
			A[i][k] = A[i][k] / A[k][k];

			//A(i,k+1:n) = A(i,k+1:n) - A(i,k)*A(k,k+1:n)
			cblas_daxpy(n-k-1, -A[i][k], &A[k][k+1], 1, &A[i][k+1], 1);
		}
	}

	//End Timer
	clock_t c_end = clock();
	
	//Calculate CPU Time in Sec
	double t_int = c_end - c_start;
	double clocks = CLOCKS_PER_SEC;
	double time = t_int / clocks;

	//Build solution struct
	s.A = A;
	s.p = p;
	s.time = time;
	s.error = 0;
	
	return s;
}