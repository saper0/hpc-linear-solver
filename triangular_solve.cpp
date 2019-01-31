/*
*	Author: Lukas Gosch
*	Date: 30.04.2018
*	Description: 
*		Implementation of a forward and backwards substitution 
*		algorithm.
*		
*		Implementation of a random lower or upper triangular matrix
*		generator for experimental evaluation.
*/

#include <iostream>
#include <ctime>
#include <random>
#include <array>
#include <fstream>
#include <cmath>
#include <vector>

using namespace std;

struct solution{
	vector<double> x; //solution vector
	double time; //time in seconds used for calculating solution vector
	int error; //0 successfull solution calculation, 1 error in calculating solution vector (integrity of struct not given)
};

vector<vector<double>> createTriangularMatrix(int, bool = false, bool = false);
solution backSubstitution(vector<vector<double>>, vector<double>, int);
solution forwardSubstitution(vector<vector<double>>, vector<double>, int);

int main(){
	//Configuration 
	int limit {15000}; //stops at given problem size
	int start_n {100}; //starting problem size
	int stepsize {100}; //problem size increase each step
	double flops {27.2 * 1e9}; //theoretical peak performance per second one processor core achieves for double precision

	//Create Log-Files
	ofstream file_back;
	ofstream file_forw;
	file_back.open("performance_back.txt");
	file_forw.open("performance_forw.txt");
	//n ... Problem size; time ... processor time;
	//relp ... achieved relative performance (compared to peak performance)
	//rfe .. relative forward error; rrn ... relative residual norm
	file_back << "n,time,relp,rfe,rrn\n";
	file_forw << "n,time,relp,rfe,rrn\n";

	//Forward/Back-Substitution + Diagnostics for different Problem Sizes
	solution s;
	int n {start_n};
	bool lower {true};
	bool condition {true};
	while(condition){
		//Generate test data for problem size n (using mersenne_twister_engine)
		vector<vector<double>> T = createTriangularMatrix(n, lower? lower:false, true);
		vector<double> x(n, 1);
		vector<double> b(n, 0);

		//Calc b
		for(int j = 0; j < n; j++){
			if(lower){
				for(int i = j; i < n; i++){
					b[i] += T[j][i]*x[j];
				}
			}else{
				for(int i = 0; i <= j; i++){
					b[i] += T[j][i]*x[j];
				}
			}
		}

		//calculate solution for given problem size
		s = lower? forwardSubstitution(T, b, n) : backSubstitution(T, b, n);
		if(s.error == 1){
			cerr << "Error: Diagonal entry of triangular matrix was zero!" << endl;
			return 1;
		}

		//Efficiency
		int flops_needed {n * n}; //O(n^2)
		double theoretical_flops {flops * s.time};
		double relp {flops_needed / theoretical_flops};
		
		//Relative forward error
		double rfe {0};
		double s_norm {0};
		for(int i = 0; i < n; i++){
			rfe += abs(s.x[i] - x[i]);
			s_norm += x[i] * x[i];
		}
		rfe = rfe / s_norm;


		//Relative residual norm
		double rrn {0};
		double b_norm {0};
		for(int i = 0; i < n; i++){
			b_norm = abs(b[i]);
		}

		vector<double> b_s(n, 0);
		for(int j = 0; j < n; j++){
			if(lower){
				for(int i = j; i < n; i++){
					b_s[i] += T[j][i]*s.x[j];
				}
			}else{
				for(int i = 0; i <= j; i++){
					b_s[i] += T[j][i]*s.x[j];
				}
			}
		}

		for(int i = 0; i < n; i++){
			rrn += abs(b_s[i]-b[i]);
		}
		rrn = rrn / b_norm;

		cout << "N: " << n << " time: "<< s.time << " Rel.P.: " << relp << " RFE: " << rfe << " RRN: " << rrn << endl;
		//Write Log File
		if(lower){
			file_forw << n << "," << s.time << "," << relp << "," << rfe << "," << rrn << "\n";
		}else{
			file_back << n << "," << s.time << "," << relp << "," << rfe << "," << rrn << "\n";
		}

		n += stepsize;
		if(n > limit && lower){ //restart loop using backward substition
			lower = false; 
			n = start_n;
		}
		if(n > limit && !lower) condition = false;

	}

	file_back.close();
	file_forw.close();

	return 0;
}

/**
*	Generates a triangular matrix of dimensions nxn in row or column-major order. 
*	Fills it with data pulled from the interval [1.0, 1.5] with random sign using a uniform 
* 	real distribution and the mersenne_twister_engine for random number generation.
*	To increase the condition number, the diagonal entries are multiplied by a factor 20.
*	n ... number of rows and columns
*	lower (per default false) ... if true, a lower triangular matrix
*		will be created, otherwise a upper triangular matrix will be created
*	column (per default false) ... if true, a column-major order matrix will be
*		created, otherwise a row-major order matrix will be created
**/
vector<vector<double>> createTriangularMatrix(int n, bool lower, bool column){
	mt19937_64 gen;
	gen.seed(100);
	uniform_real_distribution<double> dis(1, 1.5);

	vector<vector<double>> T(n, vector<double>(n));
	int sign = 1;
	//Creates Row Major Order Matrix
	if(!column){
		//fill column
		for(int i = 0; i < n; i++){
			//fill row
			for(int j = 0; j < n; j++){
				if(dis(gen) > 1.25) sign = -1;
				else sign = 1;
				if(i == j){
					T[i][j] = sign * dis(gen) * 20;
				}else if(i > j){
					if(lower) T[i][j] = sign * dis(gen);
					else T[i][j] = 0;
				}else{
					if(lower) T[i][j] = 0;
					else T[i][j] = sign * dis(gen);
				}
			}
		}
	}else{ 
		//Creates Column Major Order Matrix
		//column loop
		for(int j = 0; j < n; j++){
			//row loop
			for(int i = 0; i < n; i++){
				if(dis(gen) > 1.25) sign = -1;
				else sign = 1;
				if(i == j){
					T[j][i] = sign * dis(gen) * 20;
				}else if(i > j){
					if(lower) T[j][i] = sign * dis(gen);
					else T[j][i] = 0;
				}else{
					if(lower) T[j][i] = 0;
					else T[j][i] = sign * dis(gen);
				}
			}
		}
	}

	return T;
}


/**
* Solves the linear system Ux=b for a given upper triangular matrix U 
* stored in column major order and a given vector b.
* n... dimension of U and number of elements in b
**/
solution backSubstitution(vector<vector<double>> U, vector<double> b, int n){
	solution s;
	vector<double> x(n);

	//Start Timer
	clock_t c_start = clock();

	//Back Substitution Algorithm implemented for a Matrix stored in column major order
	for(int j = n-1; j >= 0; j--){
		if(U[j][j] == 0){
			s.error = 1;
			return s;
		}

		x[j] = b[j] / U[j][j];
		for(int i = 0; i < j; i++){
			b[i] = b[i] - U[j][i] * x[j];
		}
	}

	//End Timer
	clock_t c_end = clock();
	
	//Calculate CPU Time in Sec
	double t_int = c_end - c_start;
	double clocks = CLOCKS_PER_SEC;
	double time = t_int / clocks;

	//Build solution struct
	s.x = x;
	s.time = time;
	s.error = 0;
	
	return s;
}

/**
* Solves the linear system Lx=b for a given lower triangular matrix L
* stored in column major order and a given vector b.
* n... dimension of L and number of elements in b
**/
solution forwardSubstitution(vector<vector<double>> L, vector<double> b, int n){
	solution s;
	vector<double> x(n);

	//Start Timer
	clock_t c_start = clock();

	//Forward Substitution Algorithm implemented for a Matrix stored in column major order
	for(int j = 0; j < n; j++){
		if(L[j][j] == 0){
			s.error = 1;
			return s;
		}

		x[j] = b[j] / L[j][j];
		for(int i = j+1; i < n; i++){
			b[i] = b[i] - L[j][i] * x[j];
		}
	}

	//End Timer
	clock_t c_end = clock();
	
	//Calculate CPU Time in Sec
	double t_int = c_end - c_start;
	double clocks = CLOCKS_PER_SEC;
	double time = t_int / clocks;

	//Build solution struct
	s.x = x;
	s.time = time;
	s.error = 0;
	
	return s;
}
