#include "HinesMatrix.hpp"
#include <omp.h>
#include <iostream>

int main(int argc, char const *argv[])
{
	HinesMatrix hs;
	hs = loadMatrix("matrix_1_cell.json");

	/*
		1- All our cells have the same structure defined by hs
		2- Number of cells must be power of 2

	*/

	
	////////////////////// 1 EXECUTION TEST //////////////////////////

	/*
		HinesMatrix: Defines the structure of the cell 
		Number of cells (int power of 2): Defines the number of cells to be processed
		Threads: Defines the number of threads to be launched

	*/
	matrix_solve(hs,1024,32);

	/////////////////////////////////////////////////
	return 0;
}


