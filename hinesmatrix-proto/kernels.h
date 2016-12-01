//#define N 512000
//#define BlockInter 16
//#define BlockSize 32


struct HinesMatrix {
    double *a;
    double *b; 
    double *d; 
    double *rhs;
    int *p;
    int num_cells;
    int cell_size;
};

HinesMatrix loadMatrix(std::string filename);
void matrix_solve(HinesMatrix const& params,int Systems,int CudaBlockSize);
void allocateHMonDevice(HinesMatrix &src,HinesMatrix &dst, int size);
void copyBackToHost(HinesMatrix &src,HinesMatrix &dst, int size);

