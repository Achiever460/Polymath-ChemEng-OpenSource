#include<iostream>
#include<bits/stdc++.h>
#include<fstream>
#include<chrono>
#include<Eigen/Dense>
#include<Eigen/Sparse>
#include<Eigen/SparseCholesky>
#include<unsupported/Eigen/IterativeSolvers>
using namespace std;
using namespace Eigen;


double CalculateDeterminant(const MatrixXd& matrix) {
    double determinant = matrix.determinant();
    return determinant;
}


int GetMatrixRank(const MatrixXd& matrix) {
    FullPivLU<MatrixXd> lu_decomp(matrix);
    return lu_decomp.rank();
}


bool isSymmetric(const MatrixXd& matrix) {
    // Check if the matrix is square
    if (matrix.rows() != matrix.cols()) {
        return false;  // A non-square matrix cannot be symmetric
    }

    // Check if the matrix is symmetric
    return matrix.isApprox(matrix.transpose());
}


bool isPositiveDefinite(const MatrixXd &matrix) {
    // Check if the matrix is square
    if (matrix.rows() != matrix.cols()) {
        return false;
    }

    // Attempt Cholesky decomposition
    LLT<MatrixXd> llt(matrix);
    if (llt.info() != Eigen::Success) {
        return false; // Cholesky decomposition failed, not positive definite
    }

    // Check the determinants of principal minors (Sylvester's Criterion)
    for (int i = 1; i <= matrix.rows(); i++) {
        MatrixXd minor = matrix.topLeftCorner(i, i);
        if (minor.determinant() <= 0) {
            return false; // At least one principal minor is not positive definite
        }
    }

    return true; // Matrix is positive definite
}

// Different methods.
// 1. PCG
VectorXd pcgSolver(const MatrixXd& A, const VectorXd& b, const MatrixXd& M, double tol, int maxIter) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x;
    VectorXd z = M.triangularView<Lower>().solve(r);  // Preconditioning step
    VectorXd p = z;
    double rho = r.dot(z);
    double rho_prev = rho;

    for (int k = 0; k < maxIter; ++k) {
        VectorXd Ap = A * p;
        double alpha = rho / p.dot(Ap);
        x += alpha * p;
        r -= alpha * Ap;
        z = M.triangularView<Lower>().solve(r);
        rho_prev = rho;
        rho = r.dot(z);
        if (std::sqrt(rho) < tol)
            break;
        double beta = rho / rho_prev;
        p = z + beta * p;
    }

    return x;
}

// 2.MINRES
VectorXd minresSolver(const MatrixXd& A, const VectorXd& b, double tol, int maxIter) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x;
    VectorXd p = r;
    double alpha = r.squaredNorm() / (p.transpose() * A * p);
    VectorXd s = r - alpha * A * p;
    VectorXd v = s;
    VectorXd w;
    double beta;
    double rho;
    double rho_prev = r.norm();

    for (int k = 0; k < maxIter; ++k) {
        w = A * v;
        alpha = s.squaredNorm() / (w.transpose() * v);
        x += alpha * v;
        r = s - alpha * w;
        rho = r.norm();
        //cout << "Iteration " << k << ", Residual Norm: " << rho << endl;
        if (std::sqrt(rho) < tol )
            break;
        beta = rho / rho_prev;
        p = r + beta * (p - w);
        s = r;
        double scaling_factor = s.squaredNorm() / (p.transpose() * A * p);
        v = p + scaling_factor * v;
        rho_prev = rho;
    }

    return x;
}


// 3.SYMM LQ
VectorXd symmlqSolver(const MatrixXd& A, const VectorXd& b, double tol, int maxIter) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x;
    VectorXd u = r;
    VectorXd v = u;
    VectorXd w;
    VectorXd p = VectorXd::Zero(n);
    double alpha = r.dot(u);
    double beta;
    double gamma = alpha;

    for (int k = 0; k < maxIter; ++k) {
        w = A * v;
        alpha = r.dot(w);
        w -= (alpha / gamma) * u;
        v -= (alpha / gamma) * p;
        p = w - (w.dot(u) / gamma) * u;
        x += (alpha / gamma) * v;

        r = b - A * x;

        if (r.norm() < tol)
            break;

        beta = r.dot(w);
        gamma = alpha;
        alpha = beta - beta * beta / gamma;
        u = r - (beta / gamma) * u;
    }

    return x;
}

// 4. GMRES
VectorXd gmresSolver(const MatrixXd& A, const VectorXd& b, double tol,int maxIter) {
    GMRES<MatrixXd> solver;
    solver.setMaxIterations(maxIter);
    solver.setTolerance(tol);
    solver.compute(A);

    if (solver.info() != Success) {
        std::cerr << "Decomposition failed" << std::endl;
        return VectorXd();
    }

    VectorXd x = solver.solve(b);

    if (solver.info() != Success) {
        std::cerr << "Solving failed" << std::endl;
        return VectorXd();
    }

    return x;
}

// 5. QMR
/*VectorXd qmrSolver(const MatrixXd& A, const VectorXd& b, double tol, int maxIter) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x;
    VectorXd u = r;
    VectorXd p = u;
    VectorXd q = A * p;
    VectorXd v = q;
    double alpha;
    double beta;
    double rho;
    double omega = 1.0;

    for (int k = 0; k < maxIter; ++k) {
        rho = r.dot(u);
        alpha = rho / v.dot(q);
        x += alpha * p;
        r -= alpha * q;
        u = r;
        rho = r.dot(u);
        if (std::sqrt(rho) < tol)
            break;
        p = u + (rho / rho) * p;
        q = A * p;
        beta = q.dot(r) / rho;
        p = u - beta * p;
        v = q;
        x += omega * p;
        r -= omega * q;
        u = r;
        rho = r.dot(u);
        if (std::sqrt(rho) < tol)
            break;
        alpha = rho / v.dot(q);
        p = u + alpha * p;
        q = A * p;
        beta = q.dot(r) / rho;
        p = u - beta * p;
        v = q;
    }
    return x;
}*/

// 6.BICG
VectorXd bicgSolver(const MatrixXd& A, const VectorXd& b, double tol, int maxIter) {
    int n = A.rows();
    VectorXd x = VectorXd::Zero(n);
    VectorXd r = b - A * x;
    VectorXd rTilde = r;
    VectorXd p = r;
    VectorXd pTilde = rTilde;
    VectorXd v = A * p;
    VectorXd vTilde = A.transpose() * pTilde;
    double alpha;
    double beta;
    double rho;
    double rhoTilde;
    double omega;

    for (int k = 0; k < maxIter; ++k) {
        rho = r.dot(rTilde);
        if (std::abs(rho) < tol)
            break;

        alpha = rho / v.dot(rTilde);
        x += alpha * p;
        r -= alpha * v;
        rTilde -= alpha * vTilde;

        rhoTilde = r.dot(rTilde);
        beta = rhoTilde / rho;

        p = r + beta * p;
        pTilde = rTilde + beta * pTilde;

        v = A * p;
        vTilde = A.transpose() * pTilde;

        omega = rTilde.dot(v) / v.dot(v);
        x += omega * pTilde;
        r -= omega * v;
        rTilde -= omega * vTilde;
    }

    return x;
}

// Print function
void print(const VectorXd& x) {
    for (int i = 1; i <= x.size(); ++i) {
        std::cout << "x" << i << " = " << x(i-1) << std::endl;
    }
    cout<<endl;
}

// solver function
void solve(MatrixXd& A, MatrixXd& augmented_A, VectorXd& b) {
    // Get the current time before the operation
    auto start = std::chrono::high_resolution_clock::now();

    bool symmetric = isSymmetric(A);
    int n=A.rows();
    if (symmetric) {
        std::cout << "The matrix is symmetric." << std::endl;
        bool check = isPositiveDefinite(A);
        if(check==true){
           std::cout << "The matrix is Positive Definite." << std::endl;
           //Identity Preconditioner:
           MatrixXd M = MatrixXd::Identity(n, n);   
           // Incomplete Cholesky (IC) or Incomplete LU (ILU) Preconditioner:
           /*SimplicialLLT<MatrixXd> solver;
           solver.compute(A);
           MatrixXd M = solver.matrixL();*/
           //Diagonal Preconditioner:
           //MatrixXd M = A.diagonal().asDiagonal();
           double tol=1e-3;
           int maxIter=1e3;
           VectorXd x=pcgSolver(A,b,M,tol,maxIter);   //PCG Solver
           print(x);
        }else{
           std::cout << "The matrix is not Positive Definite." << std::endl; 
           double tol=1e-3;
           int maxIter=1e3;
           //VectorXd x=minresSolver(A,b,tol,maxIter); //MINRES
           //VectorXd x=symmlqSolver(A,b,tol,maxIter); //SYMMLQ
           VectorXd x=gmresSolver(A,b,tol,maxIter);    //GMRES
           //VectorXd x=bicgSolver(A,b,tol,maxIter);     //BICG
           print(x);
        }
    } else {
        std::cout << "The matrix is not symmetric." << std::endl;
        double tol=1e-3;
        int maxIter=1e3;
        VectorXd x=gmresSolver(A,b,tol,maxIter);    //GMRES
        //VectorXd x=qmrSolver(A,b,tol,maxIter);      //QMR
        //VectorXd x=bicgSolver(A,b,tol,maxIter);     //BICG
        print(x);
    }

    // Get the current time after the operation
    auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;
}

void printmatrix(const MatrixXd &A){
    int n=A.rows();
    int m=A.cols();
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            cout<<A(i,j)<<" ";
        }
        cout<<endl;
    }
    cout<<endl;
}

int main() {
    // Get the current time before the operation
    //auto start = std::chrono::high_resolution_clock::now();

    //Manually taken the input
    /*
    int eqn, var;

    std::cout << "Enter the number of equations: ";
    std::cin >> eqn;
    std::cout << "Enter the number of variables: ";
    std::cin >> var;

    MatrixXd A(eqn, var);
    VectorXd b(eqn);

    std::cout << "Enter the values of the coefficient matrix:" << std::endl;

    for (int i = 0; i < eqn; i++) {
        //std::cout << "Enter the coeff of equation " << (i + 1) <<" : ";
        for (int j = 0; j < var; j++) {
            double value;
            std::cin >> value;
            A(i, j) = value;
        }
    }

    std::cout << "Enter the values of the vector b:" << std::endl;

    for (int i = 0; i < eqn; i++) {
        //std::cout << "b(" << i + 1 << "): ";
        std::cin >> b(i);
    }
    */

    //Take input from the file.
    

    // Open the file for reading
    ifstream inFile("system(5).txt");

    if (!inFile) {
        cerr << "Error opening the file." << endl;
        return 1;
    }

    cout << "Reading data from the file..." << endl;
    int eqn, var;

    //cout << "Enter the number of equations: ";
    inFile >> eqn;
    //cout << "Enter the number of variables: ";
    inFile >> var;

    MatrixXd A(eqn, var);
    VectorXd b(eqn);
    // Read the values of the coefficient matrix A from the file
    for (int i = 0; i < eqn; i++) {
        for (int j = 0; j < var; j++) {
            double value;
            inFile >> value;
            A(i, j) = value;
        }
    }

    // Read the values of vector b from the file
    for (int i = 0; i < eqn; i++) {
        double value;
        inFile >> value;
        b(i) = value;
    }

    // Close the file when you're done
    inFile.close();

    //print the coefficient matrix 
    //printmatrix(A);
    //print the vector b
    //print(b);

    MatrixXd augmented_A(eqn, var + 1);
    augmented_A << A, b;

    // Calculate the ranks
    int rank_A = A.fullPivLu().rank();
    int rank_augmented_A = augmented_A.fullPivLu().rank();

    std::cout << "Rank of the matrix: " << rank_A << std::endl;
    std::cout << "Rank of the Augmented matrix: " << rank_augmented_A << std::endl;
    
    // Calculate the Determinant
    double determinant = CalculateDeterminant(A);

    //std::cout << "Determinant of the matrix: " << determinant << std::endl;

    // Consistency check
    if(eqn<var){
        cout<<"Inconsistent because the coefficient matrix is rectangular matrix"<<endl;
    }
    else{
        if(rank_A==rank_augmented_A){
           if(rank_A==var){
               if(determinant!=0){
                  cout<<"Consistent, unique solution exists"<<endl;
                  solve(A,augmented_A,b);
               }else{
                 cout<<"Inconsistent because the determinant of the coefficient matrix is equal to zero."<<endl;
               }
           }else{
               cout<<"Inconsistent because the rank of the coefficient matrix is not equal to the number of variables"<<endl;
           }
        }else{
           cout<<"Inconsistent because the rank of the coefficient matrix is not equal to rank of the Augmented matrix"<<endl; 
        }
    }
    
    // Get the current time after the operation
    //auto end = std::chrono::high_resolution_clock::now();

    // Calculate the duration in milliseconds
    //auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    //std::cout << "Time taken: " << duration.count() << " milliseconds" << std::endl;

    return 0;
}