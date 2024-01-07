#include <iostream>
#include <Eigen/Dense>

using namespace Eigen;
using namespace std;

int main() {
    int eqn;
    cout<<"No. of equations: ";
    cin>>eqn;
    int var;
    cout<<"No. of variables: ";
    cin>>var;

    MatrixXd A(eqn, var);
    VectorXd b(eqn);

    // Fill the coefficient matrix A with random values
    A = MatrixXd::Random(eqn, var);
    
    //Make the coefficient matrix symmetric
    MatrixXd B=A.transpose();
    for(int i=0;i<eqn;i++){
        for(int j=0;j<var;j++){
            A(i,j)=(A(i,j)+B(i,j))/2.0;
        }
    }

    // Fill the right-hand side vector b with random values
    b = VectorXd::Random(eqn);

    // Redirect output to a file
    freopen("system(eqn).txt", "w", stdout);
    cout<< eqn << endl;
    cout<< var << endl;
    // Display the coefficient matrix A
    cout << A << endl;

    // Display the right-hand side vector b
    cout << b << endl;

    // Close the file
    fclose(stdout);
    return 0; 
}
