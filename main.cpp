#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Eigenvalues>
#include <map>
#include <fstream>
#include <sstream>
#include <algorithm>

using namespace std;

void loadData(vector<vector<float>> &data, vector<int> &categories){
    ifstream infile("data");
    string line;
    while (getline(infile,line)){
        stringstream s(line);
        vector<float> dataEntry;
        for (int i = 0; i < 5; i++){
            string item;
            getline(s,item,',');
            if (i < 4) dataEntry.push_back(stof(item));
            else categories.push_back(stoi(item));
        }
        data.push_back(dataEntry);
    }
}

class Matrix{
public:
    Matrix(int r, int c){
        rows = r;
        cols = c;
        data = new float[rows*cols];
        dataAlloc = true;
    }

    Matrix(vector<vector<float>> &input){
        rows = input.size();
        cols = input[0].size();
        data = new float[rows*cols];
        dataAlloc = true;
        for (int i = 0; i < rows; i++){
            float *rowPtr = getPtr(i,0);
            for (int j = 0; j < cols; j++){
                rowPtr[j] = input[i][j];
            }
        }
    }

    Matrix(Matrix* m, int r){
        rows = 1;
        cols = m->cols;
        data = m->getPtr(r,0);
        dataAlloc = false;
    }

    ~Matrix(){
        if (dataAlloc) delete[] data;
    }

    float *getPtr(int r, int c){
        return data + r * cols + c;
    }

    void print(){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                cout << *getPtr(i, j) << " ";
            }
            cout << endl;
        }
    }

    void zero(){
        for (int i = 0; i < rows; i++){
            for (int j = 0; j < cols; j++){
                *getPtr(i, j) = 0;
            }
        }
    }

    void transpose(Matrix *mT){
        if (mT->rows != cols || mT->cols != rows) return;

        for (int i = 0; i < rows; i++){
            for(int j = 0; j < cols; j++){
                *mT->getPtr(j, i) = *getPtr(i, j);
            }
        }
    }

    void inverse(Matrix *m){
        Matrix adjugate(rows,cols),cofactors(rows,cols);
        if (rows < 3 || cols < 3){
            cout << "ERR" << endl;
            return;
        }
        for (int i = 0; i < rows; i++){
            int factor = (i%2)?-1:1;
            for(int j = 0; j < cols; j++){
                Matrix *exclude = new Matrix(rows-1,cols-1);
                excludedSubmat(i,j,exclude);

                *cofactors.getPtr(i,j) = exclude->determinant() * factor;
                factor = -factor;
                delete exclude;
            }
        }
        cofactors.transpose(&adjugate);
        float invDet = 1/determinant();
        multiply(invDet,&adjugate,m);
    }

    // Output eigenvalues is a single column, eigenvectors are a single column
    void getEigenDecomposition(Matrix *eigenvectors, Matrix *eigenvalues){
        Eigen::MatrixXf EigenMatrix(rows,cols);
        copyToEigen(&EigenMatrix);
        Eigen::EigenSolver<Eigen::MatrixXf> solver;
        solver.compute(EigenMatrix,true);
		Eigen::MatrixXf vectors = solver.eigenvectors().real();
		Eigen::MatrixXf values = solver.eigenvalues().real();
        copyFromEigen(&vectors,eigenvectors);
        copyFromEigen(&values,eigenvalues);
    }

    float determinant(){
        if (rows != cols) return 0;
        float det = 0;
        if (rows == 2) {
            det =  data[0] * data[3] - data[1] * data[2];
        }
        else{
            int sign = 1;
            Matrix *exclude = new Matrix(rows-1,cols-1);
            for (int i = 0; i < cols; i++){
                excludedSubmat(0,i,exclude);
                det += (*getPtr(0,i)) * sign * exclude->determinant();
                sign *= -1;
            }
            delete exclude;
        }
        return det;
    }

    static void vconcat(vector<Matrix*> &inputs, Matrix *outMat){
        int outrow = 0;
        for (int i = 0; i < inputs.size(); i++){
            for (int j = 0; j < inputs[i]->rows; j++){
                float *outrowPtr = outMat->getPtr(outrow,0);
                for (int k = 0; k < inputs[i]->cols; k++){
                    outrowPtr[k] = *inputs[i]->getPtr(j,k);
                }
                outrow++;
            }
        }
    }

    static bool multiply(Matrix *m1, Matrix *m2, Matrix *outMat){
        if (m1->cols != m2->rows || outMat->rows != m1->rows || outMat ->cols != m2->cols) return false;

        outMat->zero();

        for (int i = 0; i< m1->rows; i++){
            for (int j = 0; j < m2->cols; j++){
                for(int k = 0; k < m1->cols; k++){
                    *outMat->getPtr(i, j) += (*m1->getPtr(i, k)) * (*m2->getPtr(k, j));
                }
            }
        }
        return true;
    }

    static void multiply(float f, Matrix *m1, Matrix *outMat){
        for (int i = 0; i < m1->rows; i++){
            for (int j = 0; j < m1->cols; j++){
                *outMat->getPtr(i, j) = *m1->getPtr(i, j) * f;
            }
        }
    }

    static bool add(Matrix *m1, Matrix *m2, Matrix *outMat){
        if (m1->rows != m2->rows || outMat->rows != m1->rows || outMat->cols != m1->cols || m1->cols != m2->cols) return false;
        for (int i = 0; i < m1->rows; i++){
            for (int j = 0; j < m1->cols; j++){
                *outMat->getPtr(i, j) = *m1->getPtr(i, j) + *m2->getPtr(i, j);
            }
        }
        return true;
    }

    static bool subtract(Matrix *m1, Matrix *m2, Matrix *outMat){
        if (m1->rows != m2->rows || outMat->rows != m1->rows || outMat->cols != m1->cols || m1->cols != m2->cols) return false;
        for (int i = 0; i < m1->rows; i++){
            for (int j = 0; j < m1->cols; j++){
                *outMat->getPtr(i, j) = *m1->getPtr(i, j) - *m2->getPtr(i, j);
            }
        }
        return true;
    }

    int rows,cols;

private:
    float *data;
    bool dataAlloc;

    void excludedSubmat(int r, int c, Matrix *m){
        int outrow = 0;
        for (int i = 0; i < rows; i++){
            if (i == r) continue;
            int outcol = 0;
            for (int j = 0; j< cols; j++) {
                if (j == c) continue;
                *m->getPtr(outrow, outcol) = *getPtr(i, j);
                outcol++;
            }
            outrow++;
        }
    }

    void copyToEigen(Eigen::MatrixXf *eigenMat){
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                (*eigenMat)(i,j) = *getPtr(i,j);
            }
        }
    }

    static void copyFromEigen (Eigen::MatrixXf *eigenMat, Matrix *toMat){
        for (int i = 0; i < eigenMat->rows(); i++) {
            for (int j = 0; j < eigenMat->cols(); j++) {
                *toMat->getPtr(i,j) = (*eigenMat)(i,j);
            }
        }
    }
};

class LDA{
public:
    LDA(int nDims, int nClasses, int nBasisVectors, vector<vector<float>> &data, vector<int> labels){
        map<int,vector<vector<float>>> categories;
        vector<int> classSizes;
        for (int i = 0; i < data.size(); i++){
            if (categories.find(labels[i]) == categories.end()){
                categories[labels[i]] = vector<vector<float>>();
            }
            categories[labels[i]].push_back(data[i]);
        }

        vector<Matrix*> classMatrices;
        vector<Matrix *> classMeans;
        for (map<int,vector<vector<float>>>::iterator i = categories.begin(); i != categories.end(); i++) {
            classSizes.push_back(i->second.size());
            classMatrices.push_back(new Matrix(i->second));
            classMeans.push_back(new Matrix(1,nDims));
        }

        Matrix allMats(data.size(),nDims);
        Matrix allMeans (1,nDims);
        Matrix::vconcat(classMatrices,&allMats);

        Matrix withinClassScatter(nDims,nDims);
        withinClassScatter.zero();

        for (int i = 0; i < classMatrices.size(); i++){
            computeMeanVector(classMatrices[i],classMeans[i]);
            Matrix scatter(nDims,nDims);
            computeWithinClassScatterMatrix(classMatrices[i],classMeans[i],&scatter);
            Matrix::add(&withinClassScatter,&scatter,&withinClassScatter);
        }
        computeMeanVector(&allMats,&allMeans);
        Matrix betweenClassScatter(nDims,nDims);
        computeBetweenClassScatterMatrix(&allMeans,classMeans,classSizes,&betweenClassScatter);

		Matrix eigenVecResult(withinClassScatter.rows, nBasisVectors);
        findBasisVectors(&betweenClassScatter,&withinClassScatter, &eigenVecResult, nBasisVectors);

		transformMatrixLDA(&allMats, &eigenVecResult);
		

        for (int i = 0; i < classMatrices.size(); i++){
            delete classMatrices[i];
            delete classMeans[i];
        }
    }

    void printTransform(){
        transformMatrix->print();
    }

    ~LDA(){
        delete transformMatrix;
    }

private:
    LDA(){};

	void computeMeanVector(Matrix *classData, Matrix *meanVec) {

		// Num of coloums in classData & meanVec should be identical which is the number of features 
		if (meanVec->cols != classData->cols) { return; }

		for (int j = 0; j < classData->cols; j++) {      // Looping through each feature 
			float sum = 0.0f;
			for (int i = 0; i < classData->rows; i++) {  // Looping through each meansurement under the feature
				sum += *classData->getPtr(i, j);
			}
			sum /= classData->rows;
			*meanVec->getPtr(0, j) = sum;
		}
	}

	void computeWithinClassScatterMatrix(Matrix *classData, Matrix *meanVec, Matrix *scatter) {
		Matrix   
			x_mi(meanVec->cols, 1),                        // N x 1
			x_mi_t(1, meanVec->cols),					   // 1 x N, transposed version          
			result(meanVec->cols, meanVec->cols);		   // N x N, store vector multiplication result (vec*vec.tranpose)

		scatter->zero();                                   // Initialize

		for (int j = 0; j < classData->rows; j++) {        // Looping through all feature elements within a class
			for (int i = 0; i < meanVec->cols; i++) {      // Accessing each measurement for all features
				// x_mi: feature measurement[i][j] - feature mean[i]
				*x_mi.getPtr(i, 0) = *classData->getPtr(j, i) - *meanVec->getPtr(0, i);  
			}
			x_mi.transpose(&x_mi_t);
			Matrix::multiply(&x_mi, &x_mi_t, &result);
			Matrix::add(&result, scatter, scatter);
		}
	}

	void computeBetweenClassScatterMatrix(Matrix *overallMean, vector<Matrix *> &classMeans, vector<int> &classSizes, Matrix *scatter) {
		/*
		mi_m:   Stores the result of each class mean - allClass mean.
		mi_m_t: Transposed version of mi_m for matrix multiplication
		result: Stores the matrix multiplication result of mi_m & mi_m_t
		*/
		Matrix mi_m(overallMean->cols, overallMean->rows);
		Matrix mi_m_t(overallMean->rows, overallMean->cols);
		Matrix result(overallMean->cols, overallMean->cols);
		//Initialize
		scatter->zero();                                                   

		for (int i = 0; i < classSizes.size(); i++) {           // Looping through each class
			float *eachClassMean = classMeans[i]->getPtr(0, 0);
			for (int j = 0; j < classMeans[0]->cols; j++) {     // Looping through each feature 
				*mi_m.getPtr(0, j) = eachClassMean[j] - *overallMean->getPtr(0, j);
			}
			mi_m.transpose(&mi_m_t);
			Matrix::multiply(&mi_m, &mi_m_t, &result);
			Matrix::add(scatter, &result, scatter);
		}
		// Multiple num of measurement of each feature after summation operator
		Matrix::multiply((float)classSizes[0], scatter, scatter);
	}

	void findBasisVectors(Matrix *betweenClass, Matrix *withinClass, Matrix *eigenVecResult, int numBasisVectors) {

		Matrix invW(withinClass->rows, withinClass->cols);
		Matrix SwSb(withinClass->rows, withinClass->cols); //SwSb: Inv(beteenClass)*withinClass
		withinClass->inverse(&invW);
		Matrix::multiply(&invW, betweenClass, &SwSb);

		// eigenVec: Each coloum vector represnt one eigenvector, numBasisVectors control how many eigenvectors user required
		// eigenVal: Store user defined (numBasisVectors) number of eigenvalues in a coloum vector 
		Matrix eigenVec(withinClass->rows, withinClass->cols); //Eigenvectors
		Matrix eigenVal(withinClass->rows, 1);                 //Eigenvalues
		SwSb.getEigenDecomposition(&eigenVec, &eigenVal);

		// Create a sorted version of eigenvalues, to locate two largest eigenvalues
		Matrix eValsort(withinClass->rows, 1);
		for (int i = 0; i < withinClass->rows; i++) { *eValsort.getPtr(i, 0) = *eigenVal.getPtr(i, 0); }
		sort(eValsort.getPtr(0, 0), eValsort.getPtr(0, withinClass->rows));

		
		// store index where largest two element's idx: l_idx store 1 & 2 in our case
		vector<int> l_idx(2, 0);
		for (int i = 0; i < numBasisVectors; i++) {
			for (int j = 0; j < withinClass->rows; j++) {
				if (*eigenVal.getPtr(j, 0) == *eValsort.getPtr(0, withinClass->rows -1- i))
				{ l_idx[i] = j; }
			}
		}

		Matrix eigenvalues(numBasisVectors,1);
		for (int i = 0; i < withinClass->rows; i++) {
			for (int j = 0; j < numBasisVectors; j++) {
				*eigenVecResult->getPtr(i, j) = *eigenVec.getPtr(i, l_idx[j]);
			}
			if(i<numBasisVectors)
				*eigenvalues.getPtr(i, 0) = *eigenVal.getPtr(l_idx[i], 0);
		}

		// Double check result
		//cout << "Eigenvalues are: \n\n"; eigenvalues.print(); cout << endl << endl << endl;
		//cout << "Eigenvectors are: \n\n"; eigenvectors.print();; cout << endl << endl;
	}

	void transformMatrixLDA(Matrix *classData, Matrix *eigenvectors) {
		transformMatrix = new Matrix(classData->rows, eigenvectors->cols);
		Matrix::multiply(classData, eigenvectors, transformMatrix);
	}



    Matrix *transformMatrix;
};

int main() {
    vector<vector<float>> data;
    vector<int> categories;
    loadData(data,categories);
	int nFeatures = 4;
	int nClass = 3;
	int finalFeature = 2;
    LDA lda(nFeatures, nClass, finalFeature, data,categories);
    lda.printTransform();
}


