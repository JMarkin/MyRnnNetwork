#ifndef RNN_LIBRARY_H
#define RNN_LIBRARY_H
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <random>
#include <map>

float sigma(float x) {
    return static_cast<float>(1.0 / (1.0 + std::exp(-x)));
}

float* tanh(const float* x, int n) {
    auto *resp = new float[n];
    for(int i=0;i<n;i++)
        resp[i] = 2*sigma(2*x[i])+1;
    return resp;
}

float* dTanh(float* x, int n) {
    auto *resp = new float[n];
    for(int i=0;i<n;i++)
        resp[i] = 4*sigma(x[i])*(1-sigma(x[i]));
    return resp;
}

float* sigma(float* x, int n) {
    auto *resp = new float[n];
    for(int i=0;i<n;i++)
        resp[i] = sigma(x[i]);
    return resp;
}

float* dSigma(float* x, int n) {
    auto *resp = new float[n];
    for(int i=0;i<n;i++)
        resp[i] = sigma(x[i])*(1-sigma(x[i]));
    return resp;
}
/**
 *
 * @param M matrix n x m
 * @param V vector m
 * @param n
 * @param m
 * @return
 */
float* multiplyMatrixVector(float **M, const float *V,int n,int m) {
    auto * resp = new float[n];
    for(int i=0;i<n;i++) {
        float s =0;
        for(int j=0;j<m;j++) {
            s+= M[i][j]*V[j];
        }
        resp[i]=s;
    }
    return resp;
}

/**
 *
 * @param V1
 * @param V2
 * @param n
 * @return
 */
float* sumPointVectors(const float*V1,const float*V2,int n) {
    auto * resp = new float[n];
    for(int i=0;i<n;i++) {
        resp[i]=V1[i]+V2[i];
    }
    return resp;
}

using std::ofstream;
void saveMatrix(ofstream &out,float**M,int n,int m) {
    for(int i=0;i<n;i++) {
        for(int j=0;j<m;j++) {
            out<<M[i][j]<<" ";
        }
    }
    out<<std::endl;
}

void saveVector(ofstream &out,float*V,int n) {
    for(int i=0;i<n;i++) {
        out<<V[i]<<" ";
    }
    out<<std::endl;
}

using std::ifstream;
float** loadMatrix(ifstream &in,int n,int m) {
    auto **M = new float*[n];
    for(int i=0;i<n;i++) {
        M[i] = new float[m];
        for(int j=0;j<m;j++) {
            in>>M[i][j];
        }
    }
    return M;
}

float* loadVector(ifstream &in, int n) {
    auto * V = new float[n];
    for(int i=0;i<n;i++) {
        in>>V[i];
    }
    return V;
}

class RnnLayer {
protected:
    std::map<char,int> char2int;
    std::map<int,char> int2char;
    float *fullEncoded;
    float *X,*Y,*Hprev, *Ht;
    /**
     * nData size of X,Y
     * Hn size of H
     */
    int nData,Hn,fullLen;
    void genericBatches(int sizeBatch);
public:
    explicit RnnLayer(int Hn);

    RnnLayer();

    int getN() const;

    void setN(int n);

    float *getX() const;

    void setX(float *X);

    float *getHprev() const;

    void setHprev(float *Hprev);

    float *activateSoftmax(float *Ht, int n);

    float dLoss(const float *Yout);

    float Loss(const float *Yout);

    void loadData(char* filename);

};
/*
 * Elman/SimpleRnn network
 */
class SimpleRnn:RnnLayer {
    /**
     * hT = tanh(W*xT + UhT−1 + b)
     * y = softmax(hT*V + c)
     * U size of Hn*Hn
     * W size of Hn*nData
     * V size of nData*Hn
     * b size of Hn
     * c size of nData
     * temp1 = W*xT + UhT−1 + b
     * temp2 = VhT + c
     */
    float **W,**U,*b,**V,*c,*temp1,*temp2;
    void saveWeights(char* filename);
    void loadWeights(char* filename);
public:
    SimpleRnn(): RnnLayer(){};
    SimpleRnn(int Hn) : RnnLayer(Hn){};
    float* forward();
    void generateWight();
};

#endif