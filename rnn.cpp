#include "rnn.h"

RnnLayer::RnnLayer() = default;

float *RnnLayer::activateSoftmax(float *Ht, int n) {
    float maxH =  Ht[0];
    float s =0, *softmax = new float[n];
    for(auto i=0;i<n;i++) {
        maxH = std::fmax(Ht[i], maxH);
    }
    for(auto i=0;i<n;i++) {
        s+=std::exp(Ht[i]-maxH);
    }
    for(auto i=0;i<n;i++) {
        softmax[i]=std::exp(Ht[i]-maxH)/s;
    }
    delete s,maxH;
    return softmax;
}


float *RnnLayer::getX() const {
    return X;
}

void RnnLayer::setX(float *X) {
    RnnLayer::X = X;
}

float *RnnLayer::getHprev() const {
    return Hprev;
}

void RnnLayer::setHprev(float *Hprev) {
    RnnLayer::Hprev = Hprev;
}

int RnnLayer::getN() const {
    return nData;
}

void RnnLayer::setN(int n) {
    RnnLayer::nData = n;
}

float RnnLayer::dLoss(const float *Yout) {
    float s = 0;
    for(int i=0;i<nData;i++) {
        s-=Y[i]/Yout[i]/nData;
    }
    return s;
}

RnnLayer::RnnLayer(int Hn) {
    RnnLayer::Hn = Hn;
    Hprev = new float[Hn];
    for(int i=0;i<Hn;i++)
        Hprev[i]=0;
}

float RnnLayer::Loss(const float *Yout) {
    float s = 0;
    for(int i=0;i<nData;i++) {
        s-=Y[i]*std::log(Yout[i])/nData;
    }
    return s;
}

void RnnLayer::loadData(char *filename) {
    std::ifstream in(filename);
    std::string s;
    if(in.is_open()) {
        char ch;
        int idx =0;
        while(in.get(ch)) {
            s+=ch;
            if(char2int.find(ch)!=char2int.end()) {
                char2int[ch]=idx;
                int2char[idx]=ch;
            }
            idx++;
        }
        fullLen = static_cast<int>(s.size());
        fullEncoded = new float[s.size()];
        for(auto i=0;i<s.size();i++){
            fullEncoded[i]=char2int[s[i]];
        }
        in.close();
        delete idx,ch,s;
    }
}

void RnnLayer::genericBatches(int sizeBatch) {
    std::mt19937 gen(time(0));
    std::uniform_int_distribution<> uid(0, fullLen-sizeBatch-2);
    int idx = uid(gen);
    auto *resp = new float[sizeBatch];
    Y = new float[sizeBatch];
    for(int i=0;i<sizeBatch;i++) {
        resp[i]=fullEncoded[idx+i];
        Y[i] = fullEncoded[idx+i+1];
    }
    delete uid,gen,idx;
    nData = sizeBatch;
    setX(resp);
}


void SimpleRnn::generateWight() {
    std::mt19937 gen(time(0));
    std::uniform_real_distribution<> urd(0, 1);

    W = new float*[Hn];
    U = new float*[Hn];
    b = new float[Hn];
    V = new float*[nData];
    c = new float[nData];
    for(int i=0;i<Hn;i++) {
        W[i] = new float[nData];
        U[i] = new float[Hn];
        b[i] = static_cast<float>(urd(gen));
    }

    for(int i=0;i<Hn;i++) {
        for (int j = 0; j < nData; j++) {
            W[i][j] = static_cast<float>(urd(gen));
            V[i][j] = static_cast<float>(urd(gen));
        }
        for (int j = 0; j < Hn; j++) {
            U[i][j] = static_cast<float>(urd(gen));
        }
    }

    for(int i=0;i<nData;i++) {
        V[i] = new float[Hn];
        c[i] = static_cast<float>(urd(gen));
        for(int j=0;j<Hn;j++) {
            V[i][j] = static_cast<float>(urd(gen));
        }
    }
}

float *SimpleRnn::forward() {
    temp1 = multiplyMatrixVector(W,X,Hn,nData);
    temp1 = sumPointVectors(temp1,multiplyMatrixVector(U,Hprev,Hn,Hn),Hn);
    temp1 = sumPointVectors(temp1,b,Hn);
    Ht = tanh(temp1,Hn);
    temp2 = multiplyMatrixVector(V,Ht,nData,Hn);
    temp2 = sumPointVectors(temp2,c,nData);
    return activateSoftmax(temp2,nData);
}

void SimpleRnn::saveWeights(char *filename) {
    ofstream out(filename);
    out<<Hn<<" "<<nData<<std::endl;
    saveMatrix(out,U,Hn,Hn);
    saveMatrix(out,W,Hn,nData);
    saveMatrix(out,V,nData,Hn);
    saveVector(out,b,Hn);
    saveVector(out,c,nData);
    out.close();
}

void SimpleRnn::loadWeights(char *filename) {
    ifstream in(filename);
    if(in.is_open()){
        in>>Hn>>nData;
        U =loadMatrix(in,Hn,Hn);
        W = loadMatrix(in,Hn,nData);
        V = loadMatrix(in,nData,Hn);
        b = loadVector(in,Hn);
        c = loadVector(in,nData);
        in.close();
    }
}


