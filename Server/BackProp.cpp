#include "backprop.h"
#include <time.h>
#include <stdlib.h>
#include <iostream>
#include "integer.h"

#include <string>
#include <stdexcept>
#include <sstream>
#include "osrng.h"
#include "integer.h"
#include "nbtheory.h"
#include "dh.h"
#include "secblock.h"
#include "elgamal.h"
#include "asn.h"
#include "modarith.h"

#pragma comment(lib,"ws2_32.lib")

using namespace CryptoPP;
using namespace std;


CryptoPP::Integer RandomQuadraticResidue(CryptoPP::Integer p) {
	AutoSeededRandomPool rng;
	byte randomBytes[64];
	rng.GenerateBlock(randomBytes, 64);
	Integer t = Integer(randomBytes, 64);
	return ModularExponentiation(t, 2, p);
}

void Send(SOCKET s, CryptoPP::Integer item){
	byte output[128];
	item.Encode(output, 128);
	send(s, (char *)output, 128, 0);
}


//	initializes and allocates memory on heap
CBackProp::CBackProp(int nl,int *sz,double b,double a):beta(b),alpha(a)
{

	//	set no of layers and their sizes
	numl=nl;
	lsize=new int[numl];

	// initial size of each layer.
	for(int i=0;i<numl;i++){
		lsize[i]=sz[i];
	}

	//	allocate memory for output of each neuron
	out = new double*[numl];

	for(int i=0;i<numl;i++){
		out[i]=new double[lsize[i]];
	}

	//	allocate memory for delta
	delta = new double*[numl];

	for(int i=1;i<numl;i++){
		delta[i]=new double[lsize[i]];
	}

	//	allocate memory for weights
	weight = new double**[numl];
	prevWeight = new double**[numl];

	for(int i=1;i<numl;i++){
		weight[i]=new double*[lsize[i]];
		prevWeight[i]=new double*[lsize[i]];
	}
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			weight[i][j]=new double[lsize[i-1]+1];
			prevWeight[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	allocate memory for previous weights
	prevDwt = new double**[numl];
	dwt = new double**[numl];

	for(int i=1;i<numl;i++){
		prevDwt[i]=new double*[lsize[i]];
		dwt[i]=new double*[lsize[i]];
	}
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			prevDwt[i][j]=new double[lsize[i-1]+1];
			dwt[i][j]=new double[lsize[i-1]+1];
		}
	}

	//	seed and assign random weights
	srand((unsigned)(time(NULL)));
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) 
				//weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767
				weight[i][j][k] = 0.0;

	//	initialize previous weights to 0 for first iteration
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) {
				prevDwt[i][j][k]=(double)0.0;
				dwt[i][j][k]=(double)0.0;
			}

// Note that the following variables are unused,
//
// delta[0]
// weight[0]
// prevDwt[0]

//  I did this intentionaly to maintains consistancy in numbering the layers.
//  Since for a net having n layers, input layer is refered to as 0th layer,
//  first hidden layer as 1st layer and the nth layer as output layer. And 
//  first (0th) layer just stores the inputs hence there is no delta or weigth
//  values corresponding to it.
}



CBackProp::~CBackProp()
{
	//	free out
	for(int i=0;i<numl;i++)
		delete[] out[i];
	delete[] out;

	//	free delta
	for(int i=1;i<numl;i++)
		delete[] delta[i];
	delete[] delta;

	//	free weight
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] weight[i][j];
	for(int i=1;i<numl;i++)
		delete[] weight[i];
	delete[] weight;

	//	free prevDwt
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			delete[] prevDwt[i][j];
	for(int i=1;i<numl;i++)
		delete[] prevDwt[i];
	delete[] prevDwt;

	//	free layer info
	delete[] lsize;
}
void CBackProp::shareWeight(SOCKET conn)
{
	cout << " Share Weight " << endl;
	unsigned char buf[8];
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++)  {
				int	wei;
				recv(conn, (char *)buf, 4, 0);
				wei = (buf[0] << 24) + (buf[1] << 16) + (buf[2] << 8) + (buf[3] << 0);
				weight[i][j][k] = wei / 10000.0;
				//cout << weight[i][j][k];
			}

}
void CBackProp::storeWeight()
{
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) 
				prevWeight[i][j][k] = weight[i][j][k];
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) {
				dwt[i][j][k]=(double)0.0;
			}
}

void CBackProp::recalWeight(SOCKET conn)
{
	// server
	// share the dwt
	// re apply
	AutoSeededRandomPool rnd;
	unsigned int bits = 512;
	DH dh;
	dh.AccessGroupParameters().GenerateRandomWithKeySize(rnd, bits);

	if(!dh.GetGroupParameters().ValidateGroup(rnd, 3))
		throw runtime_error("Failed to validate prime and generator");
	size_t count = 0;
	const Integer& p = dh.GetGroupParameters().GetModulus();
	const Integer& q = dh.GetGroupParameters().GetSubgroupOrder();

	Send(conn, p);
	Send(conn, q);
	CryptoPP::Integer g = RandomQuadraticResidue(p);
	Send(conn, g);
	CryptoPP::Integer ua = RandomQuadraticResidue(q);

	ModularArithmetic group_S(p);
	CryptoPP::Integer VA = group_S.Exponentiate(g, ua);

	// receive VB from Party B.
	byte output[128];
	unsigned char buf[8];
	CryptoPP::Integer VB;
	recv(conn,(char *)output, 128,0);
	VB.Decode(output, 128);

	CryptoPP::Integer rA1 = RandomQuadraticResidue(q); 
	CryptoPP::Integer rA2 = RandomQuadraticResidue(q); 

	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) { 
				//Party A XA = dwt[i][j][k] YA = 10000.0
				CryptoPP::Integer XA, YA;
				//dwt[i][j][k] = -1505;
				XA = (int)(dwt[i][j][k] / 2); 
				YA = 1000 / 2; // divided by 10

				//Send g and p, q to B
				CryptoPP::Integer xA1 = group_S.Exponentiate(g, rA1);
				CryptoPP::Integer xA2;
				if( XA >= 0)
					xA2 = (group_S.Exponentiate((VA * VB), rA1) * group_S.Exponentiate(g, XA));
				else
					xA2 = group_S.Divide(group_S.Exponentiate((VA * VB), rA1) , group_S.Exponentiate(g, -XA));

				CryptoPP::Integer yA1 = group_S.Exponentiate(g, rA2);
				CryptoPP::Integer yA2;
				if( YA >= 0)
					yA2 = (group_S.Exponentiate((VA * VB), rA2) * group_S.Exponentiate(g, YA));
				else
					yA2 = group_S.Divide(group_S.Exponentiate((VA * VB), rA2), group_S.Exponentiate(g, -YA));

				Send(conn, xA1);
				Send(conn, xA2);
				Send(conn, yA1);
				Send(conn, yA2);

				CryptoPP::Integer theta1, theta2, pi1, pi2;
				recv(conn,(char *)output, 128,0);
				theta1.Decode(output, 128);
				recv(conn,(char *)output, 128,0);
				theta2.Decode(output, 128);
				recv(conn,(char *)output, 128,0);
				pi1.Decode(output, 128);
				recv(conn,(char *)output, 128,0);
				pi2.Decode(output, 128);

				// compute eta
				CryptoPP::Integer eta1 = group_S.Divide(theta1 , group_S.Exponentiate(pi1, ua));
				CryptoPP::Integer eta2 = group_S.Divide(theta2 , group_S.Exponentiate(pi2, ua));

				Send(conn, eta1);
				Send(conn, eta2);

				//TODO: Receive the average result
				int	peta1, peta2;
				recv(conn, (char *)buf, 4, 0);
				peta1 = (buf[0] << 24) + (buf[1] << 16) + (buf[2] << 8) + (buf[3] << 0);
				//cout << "peta1 " << peta1 << endl;
				recv(conn,(char *)buf, 4,0);
				peta2 = (buf[0] << 24) + (buf[1] << 16) + (buf[2] << 8) + (buf[3] << 0);
				//cout << "peta2 " << peta2 << endl;
				double avgWeight = (double)peta1 / ((double)peta2 * 10.0 );
				weight[i][j][k] = prevWeight[i][j][k] +  avgWeight;
				
			}
}
//	sigmoid function
double CBackProp::sigmoid(double in)
{
		return (double)(1/(1+exp(-in)));
}

//	mean square error
double CBackProp::mse(double *tgt) const
{
	double mse=0;
	for(int i=0;i<lsize[numl-1];i++){
		mse+=(tgt[i]-out[numl-1][i])*(tgt[i]-out[numl-1][i]);
	}
	return mse/2;
}


//	returns i'th output of the net
double CBackProp::Out(int i) const
{
	return out[numl-1][i];
}

double *CBackProp::LayerOut(int i) const
{
	double *lo = new double[lsize[i]];
	for(int k = 0; k < lsize[i]; k++){
		lo[k] = out[i][k];
	}
	return lo;
}

// feed forward one set of input
void CBackProp::ffwd(double *in)
{
	double sum;

	//	assign content to input layer
	for(int i=0;i<lsize[0];i++)
		out[0][i]=in[i];  // output_from_neuron(i,j) Jth neuron in Ith Layer

	//	assign output(activation) value 
	//	to each neuron usng sigmoid func
	for(int i=1;i<numl;i++){				// For each layer
		for(int j=0;j<lsize[i];j++){		// For each neuron in current layer
			sum=0.0;
			for(int k=0;k<lsize[i-1];k++){		// For input from each neuron in preceeding layer
				sum+= out[i-1][k]*weight[i][j][k];	// Apply weight to inputs and add to sum
			}
			sum+=weight[i][j][lsize[i-1]];		// Apply bias
			out[i][j]=sigmoid(sum);				// Apply sigmoid function
		}
	}
}


//	backpropogate errors from output
//	layer uptill the first hidden layer
void CBackProp::bpgt(double *in,double *tgt)
{
	double sum;

	//	update output values for each neuron
	ffwd(in);

	//	find delta for output layer
	for(int i=0;i<lsize[numl-1];i++){
		delta[numl-1][i] = out[numl-1][i] * (1 - out[numl-1][i]) * (tgt[i] - out[numl-1][i]);
	}

	//	find delta for hidden layers	
	for(int i=numl-2;i>0;i--){
		for(int j=0;j<lsize[i];j++){
			sum=0.0;
			for(int k=0;k<lsize[i+1];k++){
				sum+=delta[i+1][k]*weight[i+1][k][j];
			}
			delta[i][j]=out[i][j]*(1-out[i][j])*sum;
		}
	}
/*
	//	apply momentum ( does nothing if alpha=0 )
	for(int i=1;i<numl;i++){
		for(int j=0;j<lsize[i];j++){
			for(int k=0;k<lsize[i-1];k++){
				// delta weight
				// weight[i][j][k] += alpha * prevDwt[i][j][k];
				dwt[i][j][k] += alpha * prevDwt[i][j][k];
			}
			// delta bias
			// weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
			dwt[i][j][lsize[i-1]] += alpha*prevDwt[i][j][lsize[i-1]];
		}
	}
*/
	//	adjust weights usng steepest descent	
	for(int i = 1; i < numl; i++){
		for(int j = 0; j < lsize[i]; j++){
			for(int k = 0; k < lsize[i-1]; k++){
				prevDwt[i][j][k] = beta * delta[i][j] * out[i-1][k];
				weight[i][j][k] += prevDwt[i][j][k];
				dwt[i][j][k] += weight[i][j][k] - prevWeight[i][j][k];
			}
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
			dwt[i][j][lsize[i-1]]+=weight[i][j][lsize[i-1]] - prevWeight[i][j][lsize[i-1]];
		}
	}
}

