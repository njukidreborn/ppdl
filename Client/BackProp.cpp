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


Integer SortedList[16382];
int SortedListNum[16382];

int partion(Integer* list, int* num, int start, int end)
{
	if( start == end )
		return 0;
	int pivot = start + 1;
	for(int i = start + 1; i < end; i++){
		if( list[i] < list[start] ){
			//swap i pivot
			list[i].swap(list[pivot]);
			int temp = num[i];
			num[i] = num[pivot];
			num[pivot] = temp;
			pivot++;
		}
	}
	//swap start pivot
	list[start].swap(list[pivot-1]);
	int temp = num[start];
	num[start] = num[pivot - 1];
	num[pivot - 1] = temp;
	return pivot - 1;
}
void QuickSort(Integer* list, int *num, int start, int end)
{
	if( start < end ){
		int pivot = partion(list, num, start, end);
		QuickSort(list, num, start, pivot);
		QuickSort(list, num, pivot+1, end);
	}
}
void SortedListCreate(ModularArithmetic group_S, Integer g, Integer rB)
{
	for(int i = 0; i < 16382; i++ ){
		Integer ex = i * rB;
		Integer target = group_S.Exponentiate(g, ex);
		SortedList[i] = target;
		SortedListNum[i] = i;
	}
	QuickSort(SortedList, SortedListNum, 0, 16382);
}

int SortedListBinarySearch(Integer target, int start, int end){
	int count = 0;
	while(start <= end){
		count++;
		int mid = (start + end) / 2;
		if( SortedList[mid] == target ) {
			//cout << "Binary Search: " << count << endl;
			return SortedListNum[mid];
		}
		else if( SortedList[mid] > target ){
			end = mid - 1;
		}
		else {
			start = mid + 1;
		}
	}
	return -1;
}

int SortedListSearch(Integer target)
{
	for(int i = 0; i < 16382; i++){
		if( SortedList[i] == target ) {
			cout << "Count Sorted: " << i << endl;
			return SortedListNum[i];
		}
	}
	return -1;
}

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

void SendInt(SOCKET s, int i) {
	unsigned char buf[8];
	buf[0] = (i >> 24) & 0xff;
	buf[1] = (i >> 16) & 0xff;
	buf[2] = (i >>  8) & 0xff;
	buf[3] = (i >>  0) & 0xff;
	send(s, (char *)buf, 4, 0);
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
				weight[i][j][k]=(double)(rand())/(RAND_MAX/2) - 1;//32767
				//weight[i][j][k] = 0.0;

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
	cout << " Share " << endl;
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++)
			for(int k=0;k<lsize[i-1]+1;k++) {
				int wei = 10000 * weight[i][j][k];
				SendInt(conn, wei);
				weight[i][j][k] = wei / 10000.0;
				//cout << weight[i][j][k] << endl;
				//return ;
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
	CryptoPP::Integer p, q, g;
	byte output[128];
	unsigned char buf[8];
	recv(conn, (char *)output, 128, 0);
	p.Decode(output, 128);
	recv(conn, (char *)output, 128, 0);
	q.Decode(output, 128);
	recv(conn, (char *)output, 128, 0);
	g.Decode(output, 128);

	ModularArithmetic group_S(p);
	CryptoPP::Integer ub = RandomQuadraticResidue(q); // ub is only known by b
	CryptoPP::Integer VB = group_S.Exponentiate(g, ub);

	Send(conn, VB);
	CryptoPP::Integer rB = RandomQuadraticResidue(q);

	SortedListCreate(group_S, g, rB);

	// client 
	for(int i=1;i<numl;i++)
		for(int j=0;j<lsize[i];j++) {
			cout << "#";
			for(int k=0;k<lsize[i-1]+1;k++)  {
				// PartyB XB = dwt[i][j][k] YB = 10000.0
				CryptoPP::Integer XB, YB;
				
				XB = (int)(dwt[i][j][k] / 2);
				YB = 1000 / 2;
				CryptoPP::Integer xA1, xA2, yA1, yA2;
				CryptoPP::Integer theta1, theta2, pi1, pi2;

				recv(conn, (char *)output, 128, 0);
				xA1.Decode(output, 128);
				recv(conn, (char *)output, 128, 0);
				xA2.Decode(output, 128);
				recv(conn, (char *)output, 128, 0);
				yA1.Decode(output, 128);
				recv(conn, (char *)output, 128, 0);
				yA2.Decode(output, 128);

				//	cout << "yA2 " << yA2 << endl;
				if( XB >= 0 )
					theta1 = group_S.Exponentiate(group_S.Divide((xA2 * group_S.Exponentiate(g, XB)),  group_S.Exponentiate(xA1, ub)), rB);
				else
					theta1 = group_S.Exponentiate(group_S.Divide(xA2* group_S.Divide(1, group_S.Exponentiate(g, -XB)),  group_S.Exponentiate(xA1, ub)), rB);

				pi1 = group_S.Exponentiate(xA1, rB);

				if( YB >= 0 )
					theta2 = group_S.Exponentiate(group_S.Divide(yA2 * group_S.Exponentiate(g, YB), group_S.Exponentiate(yA1, ub)), rB);
				else
					theta2 = group_S.Exponentiate(group_S.Divide(yA2 * group_S.Divide(1, group_S.Exponentiate(g, -YB)), group_S.Exponentiate(yA1, ub)), rB);

				pi2 = group_S.Exponentiate(yA1, rB);


				Send(conn, theta1);
				Send(conn, theta2);
				Send(conn, pi1);
				Send(conn, pi2);

				CryptoPP::Integer eta1, etap1 ;
				CryptoPP::Integer eta2, etap2 ;

				recv(conn, (char *)output, 128, 0);
				eta1.Decode(output, 128);
				recv(conn, (char *)output, 128, 0);
				eta2.Decode(output, 128);

				//recal deta  and send it to PartyA
			
				int ipeta1 = SortedListBinarySearch(eta1, 0, 16382);
				if( ipeta1 < 0 )
					ipeta1 = - SortedListBinarySearch(group_S.Divide(1, eta1), 0, 16382);
				int ipeta2 = SortedListBinarySearch(eta2, 0, 16382);
				if( ipeta2 < 0 )
					ipeta2 = - SortedListBinarySearch(group_S.Divide(1, eta2), 0, 16382);
				//cout << "peta1 " << ipeta1 << endl;
				//cout << "peta2 " << ipeta2 << endl;
				
				SendInt(conn, ipeta1);
				SendInt(conn, ipeta2);
				double avgWeight = (double)ipeta1 / ((double)ipeta2 * 10);

				weight[i][j][k] = prevWeight[i][j][k] +  avgWeight;
				/*
				int d = dwt[i][j][k];
				if( d * 2 != ipeta1 ){
					cout << "fuck ";
					cout << ipeta1 << " " << d << endl;
				}
				*/
				//cout << avgWeight << endl;
				//cout << avgWeight << " " << dwt[i][j][k] / 10000.0 << endl;
			}
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
				weight[i][j][k] += alpha * prevDwt[i][j][k];
			}
			// delta bias
			weight[i][j][lsize[i-1]]+=alpha*prevDwt[i][j][lsize[i-1]];
		}
	}
	*/

	//	adjust weights usng steepest descent	
	for(int i = 1; i < numl; i++){
		for(int j = 0; j < lsize[i]; j++){
			for(int k = 0; k < lsize[i-1]; k++){
				// adjust weight
				prevDwt[i][j][k] = beta * delta[i][j] * out[i-1][k];
				weight[i][j][k] += prevDwt[i][j][k];
				dwt[i][j][k] += weight[i][j][k] - prevWeight[i][j][k];
			}
			// adjust bias
			prevDwt[i][j][lsize[i-1]]=beta*delta[i][j];
			weight[i][j][lsize[i-1]]+=prevDwt[i][j][lsize[i-1]];
			dwt[i][j][lsize[i-1]]+=weight[i][j][lsize[i-1]] - prevWeight[i][j][lsize[i-1]];
		}
	}
}

