#include "BackProp.h"
// NeuralNet.cpp : Defines the entry point for the console application.

#include "backprop.h"
#include <fstream>
#include <time.h>
#include <string>
#include <stdexcept>
#include <sstream>
#include <WINSOCK2.H>

#pragma comment(lib,"ws2_32.lib")


using namespace std;


double beta = 0.1, alpha = 0.1, Thresh =  0.00001;
int reverseInt (int i) 
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

void read_label(int * labels)
{
	ifstream file ("t10k-labels.idx1-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
        int number_of_items = 0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_items,sizeof(number_of_items));
        number_of_items = reverseInt(number_of_items);
		for( int i = 0; i < number_of_items; ++i ){
			unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
			labels[i] = temp - 0x00 ;
		}
	}
}
void train_images(int *labels, CBackProp *ae, CBackProp *sm, int num_iter)
{
	// socket transfer.
	WORD myVersionRequest;
	WSADATA wsaData;
	myVersionRequest = MAKEWORD(1,1);
	int err = WSAStartup(myVersionRequest, &wsaData);
	if (!err)
	{
		cout << ("Open socket\n");
	} 
	else
	{
		cout << ("Failed !");
		return ;
	} 
	SOCKET serSocket=socket(AF_INET,SOCK_STREAM,0);//create a socket

	SOCKADDR_IN addr;
	addr.sin_family=AF_INET;
	addr.sin_addr.S_un.S_addr=htonl(INADDR_ANY);//bind ip address
	addr.sin_port=htons(6000);// bind port

	bind(serSocket,(SOCKADDR*)&addr,sizeof(SOCKADDR));//bind
	listen(serSocket,5);//second parameter tells the max connection number.

	// start listening.
	SOCKADDR_IN clientsocket;
	int len=sizeof(SOCKADDR);
	SOCKET serConn=accept(serSocket,(SOCKADDR*)&clientsocket,&len);//如果这里不是accept而是conection的话。。就会不断的监听

	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
	if (file.is_open())
	{
		int magic_number = 0;
		int number_of_images = 0;
		int n_rows = 0;
		int n_cols = 0;
		// train autoencoder
		ae->shareWeight(serConn);
		sm->shareWeight(serConn);
		for( int k = 0; k < num_iter; k++){
			file.seekg(0, ios_base::beg);
			magic_number=0;
			number_of_images=0;
			n_rows=0;
			n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number)); 
			magic_number= reverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= reverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= reverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= reverseInt(n_cols);
			cout << n_rows << n_cols << endl;
			double img[784];
			ae->storeWeight();
			for(int i=0;i < number_of_images;++i)
			{
				for(int r=0;r<n_rows;++r)
				{
					for(int c=0;c<n_cols;++c)
					{
						unsigned char temp=0;
						file.read((char*)&temp,sizeof(temp));
						img[r * 28 + c] = (temp - 0x00) / 255.0 ;
					}
				}
				ae->bpgt(img, img); 
			}
			// calculate the delta weight
			// get the delta weight and share it with another party
			// apply the average delta to the initial weights
			ae->recalWeight(serConn);
		}
		for( int k = 0; k < num_iter; k++){
			file.seekg(0, ios_base::beg);
			magic_number=0;
			number_of_images=0;
			n_rows=0;
			n_cols=0;
			file.read((char*)&magic_number,sizeof(magic_number)); 
			magic_number= reverseInt(magic_number);
			file.read((char*)&number_of_images,sizeof(number_of_images));
			number_of_images= reverseInt(number_of_images);
			file.read((char*)&n_rows,sizeof(n_rows));
			n_rows= reverseInt(n_rows);
			file.read((char*)&n_cols,sizeof(n_cols));
			n_cols= reverseInt(n_cols);
			cout << n_rows << n_cols << endl;
			double img[784];
			sm->storeWeight();
			for(int i=0;i < number_of_images;++i)
			{
				for(int r=0;r<n_rows;++r)
				{
					for(int c=0;c<n_cols;++c)
					{
						unsigned char temp=0;
						file.read((char*)&temp,sizeof(temp));
						img[r * 28 + c] = (temp - 0x00) / 255.0 ;
					}
				}
				// read ith image
				double result[10];
				for(int k = 0; k < 10; k++)
					result[k] = 0;
				result[labels[i]] = 1;
				double * inner;
				ae->ffwd(img);
				inner = ae->LayerOut(1);
				sm->bpgt(inner, result); 
				
				/*
				 * End of one iteration. Parties start share their delta weights.
				 * get the intial weights at the beginning of this iteration
				 * get the delta weights at the end of this iteration
				 * calculate the average delta weight and apply these to initial weights.
				 */
			}
			sm->recalWeight(serConn);
		}
		cout << "Iteration ends" << endl;
    }
}
void test_images(int *labels, CBackProp *ae, CBackProp *sm)
{
	ifstream file ("t10k-images.idx3-ubyte", ios::binary);
    if (file.is_open())
    {
        int magic_number=0;
        int number_of_images=0;
        int n_rows=0;
        int n_cols=0;
        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);
        file.read((char*)&number_of_images,sizeof(number_of_images));
        number_of_images= reverseInt(number_of_images);
        file.read((char*)&n_rows,sizeof(n_rows));
        n_rows= reverseInt(n_rows);
        file.read((char*)&n_cols,sizeof(n_cols));
        n_cols= reverseInt(n_cols);
		double img[784];
		int count = 0;
		for(int i=0;i<number_of_images;++i)
        {
            for(int r=0;r<n_rows;++r)
            {
                for(int c=0;c<n_cols;++c)
                {
                    unsigned char temp=0;
                    file.read((char*)&temp,sizeof(temp));
					img[r * 28 + c] = (temp - 0x00) / 255.0;
                }
            }
			// read ith image
			ae->ffwd(img); 
			double *inner = ae->LayerOut(1);
			sm->ffwd(inner);
			int max_no = 0;
			double max = sm->Out(0);
			for(int itr = 0; itr < 10; itr++ ){
				if( sm->Out(itr) > max ){
					max = sm->Out(itr);
					max_no = itr;
				}
			}
			if( max_no == labels[i] )
				count++;
			
        }
		cout << count / 10000.0 << endl;
		cout << "Iteration ends" << endl;
    }
}
int main(int argc, char* argv[])
{
	int labels[60000];
	read_label(labels);
	int numLayers = 3, lSz[3] = {784,50,784};

	// maximum no of iterations during training
	long num_iter = 10;

	// Creating the net
	CBackProp *ae= new CBackProp(numLayers, lSz, beta, alpha);
	int softmax[2] = {50, 10};
	CBackProp *sm = new CBackProp(2, softmax, beta, alpha);
	cout<< endl <<  "Now training the network...." << endl;	
	clock_t start, end;
	double duration;
	start = clock();
	train_images(labels, ae, sm, num_iter);
	cout << "start test " << endl; 
	test_images(labels, ae, sm);
	end = clock();
	duration = (double)(end - start) / 1000.0 / 60.0; // min
	cout << " Use: " << duration << " Min" << endl;
	
	
	return 0;
}



