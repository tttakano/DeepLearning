#include"bits/stdc++.h"
#include<fstream>
#include<random>

//#include<bits/stdc++.h>
using namespace std;
#define print(x) cout<<x<<endl;
#define rep(i,a,b) for(int i=a;i<b;i++)
typedef long long ll;
const ll mod = 2000000;

#define pi 3.141592


int main() {
	default_random_engine generator;
	normal_distribution<double> distribution;
	double deg[10000];
	double x[5][10000];
	double now = 0;
	rep(i, 0, 10000) {
		x[0][i] = rand();
		x[1][i] = rand();
		x[2][i] = rand();
		x[3][i] = rand();
		x[4][i] = rand();
	}
	ofstream ofs("train.csv");
	rep(i, 0, 10000) {
		ofs << x[0][i] << "," << x[1][i] << "," << x[2][i] << "," << x[3][i] << "," << x[4][i] << "," << 32768 * cos(x[0][i] * pi / 180.0) + 32768 * sin(x[1][i] * pi / 180.0) + x[2][i] + 6000 * log10(x[3][i]) + 200 * sqrt(x[4][i]) + (float)distribution(generator) * 3000 << "," << 32768 * cos(x[0][i] * pi / 180.0) + 32768 * sin(x[1][i] * pi / 180.0) + x[2][i] + 6000 * log10(x[3][i]) + 200 * sqrt(x[4][i]) << endl;
	}
	return 0;
}