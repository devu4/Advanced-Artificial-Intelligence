#include "stdafx.h"
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include "GibbsSample.h"

using namespace std;

int main()
{
	GibbsSample lucas;

	int samples = 10000;
	string X = "S";
	string E;

	cout << "------------------------------------------------" << endl;
	cout << "Medical Diagnosis Problem Solver" << endl;
	cout << "Please enter the number of samples required:" << endl;
	cout << "------------------------------------------------" << endl;
	
	cin >> samples;

	cout << "------------------------------------------------" << endl;
	cout << "Please enter query variable from:\nA, PP, S, YF, G, LC, AD, BED, AL, C, F, CA:" << endl;
	cout << "------------------------------------------------" << endl;

	cin >> X;

	cout << "------------------------------------------------" << endl;
	cout << "Please enter evidense variables in format (F=1,C=0) :" << endl;
	cout << "Delimit variables with comma and use 1 for true or 0 for false" << endl;
	cout << "------------------------------------------------" << endl;

	cin >> E;

	//quickly parse the evidense string into a unordered_map of nodes.
	stringstream ss(E);
	string s;
	unordered_map<string, Node> evidense;

	while (getline(ss, s, ',')) {

		stringstream ss2(s);
		string var;
		string val;
		getline(ss2, var, '=');
		getline(ss2, val, '=');

		bool b = (val == "1" ? true : false);

		evidense[var] = Node(b);
	}

	//pass the variable to sampler and do calculations
	double S = lucas.DoGibbsSample(X, evidense, samples);
	//double S = lucas.rejSampling(X, evidense, samples);
	double notS = 1 - S;

	cout << "------------------------------------------------" << endl;
	cout << "With " << samples << " samples the P("<< X << "|" << E << ") : <" << S << " " << notS << ">" << endl;
	cout << "------------------------------------------------" << endl;

    return 0;
}

