#include "stdafx.h"
#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <unordered_map>

using namespace std;

int main()
{
	unordered_map<string, double> probs;

	//insert Transitional and Emission Probabilities
	//Transitional Distribution
	probs["ON|OFF"] = 0.4, probs["ON|ON"] = 0.6;
	probs["OFF|OFF"] = 0.6, probs["OFF|ON"] = 0.4;

	//Emission Distribution
	probs["Warm|ON"] = 0.35, probs["Warm|OFF"] = 0.45, 
	probs["Hot|ON"] = 0.35, probs["Hot|OFF"] = 0.1, 
	probs["Cold|ON"] = 0.3, probs["Cold|OFF"] = 0.45;

	//Initial State Probabilty
	probs["initial=ON"] = 0.5, probs["initial=OFF"] = 0.5;

	string inputSeq;
	cout << "------------------------------------------------------------" << endl;
	cout << "Please input sequence of symbols {Warm, Hot, Cold} seperated by '-' \nto return the probability of observing such sequence.\nE.g. Cold-Warm-Hot-Warm-Cold" << endl;
	cout << "------------------------------------------------------------" << endl;
	
	cin >> inputSeq;
	
	//parse input into symbols and push to vector.
	vector<string> seq;
	stringstream ss(inputSeq);
	string i;
	while (getline(ss, i, '-'))
	{
		seq.push_back(i);

	}

	//workout initial probabilities
	double stateON = probs["initial=ON"], stateOFF = probs["initial=OFF"];

	//Foreach observed symbol in sequence, Forward through input sequence and workout probabilities
	for (int i = 0; i < seq.size(); i++)
	{
		double stateONTemp = stateON;
		stateON = (stateONTemp * probs["ON|ON"] + stateOFF * probs["ON|OFF"])* probs[seq[i] + "|ON"];
		stateOFF = (stateONTemp * probs["OFF|ON"] + stateOFF * probs["OFF|OFF"])* probs[seq[i] + "|OFF"];
	}

	//return sum of both probabilities as
	double finalState = stateON + stateOFF;

	cout << "------------------------------------------------------------" << endl;
	cout << "Probability of observing sequence " << inputSeq << ":" << endl;
	cout << stateON;
	cout << finalState << endl;
	cout << "------------------------------------------------------------" << endl;

	return 0;
}

