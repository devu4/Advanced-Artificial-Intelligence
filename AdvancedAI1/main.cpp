#include "stdafx.h"
#include <iostream>
#include <string>

using namespace std;

double getProbablityOfDiseaseGivenTest(double pOfD, double pOfTGivenD, double pOfNotTGivenNotD)
{
	//workout needed probabilities in joint distribution
	double pOfNotD = 1 - pOfD;
	double pOfTGivenNotD = 1 - pOfNotTGivenNotD;

	//use bayes rule to workout P( d | t )
	double pOfDGivenT = (pOfTGivenD*pOfD) / ((pOfTGivenD*pOfD) + (pOfTGivenNotD*pOfNotD));

	return pOfDGivenT;
}

int main()
{
	cout << "------ Bayes Rule Disease Calculator -----" << endl;


	char choice;
	do {
		cout << "Please input Prior Probability of having a Disease P( d )" << endl;
		double pOfD, pOfTGivenD, pOfNotTGivenNotD;
		cin >> pOfD;

		cout << "Please input the probability that the test is positive given the person has the disease P( t | d )" << endl;
		cin >> pOfTGivenD;

		cout << "Please input the probability that the test is negative given the person does not have the disease P( -t | -d)" << endl;
		cin >> pOfNotTGivenNotD;

		cout << "The probability of having the disease given the test was positive has been calculated as:" << endl;
		cout << getProbablityOfDiseaseGivenTest(pOfD, pOfTGivenD, pOfNotTGivenNotD) << endl;

		cout << "Would you like to re-enter values for probabilities and try again? Y/N" << endl;
		cin >> choice;

	} while (choice == 'Y' || choice == 'y');

    return 0;
}

