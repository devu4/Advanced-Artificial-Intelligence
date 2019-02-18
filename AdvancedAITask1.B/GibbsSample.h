#include <iostream>
#include <string>
#include <unordered_map>
#include <vector>
#include <random>

using namespace std;

//simple node to link states to parents
class Node {
public:
	vector<string> parents;//string of parents for node
	bool value;//current state of node

	Node(vector<string> pa, bool v ){
		parents = pa;
		value = v;
	}

	Node(bool v) {
		parents = {};
		value = v;
	}

	Node() {};

};


class GibbsSample
{

public:
	unordered_map<string, Node> nodes; 

	unordered_map<string, double> bn;
	GibbsSample();
	vector<string> Children(string query);
	double DoGibbsSample(string query, unordered_map<string, Node> evidense, int N);
	~GibbsSample() {};
	bool randomTF();
	double random();
	string getFullKey(string n);
};

GibbsSample::GibbsSample() {

	//Initialise Bayes Net
	Node A = Node(randomTF());
	Node PP = Node(randomTF());
	Node S = Node(vector<string>{ "PP", "A" }, randomTF());
	Node YF = Node({ "S" }, randomTF());
	Node G = Node(randomTF());
	Node LC = Node(vector<string>{ "G", "S" }, randomTF());
	Node AD = Node({ "G" }, randomTF());
	Node BED = Node(randomTF());
	Node AL = Node(randomTF());
	Node C = Node(vector<string>{ "AL", "LC" }, randomTF());
	Node F = Node(vector<string>{ "LC", "C" }, randomTF());
	Node CA = Node(vector<string>{ "AD", "F" }, randomTF());

	nodes = { { "A", A }, { "PP", PP }, { "S", S }, { "YF", YF }, 
				  { "G", G },{ "LC", LC },{ "AD", AD },{ "BED", BED },
				  { "AL", AL },{ "C", C },{ "F", F },{ "CA", CA } };
	
	//Probability Distribution
	//Anxiety
	bn["A"] = 0.64277;
	//Peer Pressure
	bn["PP"] = 0.32997;
	//Smoking
	bn["S|-PP,-A"] = 0.43118;
	bn["S|PP,-A"] = 0.74591;
	bn["S|-PP,A"] = 0.8686;
	bn["S|PP,A"] = 0.91576;
	//Yellow Fingers
	bn["YF|-S"] = 0.23119;
	bn["YF|S"] = 0.95372;
	//genetics
	bn["G"] = 0.15953;
	//lung cancer
	bn["LC|-G,-S"] = 0.23146;
	bn["LC|G,-S"] = 0.86996;
	bn["LC|-G,S"] = 0.83934;
	bn["LC|G,S"] = 0.99351;
	//Attention Disorder
	bn["AD|-G"] = 0.28956;
	bn["AD|G"] = 0.68706;
	//Born an even day
	bn["BED"] = 0.5;
	//allergy
	bn["AL"] = 0.32841;
	//coughing
	bn["C|-AL,-LC"] = 0.1347;
	bn["C|AL,-LC"] = 0.64592;
	bn["C|-AL,LC"] = 0.7664;
	bn["C|AL,LC"] = 0.99947;
	//fatigue
	bn["F|-LC,-C"] = 0.35312;
	bn["F|LC,-C"] = 0.56514;
	bn["F|-LC,C"] = 0.80016;
	bn["F|LC,C"] = 0.89589;
	//car accident
	bn["CA|-AD,-F"] = 0.2274;
	bn["CA|AD,-F"] = 0.779;
	bn["CA|-AD,F"] = 0.78861;
	bn["CA|AD,F"] = 0.97169;
};

vector<string> GibbsSample::Children(string query)
{
	vector<string> children = {};
	for (auto n : nodes)
	{
		//If we don't already have it in our list, add it
			if 
			(
				//if child of query
				(find(n.second.parents.begin(), n.second.parents.end(), query) != n.second.parents.end())
				&&
				//and not already in children vector
				!(find(children.begin(), children.end(), n.first) != children.end())
			)
			{
				//then append.
				children.push_back(n.first);
			}
	}

	return children;
}

//Markov-Chain Monte Carlo Gibbs Sampling function
double GibbsSample::DoGibbsSample(string query, unordered_map<string, Node> evidense, int N)
{
	//create list of nonevidense states
	vector<string> nonEvidense;
	int Counts[2] = { 0 };

	for (auto i : nodes)
	{
		nonEvidense.push_back(i.first);
	}

	
	//fix evidense variables and remove these from nonevidense vector
	for (auto i : evidense)
	{
		nodes[i.first].value = i.second.value;

		for (int y=0; y < nonEvidense.size(); y++)
		{
			if(nonEvidense[y] == i.first)
				nonEvidense.erase(nonEvidense.begin() + y);
		}
	}

	//create cache of children nodes for each node
	unordered_map<string, vector<string>> children;
	for (auto n : nonEvidense)
		children[n] = Children(n);

	for (int x = 0; x < N; x++)
	{
		//Increment the count based on the current state of the query node
		if (nodes[query].value)
			Counts[0]++;
		else
			Counts[1]++;

		//loop through all non-evidense nodes
		for (auto n : nonEvidense)
		{
			//Calculate the conditional probability of this node, given its parents
			string key = getFullKey(n);
			double P_true = bn[key];
			double P_false = 1.0 - P_true;
			
			//Calculate the conditional probabilities of the child nodes given their parents
			for (auto c : children[n]) {

				nodes[n].value = true;
				double pt = bn[getFullKey(c)];

				// P(c | n = false, OtherParents(c))
				nodes[n].value = false;
				double pf = bn[getFullKey(c)];

				//Times parent node probabilty by chance of child
				// If the node is false, we need 1 - P.This is because the conditional Probability supplies us with P(c = true)
				if (nodes[c].value)
				{
					P_true *= pt;
					P_false *= pf;
				}
				else
				{
					P_true *= (1.0 - pt);
					P_false *= (1.0 - pf);
				}
			}

			//Normalise the output to get the probability this node is true
			double P = P_true / (P_true + P_false);
			//sample randomly and set value for node
			if (random() <= P)
				nodes[n].value = true;
			else
				nodes[n].value = false;
		}
	}

	//normalise counts and return probability
	double normal = (double) Counts[0] / (Counts[1] + Counts[0]);
	return normal;
}


string GibbsSample::getFullKey(string n)
{
	//varStates[n].parents[0];
	string key = n;
	vector<string> parents;

	//loop through parents and get their current states
	for (int i = 0; i < nodes[n].parents.size(); i++)
	{
		if (i == 0)
			key += "|";
		else
			key += ",";

		//name of parent
		string name = nodes[n].parents[i];

		if (nodes[name].value)
			key += name;
		else
			key += "-" + name;
	}

	return key;
}

bool GibbsSample::randomTF() {
	double rand = random();
	//cout << rand << endl;
	//get value for key to be true;
	double valueT = 0.5;


	if (rand < valueT)
		return true;
	else
		return false;

}

double GibbsSample::random() {

	//create random number
	random_device rd;
	default_random_engine generator(rd()); // rd() provides a random seed
	uniform_real_distribution<double> distribution(0.00, 1.00);
	return (double)distribution(generator);
}