#include <iostream>
#include <fstream>
#include <algorithm>
#include "WordModelMgr.h"
#include "DictionaryTrie.h"

#define DICTIONARY_DATA_FILE "data/sowpods.txt"
#define TIMING_PENALTY_THRES 300
#define WORD_BREAK_HIGH_THRES 600
#define WORD_BREAK_LOW_THRES 400
#define DECAY_PER_MS (1.0 / 400)

using namespace std;
namespace posixtime = boost::posix_time;


// Interface used for integration with Python wrapper via ctypes.
extern "C"
{
	WordModelMgr * WordModelMgr_new() { return new WordModelMgr(); }
	void WordModelMgr_delete(WordModelMgr * wmm) { delete wmm; }
	bool WordModelMgr_Initialize(WordModelMgr * wmm) { return wmm->Initialize(); }
	void WordModelMgr_AddLetterPrediction(WordModelMgr * wmm, int letterIndex, double confidence, double doubleProb) 
	{ wmm->AddLetterPrediction(letterIndex, confidence, doubleProb); }
	void WordModelMgr_FinalizePrediction(WordModelMgr * wmm) { wmm->FinalizePrediction(); }
	const char * WordModelMgr_GetBestPrediction(WordModelMgr * wmm) { return wmm->GetBestPrediction(); }
	const char * WordModelMgr_GetNextPrediction(WordModelMgr * wmm, double * prob) { return wmm->GetNextPrediction(prob); }
	void WordModelMgr_Reset(WordModelMgr * wmm) { wmm->Reset(); }
	void WordModelMgr_DumpCandidates(WordModelMgr * wmm) { wmm->DumpCandidates(); }
}


//================================================================================
// PhraseCandidate implementation
//================================================================================
DictionaryTrieNode * PhraseCandidate::m_rootNode = 0;
const double PhraseCandidate::MIN_PROBABILITY = 0.001;

vector<PhraseCandidate *> PhraseCandidate::BuildPhraseCandidates(int letterIndex, double confidence, double doubleProb)
{
	// Build a new set of phrase candidates based on this one.

	// For any existing phrase candidate, applying a new letter has several possible outcomes:
	// 1) The new letter was a false positive.
	//    - This possibility is accounted for simply by keeping the existing phrase candidate.
	// 2) The new letter is appended to the current word.
	// 3) The new letter is a double appended to the current word.
	// 4) The new letter is the start of a new word.
	// 5) The new letter is a double that starts a new word.
	// We also must reduce existing phrase probability to reflect chance of true positive.
	char letter = 'A' + letterIndex;

	// Timing calculations.
	posixtime::ptime currTime = posixtime::microsec_clock::local_time();
	posixtime::time_duration msDiff;
	double penalty = 1.0;
	double wordBreakProb = 0.25;
	if (!m_phrase.empty())
	{
		msDiff = currTime - m_createTime;
		int ms = static_cast<int>(msDiff.total_milliseconds());
		if (ms < TIMING_PENALTY_THRES)
		{
			// New letter arrived faster than ideally expected.
			// Apply penalty proportionally.
			penalty = (static_cast<double>(ms) / TIMING_PENALTY_THRES);
		}

		// Calculate word break probability.
		// This is driven by the timing.  If there was a long delay, it's more likely that this represents a word break.
		// Word break probability is constrained to the range [0.25 .. 0.75].
		if (ms <= WORD_BREAK_LOW_THRES)
			wordBreakProb = 0.25;
		else if (ms < WORD_BREAK_HIGH_THRES)
		{
			double wordBreakScale = ((static_cast<double>(ms) - WORD_BREAK_LOW_THRES) /
									 (WORD_BREAK_HIGH_THRES - WORD_BREAK_LOW_THRES));
			wordBreakProb = 0.25 + (wordBreakScale * 0.5);
		}
		else
			wordBreakProb = 0.75;

		// Check if this is a duplicate of the last letter.
		//if (*m_phrase.rbegin() == letter)
		//	penalty /= 2.0;

		// Penalize confidence as needed.
		confidence *= penalty;
	}

	// Decay towards target probability.
	msDiff = currTime - m_lastDecay;
	m_lastDecay = currTime;
	if (m_masterProb > m_targetProb)
	{
		m_masterProb -= static_cast<double>(msDiff.total_milliseconds()) * DECAY_PER_MS;
		if (m_masterProb < m_targetProb)
			m_masterProb = m_targetProb;
	}

	// Quickly return nothing if new baseline probability is below the minimum.
	vector<PhraseCandidate *> candidates;
	double newProb = m_masterProb * confidence;
	if (newProb < MIN_PROBABILITY)
		return candidates;

	// Reduce probability of existing phrase candidate.
	m_targetProb *= (1.0 - confidence);

	// Build new phrase candidates.
	DictionaryTrieNode * node;
	if (m_currNode->m_isWord)
	{
		// Currently at a complete word - it's possible to start a new word.
		// Build case: letter is start of a new word.
		string phrase = m_phrase;
		phrase.push_back(' ');
		phrase.push_back(letter);
		node = m_rootNode->m_dtnSubNodes[letterIndex];
		candidates.push_back(new PhraseCandidate(phrase, newProb * wordBreakProb * (1.0 - doubleProb), node, currTime));

		// Build case: letter is a double starting a new word (e.g.: llama).
		node = node->m_dtnSubNodes[letterIndex];
		if (node)
		{
			phrase.push_back(letter);
			candidates.push_back(new PhraseCandidate(phrase, newProb * wordBreakProb * doubleProb, node, currTime));
		}
	}

	node = m_currNode->m_dtnSubNodes[letterIndex];
	if (node)
	{
		// The new letter could continue the existing word.
		// Build case: letter is appended to current word.
		string phrase = m_phrase;
		phrase.push_back(letter);
		candidates.push_back(new PhraseCandidate(phrase, newProb * (1.0 - wordBreakProb) * (1.0 - doubleProb), node, currTime));

		// Build case: letter is a double appended to current word (e.g.: boo).
		node = node->m_dtnSubNodes[letterIndex];
		if (node)
		{
			phrase.push_back(letter);
			candidates.push_back(new PhraseCandidate(phrase, newProb * (1.0 - wordBreakProb) * doubleProb, node, currTime));
		}
	}
		
	return candidates;
}

void PhraseCandidate::DumpCandidate()
{
	cout << m_phrase << ": " << m_masterProb << endl;
}


//================================================================================
// WordModelMgr implementation
//================================================================================
WordModelMgr::~WordModelMgr()
{
	DeleteCandidates();
}

bool WordModelMgr::Initialize()
{
	// Load the dictionary, establish the basic "root" candidate, etc...
	if (!LoadDictionary())
		return false;

	m_candidates.push_back(new PhraseCandidate("", 1.0, m_dt.GetRootNode(), posixtime::microsec_clock::local_time()));

	PhraseCandidate::SetRootNode(m_dt.GetRootNode());

	return true;
}

void WordModelMgr::AddLetterPrediction(int letterIndex, double confidence, double doubleProb)
{
	// Build vector of new candidate phrases by applying the new letter prediction to all existing phrase candidates.
	vector<PhraseCandidate *> newCandidates;
	for (vector<PhraseCandidate *>::iterator iter = m_candidates.begin(); iter != m_candidates.end(); ++iter)
	{
		vector<PhraseCandidate *> candidates = (*iter)->BuildPhraseCandidates(letterIndex, confidence, doubleProb);
		newCandidates.insert(newCandidates.end(), candidates.begin(), candidates.end());
	}

	// Remove candidates with probability below the minimum threshold.
	m_candidates.erase(remove_if(m_candidates.begin(), m_candidates.end(), PhraseCandidate::IsBelowMinProbability),
					   m_candidates.end());

	// Insert new candidates into master vector.
	m_candidates.insert(m_candidates.end(), newCandidates.begin(), newCandidates.end());

	// Sort and normalize probabilities.
	UpdateBestCandidateAndNormalize();
}

void WordModelMgr::FinalizePrediction()
{
	// Remove candidates that ended in the middle of a word.
	m_candidates.erase(remove_if(m_candidates.begin(), m_candidates.end(), PhraseCandidate::IsNotFinal), m_candidates.end());

	// Final update/normalization - sort by target probability in case some candidates are still in the process of decaying.
	UpdateBestCandidateAndNormalize(true);
}

const char * WordModelMgr::GetBestPrediction()
{
	// Return first phrase candidate - vector should already be sorted.
	if (m_candidates.empty())
		return "";
	else
		return m_candidates[0]->GetPhrase().c_str();
}

const char * WordModelMgr::GetNextPrediction(double * prob)
{
	// Return next phrase candidate and increment current index.
	// Also, return its probability in the prob parameter.
	const char * ret = NULL;
	if ( (m_currIndex < static_cast<int>(m_candidates.size())) && (m_currIndex != -1) )
	{
		*prob = m_candidates[m_currIndex]->GetTargetProbability();
		ret = m_candidates[m_currIndex]->GetPhrase().c_str();
		m_currIndex++;
	}
	return ret;
}

void WordModelMgr::Reset()
{
	// Prepare the Word Model Manager to start fresh.
	DeleteCandidates();
	m_candidates.push_back(new PhraseCandidate("", 1.0, m_dt.GetRootNode(), posixtime::microsec_clock::local_time()));
	m_currIndex = -1;
}

void WordModelMgr::DumpCandidates()
{
	// Sort and print all phrase candidates and their probabilities for diagnostic purposes.
	sort(m_candidates.begin(), m_candidates.end(), PhraseCandidate::SortByProbability);

	cout << endl << "Current candidates:" << endl;
	for (vector<PhraseCandidate *>::iterator iter = m_candidates.begin(); iter != m_candidates.end(); ++iter)
		(*iter)->DumpCandidate();
}

bool WordModelMgr::LoadDictionary()
{
	// Load all words in the linefeed-delimited dictionary file into a trie for efficient storage/processing.

	// Extract entire dictionary data file into buffer.
	ifstream ifs(DICTIONARY_DATA_FILE, ios::binary);
	if (!ifs.is_open())
	{
		cerr << "ERROR: Cannot open dictionary file." << endl;
		return false;
	}

	ifs.seekg(0, ios::end);
	int length = static_cast<int>(ifs.tellg());
	ifs.seekg(0, ios::beg);
	char * buffer = new char[length];
	ifs.read(buffer, length);
	ifs.close();
	
	// Load the words into the dictionary trie.
	m_dt.LoadWords(buffer, length);

	delete [] buffer;
	return true;
}

void WordModelMgr::UpdateBestCandidateAndNormalize(bool sortByTargetProb /* = false */)
{
	// Sort the phrase candidates, and then normalize all probabilities so that best candidate has probability 1.0.
	if (sortByTargetProb)
		sort(m_candidates.begin(), m_candidates.end(), PhraseCandidate::SortByTargetProbability);
	else
		sort(m_candidates.begin(), m_candidates.end(), PhraseCandidate::SortByProbability);

	double highestProb = PhraseCandidate::MIN_PROBABILITY;
	if (m_candidates.size() > 0)
		highestProb = m_candidates[0]->GetProbability();

	double normFactor = 1.0 / highestProb;
	for (vector<PhraseCandidate *>::iterator iter = m_candidates.begin(); iter != m_candidates.end(); ++iter)
		(*iter)->AdjustProbability(normFactor);

	m_currIndex = 0;
}

void WordModelMgr::DeleteCandidates()
{
	// Clear out the main phrase candidate vector.
	for (vector<PhraseCandidate *>::iterator iter = m_candidates.begin(); iter != m_candidates.end(); ++iter)
		delete (*iter);
	m_candidates.clear();
}
