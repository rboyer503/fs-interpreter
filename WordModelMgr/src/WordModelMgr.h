#ifndef __WORD_MODEL_MGR_H__
#define __WORD_MODEL_MGR_H__

#include <boost/date_time/posix_time/posix_time.hpp>
#include <string>
#include <vector>
#include "DictionaryTrie.h"


// Class representing a possible phrase that could be constructed based on the letter predictions so far provided.
// The actual textual phrase and probability of validity are maintained, along with some miscellaneous state info.
class PhraseCandidate
{
	// Root node of the dictionary trie.
	static DictionaryTrieNode * m_rootNode;

	// The textual phrase.
	std::string m_phrase;

	// Relative probability that this phrase candidate is correct.
	// (Note: This does not indicate absolute probability.  This is only useful for purposes of comparison.)
	double m_masterProb;

	// Target probability to decay to, based on confidence in phrase extensions.
	double m_targetProb;

	// Current node in the dictionary trie (node of the last letter).
	DictionaryTrieNode * m_currNode;

	// Time this phrase candidate was created.
	// Used for calculation of timing penalties.
	// (For example, if two letters are added within a few milliseconds, it's very unlikely that they are both valid.)
	boost::posix_time::ptime m_createTime;

	// Last time master probability was decayed towards target probability.
	boost::posix_time::ptime m_lastDecay;

public:
	// Minimum probability before phrase candidate is culled.
	static const double MIN_PROBABILITY;

	static void SetRootNode(DictionaryTrieNode * node)
	{
		m_rootNode = node;
	}

	static bool IsBelowMinProbability(PhraseCandidate * candidate)
	{
		// Predicate for testing whether phrase candidate has dipped below probability threshold.
		return (candidate->m_masterProb < MIN_PROBABILITY);
	}

	static bool IsNotFinal(PhraseCandidate * candidate)
	{
		// Predicate for testing whether phrase candidate ends on a valid word.
		return (!candidate->m_currNode->m_isWord);
	}

	static bool SortByProbability(const PhraseCandidate * l,
								  const PhraseCandidate * r)
	{
		// Comparison predicate for sorting phrase candidates by probability.
		return (l->m_masterProb > r->m_masterProb);
	}

	static bool SortByTargetProbability(const PhraseCandidate * l,
										const PhraseCandidate * r)
	{
		// Comparison predicate for sorting phrase candidates by target probability.
		return (l->m_targetProb > r->m_targetProb);
	}

	PhraseCandidate(std::string init, double confidence, DictionaryTrieNode * node, const boost::posix_time::ptime & currTime) :
		m_phrase(init), m_masterProb(confidence), m_targetProb(confidence), m_currNode(node),
		m_createTime(currTime), m_lastDecay(currTime)
	{
	}

	double GetProbability() const
	{
		return m_masterProb;
	}

	std::string & GetPhrase()
	{
		return m_phrase;
	}

	void AdjustProbability(double factor)
	{
		m_masterProb *= factor;
		m_targetProb *= factor;
	}

	std::vector<PhraseCandidate *> BuildPhraseCandidates(int letterIndex, double confidence, double doubleProb);

	void DumpCandidate();
};


// Top-level class orchestrating the building and management of phrase candidates.
// As letter predictions are added, this class builds and assigns relative probabilities to all possible resultant phrases.
// The number of phrase candidates will increase exponentially.  To avoid unfavorable computational complexity, phrase
// candidates are dropped as their relative probability decreases below a minimum threshold and word validity is efficiently
// checked by traversing a trie.
class WordModelMgr
{
	// The dictionary trie, efficiently storing all words in the dictionary.
	DictionaryTrie m_dt;

	// The main vector of phrase candidates.
	std::vector<PhraseCandidate *> m_candidates;

	// Index of next best prediction (for state when looping on GetNextPrediction()).
	int m_currIndex;

public:
	WordModelMgr() : m_currIndex(-1)
	{}
	~WordModelMgr();
	bool Initialize();
	void AddLetterPrediction(int letterIndex, double confidence, double doubleProb);
	void FinalizePrediction();
	const char * GetBestPrediction();
	const char * GetNextPrediction(double * prob);
	void Reset();
	void DumpCandidates();

private:
	bool LoadDictionary();
	void UpdateBestCandidateAndNormalize(bool sortByTargetProb = false);
	void DeleteCandidates();
};

#endif
