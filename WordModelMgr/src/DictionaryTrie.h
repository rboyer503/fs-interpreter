#ifndef __DICTIONARY_TRIE_H__
#define __DICTIONARY_TRIE_H__

// Class representing a node of the dictionary trie data structure.
// Each node implicitly represents a letter of the alphabet and contains an array of up to 26 additional nodes - one for each
// of the possible "next letters".
class DictionaryTrieNode
{
public:
	static const int NUM_SUBNODES = 26;

	// Pointer to an array of 26 sub-nodes.
    // Each sub-node represents a letter of the alphabet.
	DictionaryTrieNode ** m_dtnSubNodes;

	// Does this node represent a valid word?
	bool m_isWord;

	// Did this object create its own sub-nodes?
	bool m_ownsSubNodes;

	DictionaryTrieNode(bool isWord)
	{
		// Allocate memory for sub-node pointers and null them out.
		// (Actual memory for each sub-node will be allocated as necessary.)
		m_dtnSubNodes = new DictionaryTrieNode*[NUM_SUBNODES];
		for (int i = 0; i < NUM_SUBNODES; ++i)
			m_dtnSubNodes[i] = 0;

		m_ownsSubNodes = true;
		m_isWord = isWord;
	}

	DictionaryTrieNode(const DictionaryTrieNode & node)
	{
		// Make a shallow copy.
		m_dtnSubNodes = node.m_dtnSubNodes;
		m_isWord = node.m_isWord;

		// Ownership remains with original trie node.
		m_ownsSubNodes = false;
	}

	~DictionaryTrieNode()
	{
		// Destroy sub-nodes if we have ownership.
		if (m_ownsSubNodes)
		{
			for (int i = 0; i < NUM_SUBNODES; ++i)
				delete m_dtnSubNodes[i];
			delete [] m_dtnSubNodes;
		}
	}
};


// Class representing the entire dictionary trie.
// The root node contains an array with elements for each letter of the alphabet.
// All words starting with 'A' will be found under element 0, 'B' in 1, and so forth.
class DictionaryTrie
{
	// Node representing the root of the dictionary trie.
	DictionaryTrieNode * m_dtnRoot;

public:
	DictionaryTrie()
	{
		// Create the root node and a layer of sub-nodes representing each letter.
        m_dtnRoot = new DictionaryTrieNode(false);
        for (int i = 0; i < DictionaryTrieNode::NUM_SUBNODES; i++)
            m_dtnRoot->m_dtnSubNodes[i] = new DictionaryTrieNode(false);
	}

	~DictionaryTrie()
	{
		delete m_dtnRoot;
	}

	DictionaryTrieNode * GetRootNode()
	{
		return m_dtnRoot;
	}

	void LoadWords(char * data, int length);
};

#endif
