#include "DictionaryTrie.h"


void DictionaryTrie::LoadWords(char * data, int length)
{
	// Build the Trie data structure from all linefeed-delimited words provided.
	DictionaryTrieNode * tempNode = m_dtnRoot;
	for (int i = 0; i < length; ++i)
	{
		if (*data == 0xA)
		{
			// Delimiter found, mark current node as a complete word.
			tempNode->m_isWord = true;
			tempNode = m_dtnRoot;
		}
		else
		{
			// Traverse to corresponding node, constructing it if needed.
			int subindex = *data - 'a';
			if (!tempNode->m_dtnSubNodes[subindex])
				tempNode->m_dtnSubNodes[subindex] = new DictionaryTrieNode(false);
			tempNode = tempNode->m_dtnSubNodes[subindex];
		}
		data++;
	}
}
