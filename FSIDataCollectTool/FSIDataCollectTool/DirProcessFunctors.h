#pragma once

class AdaptiveCanny;

class DirProcessFunctor
{
public:
	virtual ~DirProcessFunctor() {};
	virtual void operator()(std::string filePath, char label) = 0;
};

class WriteDataLabelFunctor : public DirProcessFunctor
{
	static const int OUT_SIZE_TABLE_SIZE = 5;
	static const int OUT_LEAN_TABLE_SIZE = 25;

	enum eOutSize { OUT_MEDIUM, OUT_LARGE, OUT_SMALL };
	enum eOutLean { OUT_MIDDLE, OUT_LEFT, OUT_RIGHT };
	static eOutSize _outSizeTable[OUT_SIZE_TABLE_SIZE];
	static eOutLean _outLeanTable[OUT_LEAN_TABLE_SIZE];

	std::ofstream _ofData;
	std::ofstream _ofLabels;
	int _imageSize;
	int _currOutIndex;
	char _currLabel[1];

protected:
	AdaptiveCanny * _adaptiveCanny;
	cv::Mat _focusWeights;

	WriteDataLabelFunctor(std::string dataFileName, std::string labelFileName, int imageSize);
	~WriteDataLabelFunctor();
	void ExplodeAndWriteData(std::vector<cv::Mat> & inputData, char label);

private:
	void GenerateDataAndLabel(std::vector<cv::Mat> & outputData);
};

class WriteSnapshotsFunctor : public WriteDataLabelFunctor
{
public:
	WriteSnapshotsFunctor(std::string dataFileName, std::string labelFileName);
	virtual ~WriteSnapshotsFunctor() {};
	virtual void operator()(std::string filePath, char label);
};

class WriteClipsFunctor : public WriteDataLabelFunctor
{
public:
	WriteClipsFunctor(std::string dataFileName, std::string labelFileName);
	virtual ~WriteClipsFunctor() {};
	virtual void operator()(std::string filePath, char label);
};

class CalcStatsFunctor : public DirProcessFunctor
{
	cv::Scalar _totalMean, _totalStdDev;
	cv::Scalar _maxMean, _maxStdDev;
	cv::Scalar _minMean, _minStdDev;
	int _imageCount;

public:
	CalcStatsFunctor();
	virtual void operator()(std::string filePath, char label);
	void DisplayResults() const;
};
