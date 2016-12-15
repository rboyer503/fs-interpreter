#pragma once

class DirProcessFunctor;
class AdaptiveCanny;


class ImageProcessCoordinator
{
	enum eIPCMode { CAM_AUTO_ADJUST, FIND_HAND, TRACK_HAND, FINAL_eIPCMode_ENTRY };
	static char * eIPCModeStrings[FINAL_eIPCMode_ENTRY];

	eIPCMode _mode;
	boost::posix_time::ptime _initTime;
	cv::CascadeClassifier _cascade;
	int _minNeighbors;
	cv::Rect _looseBound;
	cv::Mat _lut;
	cv::Mat _focusWeights;

	char _snapshotLetter;
	int _snapshotCount;
	int _snapshotIndices[26];
	bool _isClip;

	std::string _currDirName;

	cv::Scalar _avgMean;
	cv::Scalar _avgStdDev;

	AdaptiveCanny * _adaptiveCanny;

public:
	ImageProcessCoordinator();
	~ImageProcessCoordinator();
	bool Initialize();
	void ProcessFrame(cv::Mat & inputFrame);
	void StartCapture(char letter, int numSnapshots, bool clip);
	void WriteSnapshotFiles();
	void WriteClipFiles();
	void CalcDataFileStats();

private:
	void SetMode(eIPCMode mode);
	double GetHandOutlierFactor(const cv::Mat & handROI, bool debugOutput);
	void SetLooseBound(const cv::Rect & handBound);
	void ClipLooseBound();
	void BuildLUT(const cv::Mat & frame, int smoothFactor, double thresholdFactor);
	void BuildHistograms(const cv::Mat & frame, std::vector<cv::Mat> & histograms);
	void ProcessDirectory(std::string dir, char currLabel, DirProcessFunctor & functor);
};
