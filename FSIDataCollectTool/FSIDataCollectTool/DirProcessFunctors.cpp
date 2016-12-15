#include "stdafx.h"
#include "DirProcessFunctors.h"
#include "AdaptiveCanny.h"
#include "Utility.h"
#include "Globals.h"

using namespace std;
using namespace cv;


WriteDataLabelFunctor::eOutSize WriteDataLabelFunctor::_outSizeTable[OUT_SIZE_TABLE_SIZE] = 
	{ OUT_MEDIUM, OUT_LARGE, OUT_MEDIUM, OUT_SMALL, OUT_MEDIUM };
WriteDataLabelFunctor::eOutLean WriteDataLabelFunctor::_outLeanTable[OUT_LEAN_TABLE_SIZE] = 
	{ OUT_MIDDLE, OUT_LEFT, OUT_MIDDLE, OUT_RIGHT, OUT_MIDDLE,
	  OUT_LEFT, OUT_MIDDLE, OUT_RIGHT, OUT_MIDDLE, OUT_MIDDLE,
	  OUT_MIDDLE, OUT_RIGHT, OUT_MIDDLE, OUT_MIDDLE, OUT_LEFT,
	  OUT_RIGHT, OUT_MIDDLE, OUT_MIDDLE, OUT_LEFT, OUT_MIDDLE,
	  OUT_MIDDLE, OUT_MIDDLE, OUT_LEFT, OUT_MIDDLE, OUT_RIGHT};


WriteDataLabelFunctor::WriteDataLabelFunctor(string dataFileName, string labelFileName, int imageSize) :
	_imageSize(imageSize), _currOutIndex(0)
{
	_ofData.open(dataFileName, ofstream::binary);
	_ofLabels.open(labelFileName, ofstream::binary);

	_adaptiveCanny = new AdaptiveCanny(0.015, 2.0, false);

	srand(static_cast<unsigned int>(time(NULL)));
}

WriteDataLabelFunctor::~WriteDataLabelFunctor()
{
	if (_adaptiveCanny)
		delete _adaptiveCanny;
}

void WriteDataLabelFunctor::ExplodeAndWriteData(vector<Mat> & inputData, char label)
{
	// Update current label.
	_currLabel[0] = label;

	// Generate 3 data points with variety of sizes/leans/constrasts.
	// Generate random size adjustment factor between ~ 0.05 and 0.1, with higher probability of lower value.
	double sizeAdjustBase = ((rand() % 57) / 100.0) + 1.44; // Random value in range [1.44 .. 2.0]
	double sizeAdjust = (pow(sizeAdjustBase, 3) + 2.0) / 100.0; // Random size adjustment factor in range [~0.05 .. 0.1]
	int sizeAdjustFactor = static_cast<int>(inputData[0].cols * sizeAdjust); // 0.0625);
	int sizeAdjustOffsetTop = sizeAdjustFactor / 2;
	int sizeAdjustOffsetBottom = sizeAdjustOffsetTop;
	if ((sizeAdjustFactor % 2) == 1)
		++sizeAdjustOffsetBottom;
	for (int i = 0; i < 3; ++i)
	{
		// Pulling all randomization processing to start of loop to ensure all component images in the input vector use same size/lean/constrast.
		// Generate random angle between ~ 2 and 8, with higher probability of lower value.
		double angleBase = ((rand() % 101) / 100.0) + 1.25; // Random value in range [1.25 .. 2.25]
		double angle = pow(angleBase, 3); // Random angle in range [~2.0 .. ~11.39]

		// Generate random alpha and beta for constrast randomization for roughly 50% of data points.
		double alphaDiff = 0.0;
		double beta = 0.0;
		if ( (rand() % 2) == 0 )
		{
			alphaDiff = pow(static_cast<double>(rand() % 116), 3) / 5000000.0; // Random alpha diff value in range [0.0 .. ~0.3]
			if ( (rand() % 2) == 0 )
				alphaDiff = -alphaDiff;
			else
				beta = alphaDiff * 128.0; 
		}

		// Process each component image in input data to generate output data.
		vector<Mat> outputData;
		for (vector<Mat>::iterator it = inputData.begin(); it != inputData.end(); ++it)
		{
			// Size manipulation.
			Mat tempImage;
			Mat clipROI;
			Mat resizedQuality;
			switch (_outSizeTable[_currOutIndex % OUT_SIZE_TABLE_SIZE])
			{
			case OUT_MEDIUM:
				(*it).copyTo(resizedQuality);
				break;
			case OUT_LARGE:
				resize((*it), tempImage, Size((*it).cols + sizeAdjustFactor, (*it).rows + sizeAdjustFactor));
				clipROI = tempImage(Rect(sizeAdjustOffsetTop, sizeAdjustOffsetTop, (*it).cols, (*it).rows));
				clipROI.copyTo(resizedQuality);
				break;
			case OUT_SMALL:
				resize((*it), tempImage, Size((*it).cols - sizeAdjustFactor, (*it).rows - sizeAdjustFactor));
				copyMakeBorder(tempImage, resizedQuality, 
								sizeAdjustOffsetTop, sizeAdjustOffsetBottom, 
								sizeAdjustOffsetTop, sizeAdjustOffsetBottom, 
								BORDER_REPLICATE);
				break;
			}

			// Lean manipulation.
			Mat * finalQuality;
			Mat leanedQuality;
			Mat rotationMatrix;
			eOutLean lean = _outLeanTable[_currOutIndex % OUT_LEAN_TABLE_SIZE];
			switch (lean)
			{
			case OUT_MIDDLE:
				finalQuality = &resizedQuality;
				break;
			case OUT_LEFT:
			case OUT_RIGHT:
				Point2f pointCenter(resizedQuality.cols / 2.0f, resizedQuality.rows / 2.0f);
				rotationMatrix = getRotationMatrix2D(pointCenter, (lean == OUT_LEFT ? angle : -angle), 1.0);
				warpAffine(resizedQuality, leanedQuality, rotationMatrix, resizedQuality.size());
				finalQuality = &leanedQuality;
				break;
			}

			// Constrast manipulation.
			Mat alphaBetaMatrix;
			(*finalQuality).convertTo(alphaBetaMatrix, -1, 1.0 + alphaDiff, -beta);

			// Load final image to output data vector.
			outputData.push_back(alphaBetaMatrix);
		}

		// Write output data to file along with label entry.
		GenerateDataAndLabel(outputData);
	}
}

void WriteDataLabelFunctor::GenerateDataAndLabel(vector<Mat> & outputData)
{
	int maxShift = static_cast<int>(outputData[0].cols * BOUND_FOCUS_OFFSET);

	for (int i = 0; i < 4; i++)
	{
		int xShift = (rand() % maxShift);
		if ( (rand() % 2) == 0 )
			xShift = -xShift;
		int yShift = (rand() % maxShift);
		if ( (rand() % 2) == 0 )
			yShift = -yShift;

		// Resize focus area of final master quality images.
		// Write each to data file.
		// Write only one entry to label file.
		Rect focusRect = GetBoundingBox(BOUND_FOCUS, Rect(0, 0, outputData[0].cols, outputData[0].rows), true);
		focusRect.x += xShift;
		focusRect.y += yShift;
		for (vector<Mat>::iterator it = outputData.begin(); it != outputData.end(); ++it)
		{
			Mat focusQuality = (*it)(focusRect);

			Mat resizedQuality;
			resize(focusQuality, resizedQuality, Size(_imageSize, _imageSize));

			_ofData.write(reinterpret_cast<const char *>(resizedQuality.data), resizedQuality.rows * resizedQuality.cols);
		}

		_ofLabels.write(_currLabel, 1);
		++_currOutIndex;
	}
}


WriteSnapshotsFunctor::WriteSnapshotsFunctor(std::string dataFileName, std::string labelFileName) :
	WriteDataLabelFunctor(dataFileName, labelFileName, IMAGE_SIZE)
{
}

void WriteSnapshotsFunctor::operator()(string filePath, char label)
{
	// Start processing with base images.
	string substr("_image.jpg");
	size_t pos;
	if ( (pos = filePath.find(substr)) != string::npos ) 
	{
		// Trim "_image.jpg" and replace with _quality.img.
		string qualityPath = filePath;
		qualityPath.replace(pos, substr.length(), "_quality.jpg");
		if (!FileExists(qualityPath.c_str()))
		{
			cerr << "ERROR: Missing file: " << qualityPath << endl;
			return;
		}

		// Read image and quality matrices in and perform pre-processing.
		Mat readImage = imread(filePath);
		Mat grayImage;
		cvtColor(readImage, grayImage, CV_BGR2GRAY);
		Mat readQuality = imread(qualityPath, CV_LOAD_IMAGE_GRAYSCALE);
		GaussianBlur(readQuality, readQuality, Size(3, 3), 0.0);

		// Use adaptive canny edge detection and distance transform to generate edge-based quality matrix.
		Mat edges;
		if (!_adaptiveCanny->Process(grayImage, edges))
			cerr << "ERROR: Adaptive canny failed." << endl;

		Mat distMatrix;
		distanceTransform(255 - edges, distMatrix, CV_DIST_L2, CV_DIST_MASK_PRECISE);
		ConvertDistanceToWeight(distMatrix);
		
		// Update focus weights if image size has changed since last time.
		if ( (_focusWeights.cols != readQuality.cols) ||
			 (_focusWeights.rows != readQuality.rows) )
			_focusWeights = GetFocusWeights(Rect(0, 0, readQuality.cols, readQuality.rows));

		// Calculate master quality matrix, incorporating:
		// 1) Color-based quality matrix (priority to pixels with hand-like colors)
		// 2) Edge-based quality matrix (priority to pixels near edges)
		// 3) Focus weights (priority to pixels near the center)
		Mat readQualityNorm;
		normalize(readQuality, readQualityNorm, 0.0, 1.0, CV_MINMAX, CV_32F);

		Mat masterMatrix = readQualityNorm.mul(distMatrix).mul(_focusWeights);
		Mat masterMatrixNorm;
		normalize(masterMatrix, masterMatrixNorm, 0.0, 255.0, CV_MINMAX, CV_8U);

		// Dump final matrix into input data vector and forward to base class for image multiplication and writing.
		vector<Mat> inputData;
		inputData.push_back(masterMatrixNorm);
		ExplodeAndWriteData(inputData, label);
	}
}


WriteClipsFunctor::WriteClipsFunctor(string dataFileName, string labelFileName) :
	WriteDataLabelFunctor(dataFileName, labelFileName, CLIP_IMAGE_SIZE)
{
}

void WriteClipsFunctor::operator()(string filePath, char label)
{
	// Adjust current label.
	if (label == 9) // J
		label = 1;
	else if (label == 15) // Q (Neither J nor Z)
		label = 0;
	else if (label == 24)
		label = 2; // Z

	// Start processing with first base image.
	string substr("1_image.jpg");
	size_t pos;
	if ( (pos = filePath.find(substr)) != string::npos ) 
	{
		string filePathPrefix = filePath.substr(0, pos);

		// Pre-process each of the images in the clip and load them into vector.
		vector<Mat> inputData;
		for (int i = 1; i <= CLIP_LENGTH; ++i)
		{
			ostringstream oss;
			oss << filePathPrefix << i << "_image.jpg";
			string imagePath = oss.str();
			if (!FileExists(imagePath.c_str()))
			{
				cerr << "ERROR: Missing file: " << imagePath << endl;
				return;
			}

			oss.str("");
			oss << filePathPrefix << i << "_quality.jpg";
			string qualityPath = oss.str();
			if (!FileExists(qualityPath.c_str()))
			{
				cerr << "ERROR: Missing file: " << qualityPath << endl;
				return;
			}

			// Read image and quality matrices in and perform pre-processing.
			Mat readImage = imread(imagePath);
			Mat grayImage;
			cvtColor(readImage, grayImage, CV_BGR2GRAY);
			Mat readQuality = imread(qualityPath, CV_LOAD_IMAGE_GRAYSCALE);
			GaussianBlur(readQuality, readQuality, Size(3, 3), 0.0);

			// Use adaptive canny edge detection and distance transform to generate edge-based quality matrix.
			Mat edges;
			if (!_adaptiveCanny->Process(grayImage, edges))
				cerr << "ERROR: Adaptive canny failed." << endl;

			Mat distMatrix;
			distanceTransform(255 - edges, distMatrix, CV_DIST_L2, CV_DIST_MASK_PRECISE);
			ConvertDistanceToWeight(distMatrix);
		
			// Update focus weights if image size has changed since last time.
			if ( (_focusWeights.cols != readQuality.cols) ||
				 (_focusWeights.rows != readQuality.rows) )
				_focusWeights = GetFocusWeights(Rect(0, 0, readQuality.cols, readQuality.rows));

			// Calculate master quality matrix, incorporating:
			// 1) Color-based quality matrix (priority to pixels with hand-like colors)
			// 2) Edge-based quality matrix (priority to pixels near edges)
			// 3) Focus weights (priority to pixels near the center)
			Mat readQualityNorm;
			normalize(readQuality, readQualityNorm, 0.0, 1.0, CV_MINMAX, CV_32F);

			Mat masterMatrix = readQualityNorm.mul(distMatrix).mul(_focusWeights);
			Mat masterMatrixNorm;
			normalize(masterMatrix, masterMatrixNorm, 0.0, 255.0, CV_MINMAX, CV_8U);

			// Dump final matrix into input data vector.
			inputData.push_back(masterMatrixNorm);
		}

		// Forward all input data to base class for image multiplication and writing.
		ExplodeAndWriteData(inputData, label);
	}
}


CalcStatsFunctor::CalcStatsFunctor() : 
	_imageCount(0)
{
	_totalMean = _totalStdDev = Scalar::all(0.0);
	_maxMean = _maxStdDev = Scalar::all(-9999999.0);
	_minMean = _minStdDev = Scalar::all(9999999.0);
}

void CalcStatsFunctor::operator()(string filePath, char label)
{
	if (label > 0)
		return;

	// Only processing base images.
	string substr("_image.jpg");
	size_t pos;

	if ( (pos = filePath.find(substr)) != string::npos ) 
	{
		Mat readImage = imread(filePath);

		// Examine tight bound ROI.
		Rect tightRect = GetBoundingBox(BOUND_TIGHT, Rect(0, 0, readImage.cols, readImage.rows), true);
		Mat tightImage = readImage(tightRect);

		// Looking for Y, Cr, Cb stats.
		Mat yccImage;
		cvtColor(tightImage, yccImage, CV_BGR2YCrCb);

		Scalar mean, stdDev;
		meanStdDev(yccImage, mean, stdDev);
		_totalMean += mean;
		_totalStdDev += stdDev;
		for (int i = 0; i < 3; ++i)
		{
			if (mean[i] > _maxMean[i])
				_maxMean[i] = mean[i];
			if (mean[i] < _minMean[i])
				_minMean[i] = mean[i];
			if (stdDev[i] > _maxStdDev[i])
				_maxStdDev[i] = stdDev[i];
			if (stdDev[i] < _minStdDev[i])
				_minStdDev[i] = stdDev[i];
		}
		++_imageCount;
	}
}

void CalcStatsFunctor::DisplayResults() const
{
	cout << "Statistics: Y, Cr, Cb:" << endl;
	Scalar mean = _totalMean / _imageCount;
	cout << "\tAverage mean: " << mean[0] << ", " << mean[1] << ", " << mean[2] << endl;
	Scalar stdDev = _totalStdDev / _imageCount;
	cout << "\tAverage stdDev: " << stdDev[0] << ", " << stdDev[1] << ", " << stdDev[2] << endl;
	cout << "\tMax mean: " << _maxMean[0] << ", " << _maxMean[1] << ", " << _maxMean[2] << endl;
	cout << "\tMin mean: " << _minMean[0] << ", " << _minMean[1] << ", " << _minMean[2] << endl;
	cout << "\tMax stdDev: " << _maxStdDev[0] << ", " << _maxStdDev[1] << ", " << _maxStdDev[2] << endl;
	cout << "\tMin stdDev: " << _minStdDev[0] << ", " << _minStdDev[1] << ", " << _minStdDev[2] << endl;
}
