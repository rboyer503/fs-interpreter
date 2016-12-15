#include "stdafx.h"
#include "ImageProcessCoordinator.h"
#include "DirProcessFunctors.h"
#include "AdaptiveCanny.h"
#include "Utility.h"
#include "Globals.h"

using namespace std;
using namespace cv;
namespace posixtime = boost::posix_time;


char * ImageProcessCoordinator::eIPCModeStrings[FINAL_eIPCMode_ENTRY] = { "CAM_AUTO_ADJUST", "FIND_HAND", "TRACK_HAND" };


ImageProcessCoordinator::ImageProcessCoordinator()
{
	SetMode(CAM_AUTO_ADJUST);
	_initTime = posixtime::microsec_clock::local_time();
	_snapshotLetter = _snapshotCount = 0;
	for (int i = 0; i < 26; ++i)
		_snapshotIndices[i] = 0;
	_isClip = false;

	// From data file statistics, 1/16/16:
	// Statistics: Y, Cr, Cb:
    //		Average mean: 110.491, 150.943, 109.697
    //		Average stdDev: 36.3768, 7.31453, 8.04819
    //		Max mean: 219.928, 175.689, 123.786
    //		Min mean: 48.8354, 136.6, 86.1239
    //		Max stdDev: 83.0585, 19.4679, 17.5821
    //		Min stdDev: 13.1801, 2.71406, 4.65383
	// We will tighten Y (intensity) requirements since false positives typically have a low mean intensity.
	_avgMean = Scalar(110.491 * 1.2, 150.943, 109.697);
	_avgStdDev = Scalar(36.3768 * 0.8, 7.31453, 8.04819);

	_adaptiveCanny = NULL;
}

ImageProcessCoordinator::~ImageProcessCoordinator()
{
	if (_adaptiveCanny)
		delete _adaptiveCanny;
}

bool ImageProcessCoordinator::Initialize()
{
	if (!_cascade.load(HAAR_MODEL_FILENAME))
	{ 
		cerr << "Error: Failure loading cascade classifier from " << HAAR_MODEL_FILENAME << "." << endl;
		return false;
	}
	_minNeighbors = 3;

	namedWindow("TEST1", CV_WINDOW_NORMAL);
	namedWindow("TEST2", CV_WINDOW_NORMAL);
	namedWindow("TEST3", CV_WINDOW_NORMAL);
	namedWindow("TEST4", CV_WINDOW_NORMAL);

	return true;
}

void ImageProcessCoordinator::ProcessFrame(Mat & inputFrame)
{
	switch (_mode)
	{
	case CAM_AUTO_ADJUST:
		{
			// Waiting for camera auto-adjust to complete.
			// After 5 seconds elapses, move to next mode.
			posixtime::time_duration msdiff = posixtime::microsec_clock::local_time() - _initTime;
			if (msdiff.total_seconds() >= 5)
				SetMode(FIND_HAND);
			break;
		}

	case FIND_HAND:
		{
			// Use cascade classifier to identify hand in the frame.
			Mat frameGray;
			cvtColor(inputFrame, frameGray, CV_BGR2GRAY);
			equalizeHist(frameGray, frameGray);

			vector<Rect> hands;
			_cascade.detectMultiScale(frameGray, hands, 1.05, _minNeighbors, 0, Size(80, 80));
			if (hands.size() > 1)
			{
				// Tighten cascade classifier's neighbor requirement if we're detecting more than one hand.
				_minNeighbors++;
			}
			else if (hands.size() == 1)
			{
				/// TODO: Further validation of hand detection (based on CNN response).

				// Basic hand candidate validation based on YCrCb ROI stats.
				Mat handROI = inputFrame(hands[0]);
				if (GetHandOutlierFactor(handROI, true) < MAX_OUTLIER_FACTOR)
				{
					// Initialize loose bounding box, clip to main window, and calculate weights for it.
					SetLooseBound(hands[0]);
					ClipLooseBound();
					_focusWeights = GetFocusWeights(Rect(0, 0, _looseBound.width, _looseBound.height));

					// Build look up table - perform gaussian filter first to ensure similarity with tracking frames.
					GaussianBlur(inputFrame, inputFrame, Size(5, 5), 0.0);
					BuildLUT(inputFrame, 7, 0.0);
				
					// Initialize adaptive canny object.
					_adaptiveCanny = new AdaptiveCanny(0.015, 2.0, true);

					SetMode(TRACK_HAND);
				}
			}

			break;
		}

	case TRACK_HAND:
		{
			// Limit processing to loose bounding box and reduce noise.
			Mat frameROI = inputFrame(_looseBound);
			GaussianBlur(frameROI, frameROI, Size(5, 5), 0.0);

			// Use LUT to generate color-based quality matrix to give priority to pixels with hand colors.
			Mat frameYCC;
			cvtColor(frameROI, frameYCC, CV_BGR2YCrCb);

			Mat lutMatrix;
			LUT(frameYCC, _lut, lutMatrix);

			vector<Mat> lutChannels;
			split(lutMatrix, lutChannels);
			Mat colorMatrix = lutChannels[0].mul(lutChannels[1]).mul(lutChannels[2]);
			GaussianBlur(colorMatrix, colorMatrix, Size(3, 3), 0.0);

			// Take snapshot if requested.
			if (_snapshotCount)
			{
				// Normalize color-based quality matrix and convert to grayscale image.
				Mat colorMatrixNorm;
				normalize(colorMatrix, colorMatrixNorm, 0.0, 255.0, CV_MINMAX, CV_8U);

				// Write raw image and color-based quality image.
				int snapIndex = _snapshotIndices[_snapshotLetter - 'a'];
				if (!_isClip || _snapshotCount == 1)
					_snapshotIndices[_snapshotLetter - 'a']++;					

				ostringstream oss;
				oss << _currDirName << "\\" << snapIndex;
				if (_isClip)
					oss << "_" << (9 - _snapshotCount);
				oss << "_image.jpg";
				imwrite(oss.str(), frameROI);
				
				oss.str("");
				oss << _currDirName << "\\" << snapIndex;
				if (_isClip)
					oss << "_" << (9 - _snapshotCount);
				oss << "_quality.jpg";
				imwrite(oss.str(), colorMatrixNorm);

				if (--_snapshotCount == 0)
					cout << "Capture(s) completed." << endl;
				else
					cout << "  " << _snapshotCount << " left..." << endl;
			}

			// Use adaptive canny edge detection and distance transform to generate edge-based quality matrix.
			Mat frameGray;
			cvtColor(frameROI, frameGray, CV_BGR2GRAY);

			Mat edges;
			if (!_adaptiveCanny->Process(frameGray, edges))
				cerr << "ERROR: Adaptive canny failed." << endl;
			imshow("TEST1", edges);

			Mat distMatrix;
			distanceTransform(255 - edges, distMatrix, CV_DIST_L2, CV_DIST_MASK_PRECISE);
			ConvertDistanceToWeight(distMatrix);

			// Normalize and convert distance matrix for debug purposes.
			Mat distMatrixNorm;
			normalize(distMatrix, distMatrixNorm, 0.0, 255.0, CV_MINMAX, CV_8U);
			imshow("TEST2", distMatrixNorm);

			// Calculate master quality matrix, incorporating:
			// 1) Color-based quality matrix (priority to pixels with hand-like colors)
			// 2) Edge-based quality matrix (priority to pixels near edges)
			// 3) Focus weights (priority to pixels near the center)
			Mat masterMatrix = colorMatrix.mul(distMatrix).mul(_focusWeights);
			Mat masterMatrixNorm;
			normalize(masterMatrix, masterMatrixNorm, 0.0, 255.0, CV_MINMAX);

			// Determine adjusted position of bounding box based on "weighted center of color mass".
			Mat reducedCols;
			reduce(masterMatrixNorm, reducedCols, 0, CV_REDUCE_SUM);

			Scalar_<float> centroidDiv = sum(reducedCols);
			if (centroidDiv[0] < 0.5f)
				centroidDiv[0] = 1.0f;
			float val = 0.0f;
			for (int i = 0; i < reducedCols.cols; ++i)
				val += reducedCols.at<float>(i) * i;
			val /= centroidDiv[0];

			_looseBound.x -= (_looseBound.width / 2 - static_cast<int>(val));

			Mat reducedRows;
			reduce(masterMatrixNorm, reducedRows, 1, CV_REDUCE_SUM);

			centroidDiv = sum(reducedRows);
			if (centroidDiv[0] < 0.5f)
				centroidDiv[0] = 1.0f;
			val = 0.0f;
			for (int i = 0; i < reducedRows.rows; ++i)
				val += reducedRows.at<float>(i) * i;
			val /= centroidDiv[0];

			_looseBound.y -= (_looseBound.height / 2 - static_cast<int>(val));
			_looseBound.y -= (_looseBound.height / 20);

			ClipLooseBound();

			// Display master quality matrix as a grayscale image.
			Mat masterMatrixDisplay;
			masterMatrixNorm.convertTo(masterMatrixDisplay, CV_8U);
			imshow("TEST3", masterMatrixDisplay);
			
			// DEBUG: Process to target features for CNN for real-time display.
			// Zoom to focus bounds and resize.
			Rect focusRect = GetBoundingBox(BOUND_FOCUS, _looseBound, true);
			Mat focusMaster = masterMatrixDisplay(focusRect);
			Mat resizedMaster;
			resize(focusMaster, resizedMaster, Size(IMAGE_SIZE, IMAGE_SIZE));
			imshow("TEST4", resizedMaster);
			
			// Draw bounding boxes.
			rectangle(inputFrame, _looseBound, Scalar(255, 255, 255), 2);
			rectangle(inputFrame, GetBoundingBox(BOUND_TIGHT, _looseBound, false), Scalar(255, 255, 255), 2);

			break;
		}
	}
}

void ImageProcessCoordinator::StartCapture(char letter, int numSnapshots, bool clip)
{
	// Don't interrupt existing operation.
	if (_snapshotCount == 0)
	{
		// Track snapshot parameters.
		_snapshotLetter = letter;
		_snapshotCount = numSnapshots;
		_isClip = clip;

		// Create directory if needed.
		_currDirName = ".\\TempData\\";
		_currDirName.append(1, _snapshotLetter);
		CreateDirectory(_currDirName.c_str(), NULL);

		cout << "Starting " << (clip ? "clip" : "snapshot") << " capture(s) for letter '" << letter << "'." << endl;
	}
}

void ImageProcessCoordinator::WriteSnapshotFiles()
{
	cout << "Starting snapshot data and label file generation." << endl;
	WriteSnapshotsFunctor snapshotsFunctor(DATA_RAW_FILENAME, LABEL_RAW_FILENAME);
	ProcessDirectory(".\\data", '\0', snapshotsFunctor);
	cout << "File generation complete." << endl;
}

void ImageProcessCoordinator::WriteClipFiles()
{
	cout << "Starting clip data and label file generation." << endl;
	WriteClipsFunctor clipsFunctor(CLIP_DATA_RAW_FILENAME, CLIP_LABEL_RAW_FILENAME);
	ProcessDirectory(".\\clipData", '\0', clipsFunctor);
	cout << "File generation complete." << endl;
}

void ImageProcessCoordinator::CalcDataFileStats()
{
	cout << "Starting data file statistics calculations." << endl;
	CalcStatsFunctor csFunctor;
	ProcessDirectory(".\\data", '\0', csFunctor);
	cout << "Data file statistics calculations complete." << endl;
	csFunctor.DisplayResults();
}

void ImageProcessCoordinator::SetMode(eIPCMode mode)
{
	_mode = mode;
	cout << "MODE: " << eIPCModeStrings[mode] << endl;
}

double ImageProcessCoordinator::GetHandOutlierFactor(const cv::Mat & handROI, bool debugOutput)
{
	// Calculate YCrCb stats on candidate ROI.
	Mat handYCC;
	cvtColor(handROI, handYCC, CV_BGR2YCrCb);

	Scalar mean, stdDev;
	meanStdDev(handYCC, mean, stdDev);

	double outlierFactor = 0.0;
	for (int i = 0; i < 3; ++i)
	{
		double offset = (mean[i] - _avgMean[i]) / _avgStdDev[i];
		outlierFactor += abs(offset);
	}

	if (debugOutput)
	{
		cout << "Hand candidate statistics:" << endl;
		cout << "\tROI size: " << handROI.cols << endl;
		cout << "\tMean: " << mean[0] << ", " << mean[1] << ", " << mean[2] << endl;
		cout << "\tStdDev: " << stdDev[0] << ", " << stdDev[1] << ", " << stdDev[2] << endl;
		cout << "\tOutlier factor: " << outlierFactor << endl;
	}

	return outlierFactor;
}

void ImageProcessCoordinator::SetLooseBound(const Rect & handBound)
{
	_looseBound = GetBoundingBox(BOUND_LOOSE, handBound, false);
}

void ImageProcessCoordinator::ClipLooseBound()
{
	if (_looseBound.x < 0)
		_looseBound.x = 0;
	else if (_looseBound.x + _looseBound.width > FRAME_WIDTH)
		_looseBound.x = FRAME_WIDTH - _looseBound.width;

	if (_looseBound.y < 0)
		_looseBound.y = 0;
	else if (_looseBound.y + _looseBound.height > FRAME_HEIGHT)
		_looseBound.y = FRAME_HEIGHT - _looseBound.height;
}

void ImageProcessCoordinator::BuildLUT(const cv::Mat & frame, int smoothFactor, double thresholdFactor)
{
	// Using YCC color space for LUT.
	Mat frameYCC;
	cvtColor(frame, frameYCC, CV_BGR2YCrCb);

	// Get ROI - use tight bounding box to ensure ROI is entirely hand.
	Rect tightBound = GetBoundingBox(BOUND_TIGHT, _looseBound, false);
	Mat frameROI = frameYCC(tightBound);

	// Compute histograms for each channel, on both the full image and ROI.
	vector<Mat> fullHist;
	BuildHistograms(frameYCC, fullHist);

	vector<Mat> roiHist;
	BuildHistograms(frameROI, roiHist);

	// Rescale histograms, difference them, and normalize the result to create LUTs for each channel.
	vector<Mat> fullHistNormalized;
	vector<Mat> histDiff;
	vector<Mat> luts;
	double scaleFactor = tightBound.area() / static_cast<double>(frame.rows * frame.cols);
	for (int i = 0; i < frame.channels(); ++i)
	{
		// Resize full image histogram to bring into scale with ROI histogram.
		fullHistNormalized.push_back(fullHist[i] * scaleFactor);

		// Find difference between full image and ROI histograms.
		// Entries with large positive values correspond to channel values commonly found in the ROI, but not the full image.
		histDiff.push_back(roiHist[i] - fullHistNormalized[i]);

		// Smooth the difference histogram to reduce noise and avoid overfitting.
		blur(histDiff[i], histDiff[i], Size(1, smoothFactor));

		// Normalize difference histogram to build LUT.
		Mat lutMat;
		normalize(histDiff[i], lutMat, 0.0, 1.0, NORM_MINMAX);
		luts.push_back(lutMat);
	}

	// Combine normalized difference histograms into the merged LUT.
	merge(luts, _lut);
}

void ImageProcessCoordinator::BuildHistograms(const cv::Mat & frame, vector<Mat> & histograms)
{
	// Split into individual channels.
	vector<Mat> channels;
	split(frame, channels);

	// Generate histogram for each channel.
	const int histSize[] = {256};
	const float range[] = {0, 256};
	const float * histRange[] = {range};

	for (vector<Mat>::iterator it = channels.begin(); it < channels.end(); ++it)
	{
		Mat currHist;
		calcHist(&(*it), 1, NULL, Mat(), currHist, 1, histSize, histRange);
		histograms.push_back(currHist);
	}
}

void ImageProcessCoordinator::ProcessDirectory(string dir, char currLabel, DirProcessFunctor & functor) // ofstream & ofData, ofstream & ofLabels)
{
	cout << "Processing directory: " << dir << endl;

	// Look for all contents:
	string dirQuery = dir + "\\*.*";

	WIN32_FIND_DATA fdFile;
    HANDLE hFind;
	if ( (hFind = FindFirstFile(dirQuery.c_str(), &fdFile)) == INVALID_HANDLE_VALUE )
	{
		cerr << "Error: Path '" << dir.c_str() << "' not found." << endl;
		return;
	}

	// Process each file and directory.
	do
    {
        //  Ignore . and ..
        if ( (strcmp(fdFile.cFileName, ".") != 0) && 
			 (strcmp(fdFile.cFileName, "..") != 0) )
        {
            // Build full path.
			string fullPath = dir + "\\" + fdFile.cFileName;

            if (fdFile.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
            {
				// Check for special "a" - "z" directories.
				if (strlen(fdFile.cFileName) == 1)
				{
					if ( (fdFile.cFileName[0] >= 'a') &&
						 (fdFile.cFileName[0] <= 'z') )
					{
						// Update current label appropriately - skip 'j'.
						currLabel = (fdFile.cFileName[0] - 'a');
						if (fdFile.cFileName[0] > 'j')
							--currLabel;
					}
				}

				// Recursively process sub-directory.
                ProcessDirectory(fullPath, currLabel, functor);
            }
            else
			{
				// Pass file and current label to specified functor.
				functor(fullPath, currLabel);
            }
        }
    }
    while (FindNextFile(hFind, &fdFile)); 

	FindClose(hFind);
}
