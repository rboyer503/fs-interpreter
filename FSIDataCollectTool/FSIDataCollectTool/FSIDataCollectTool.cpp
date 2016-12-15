// FSInterpreter.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "ImageProcessCoordinator.h"
#include "Globals.h"

using namespace std;
using namespace cv;
namespace posixtime = boost::posix_time;


struct AppState
{
	bool done;			// If true: exit app
	bool paused;		// If true: suspend image processing
	bool singleMode;	// If true: capture individual images instead of bursts
	bool jzMode;		// If true: capture clips of 'j', 'z', or neither instead of static letters
	ifstream ifData;	// File stream for viewing/navigating raw data file
	int numImages;		// Number of images in data file
	int currImageIndex; // Current index of image in data file
	int seqSize;		// Number of snapshots to collect in sequence
	int seqLeft;		// Number of snapshots left to collect
	int seqCount;		// Remaining times to interate through seqSize frames
						// (Beep sound plays for each count)
	char captureKey;	// Key to capture for sequence

	AppState() :
		done(false), paused(false), singleMode(false), jzMode(false),
		numImages(0), currImageIndex(0),
		seqSize(0), seqLeft(0), seqCount(0), captureKey(0)
	{}
};


void usage();
bool validateIntParam(const char * param, int & output, int min, int max);
void displayMenu();
int processVideo(VideoCapture & capture);


int _tmain(int argc, _TCHAR* argv[])
{
	if (argc != 2)
	{
		cerr << "Error: Wrong number of arguments." << endl;
        usage();
        return 1;
    }

	int deviceID;
	if (!validateIntParam(argv[1], deviceID, 0, 9))
	{
		cerr << "Error: Invalid device ID." << endl;
		usage();
		return 1;
	}

	// Establish video capture for specified device.
	VideoCapture capture(deviceID);
	if (!capture.isOpened())
	{
        cerr << "Error: Failed to open video device." << endl;
        usage();
        return 1;
    }

	// Ensure auto-exposure/gain are active so we quickly arrive at reasonable values.
	// Exposure can manually be tuned as needed using +/- keys.
	// Note that this code relies on a custom modification to OpenCV's DirectShow videoInput class.
	// (I could find no other way to reliably restore auto-exposure/gain after manually changing the settings.)
	cout << "Initial exposure: " << capture.get(CV_CAP_PROP_EXPOSURE) << endl;
	if (!capture.set(CV_CAP_PROP_EXPOSURE, 0.0))
	{
		cerr << "ERROR: Couldn't activate auto-exposure/gain." << endl;
	}

	// Create main window for displaying raw video feed.
	namedWindow(MAIN_WINDOW_NAME);

	return processVideo(capture);
}

void usage()
{
	cout << "Usage: FSIDataCollectTool <device ID>" << endl;
	cout << "       <device ID>: ID of video device (0-9)" << endl;
}

bool validateIntParam(const char * param, int & output, int min, int max)
{
	// Integer parameter validation with range checking.
	istringstream iss(param);
	if (!(iss >> output))
		return false;

	if ( (output < min) || (output > max) )
		return false;

	return true;
}

void displayMenu()
{
	cout << "Fingerspelling Interpreter Data Collection Tool" << endl;
	cout << "\tESC:    Exit" << endl;
	cout << "\tSPACE:  Pause" << endl;
	cout << "\t?:      Display this menu" << endl;
	cout << "\t!:      Toggle single image capture mode" << endl;
	cout << "\t@:      Toggle J/Z capture mode" << endl;
	cout << "\t[a..z]: Capture " << SMALL_BURST_SIZE << " snapshots" << endl;
	cout << "\t[A..Z]: Capture " << LARGE_BURST_SIZE << " snapshots" << endl; 
	cout << "\t1:      Generate data and label files" << endl;
	cout << "\t2:      Load data file" << endl;
	cout << "\t3:      Split data and label files" << endl;
	cout << "\t4:      Calculate data file statistics" << endl;
	cout << "\t</>:    Move back/forward one image" << endl;
	cout << "\t[/]:    Move back/forward one letter" << endl;
	cout << "\t{/}:    Move back/forward one collection" << endl;
	cout << "\t+/-:    Increase/decrease camera exposure" << endl << endl;
}

int processVideo(VideoCapture & capture)
{
	// Create Data Collection Viewer window for visualizing data files.
	namedWindow(DCV_WINDOW_NAME, CV_WINDOW_NORMAL);

	// Initialize object managing all image processing.
	ImageProcessCoordinator ipc;
	if (!ipc.Initialize())
		return 1;

	displayMenu();

	AppState state;
	while (!state.done)
	{
		Mat frame;
		posixtime::time_duration msdiff;

		if (!state.paused)
		{
			// Grab frame from video device.
			posixtime::ptime start = posixtime::microsec_clock::local_time();
			capture >> frame;
			if (frame.empty())
				break;
			//msdiff = posixtime::microsec_clock::local_time() - start;
			//cout << "Profile 1 (grabbed):    " << msdiff.total_milliseconds() << " ms" << endl;
			
			// Perform all frame processing.
			ipc.ProcessFrame(frame);
			//msdiff = posixtime::microsec_clock::local_time() - start;
			//cout << "Profile 2 (processed):  " << msdiff.total_milliseconds() << " ms" << endl;

			// Display frame.
			imshow(MAIN_WINDOW_NAME, frame);
			msdiff = posixtime::microsec_clock::local_time() - start;
			//cout << "Profile 3 (shown):      " << msdiff.total_milliseconds() << " ms" << endl;
		}

		// Adjust waitKey delay to hold frame rate roughly constant.
		int delay = static_cast<int>(msdiff.total_milliseconds());
		if (delay < MIN_FRAME_LENGTH_MS)
			delay = MIN_FRAME_LENGTH_MS - delay;
		else
			delay = 1;

		// Handle user input.
		int moveOffset = 0;
		char key = static_cast<char>(waitKey(delay));
		if (key == 27) // ESC
			state.done = true;
		else if (key == 32) // SPACE
		{
			state.paused = !state.paused;
			cout << "[" << (state.paused ? "PAUSED" : "UNPAUSED") << "]" << endl;
		}
		else if (key == '?')
			displayMenu();
		else if (key == '!')
		{
			state.singleMode = !state.singleMode;
			cout << "[" << (state.singleMode ? "SINGLE MODE" : "BURST MODE") << "]" << endl;
		}
		else if (key == '@')
		{
			state.jzMode = !state.jzMode;
			cout << "[" << (state.jzMode ? "J/Z MODE" : "NORMAL MODE") << "]" << endl;
		}
		else if ( (key >= 'a') && (key <= 'z') )
		{
			// Capture small burst or single snapshot.
			if (!state.jzMode)
			{
				// Normal mode: capture 'a'-'z', ignoring 'j'.
				// ('z' is sentinel symbol, not actual z.)
				if (key != 'j')
				{
					// For single mode, immediately capture individual snapshot.
					if (state.singleMode)
						ipc.StartCapture(key, 1, false);
					else
					{
						// For burst mode, prepare for sequence.
						state.captureKey = key;
						state.seqSize = SMALL_BURST_SIZE;
						state.seqLeft = 0;
						state.seqCount = BEEP_COUNT;
					}
				}
			}
			else
			{
				// Capture clip of 'j', 'z', or neither ('q').
				if ( (key == 'j') || (key == 'z') || (key == 'q') )
				{
					state.captureKey = key;
					state.seqSize = CLIP_LENGTH;
					state.seqLeft = 0;
					state.seqCount = BEEP_COUNT;
				}
			}
		}
		else if ( (key >= 'A') && (key <= 'Z') )
		{
			// Capture large burst or single snapshot.
			if (state.singleMode)
				ipc.StartCapture(tolower(key), 1, false);
			else if (!state.jzMode)
			{
				// Burst mode, prepare for sequence.
				state.captureKey = tolower(key);
				state.seqSize = LARGE_BURST_SIZE;
				state.seqLeft = 0;
				state.seqCount = BEEP_COUNT;
			}
		}
		else if (key == '1')
		{
			// Generate data and label files.
			if (state.jzMode)
				ipc.WriteClipFiles();
			else
				ipc.WriteSnapshotFiles();
		}
		else if (key == '2')
		{
			// Reset ifstream for reuse.
			if (state.ifData.is_open())
			{
				state.ifData.close();
				state.ifData.clear();
			}

			// Determine filename and image size based on J/Z mode.
			const char * fileName;
			int imageSize;
			if (state.jzMode)
			{
				fileName = CLIP_DATA_RAW_FILENAME;
				imageSize = CLIP_IMAGE_SIZE;
			}
			else
			{
				fileName = DATA_RAW_FILENAME;
				imageSize = IMAGE_SIZE;
			}

			// Open data file and calculate number of images.
			state.ifData.open(fileName, ifstream::binary);
			state.ifData.seekg(0, state.ifData.end);
			state.numImages = static_cast<int>((state.ifData.tellg() / (imageSize * imageSize)));
			state.ifData.seekg(0, state.ifData.beg);
			state.currImageIndex = 0;
		}
		else if (key == '3')
		{
			// Split data and label files.
			if (!state.jzMode)
			{
				// Open file streams for input and output files.
				ifstream ifInData(DATA_RAW_FILENAME, ifstream::binary);
				ifstream ifInLabels(LABEL_RAW_FILENAME, ifstream::binary);

				ofstream ofOutData[NUM_SPLIT_FILES];
				ofstream ofOutLabels[NUM_SPLIT_FILES];
				ostringstream oss;
				for (int i = 0; i < NUM_SPLIT_FILES; ++i)
				{
					oss.str("");
					oss << "data" << (i + 1) << ".raw";
					ofOutData[i].open(oss.str(), ofstream::binary);

					oss.str("");
					oss << "labels" << (i + 1) << ".raw";
					ofOutLabels[i].open(oss.str(), ofstream::binary);
				}

				// Distribute input records to output files.
				char rawData[IMAGE_SIZE * IMAGE_SIZE];
				char rawLabel[1];
				int index = 0;
				while (ifInData.read(rawData, IMAGE_SIZE * IMAGE_SIZE))
				{
					ifInLabels.read(rawLabel, 1);
					ofOutData[index].write(rawData, IMAGE_SIZE * IMAGE_SIZE);
					ofOutLabels[index].write(rawLabel, 1);

					index = ((index + 1) % NUM_SPLIT_FILES);
				}

				// Close output files.
				for (int i = 0; i < NUM_SPLIT_FILES; ++i)
				{
					ofOutData[i].close();
					ofOutLabels[i].close();
				}

				ifInData.close();
				ifInLabels.close();
			}
		}
		else if (key == '4')
			ipc.CalcDataFileStats();
		else if (key == '<')
			moveOffset = -1;
		else if (key == '>')
			moveOffset = 1;
		else if (key == '[')
		{
			if (state.jzMode)
				moveOffset = -CLIP_LENGTH;
			else
				moveOffset = (state.singleMode ? -NUM_SAMPLES_PER_CAPTURE : -NUM_SAMPLES_PER_LETTER);
		}
		else if (key == ']')
		{
			if (state.jzMode)
				moveOffset = CLIP_LENGTH;
			else
				moveOffset = (state.singleMode ? NUM_SAMPLES_PER_CAPTURE : NUM_SAMPLES_PER_LETTER);
		}
		else if (key == '{')
			moveOffset = (state.singleMode ? (-NUM_SAMPLES_PER_CAPTURE * NUM_LETTERS)
										   : (-NUM_SAMPLES_PER_LETTER * NUM_LETTERS));
		else if (key == '}')
			moveOffset = (state.singleMode ? (NUM_SAMPLES_PER_CAPTURE * NUM_LETTERS) 
										   : (NUM_SAMPLES_PER_LETTER * NUM_LETTERS));
		else if (key == '-')
			capture.set(CV_CAP_PROP_EXPOSURE, capture.get(CV_CAP_PROP_EXPOSURE) + 1.0);
		else if (key == '+')
			capture.set(CV_CAP_PROP_EXPOSURE, capture.get(CV_CAP_PROP_EXPOSURE) - 1.0);

		// Handle navigation through loaded data file.
		if (moveOffset)
		{
			// Adjust move offset to bound at first and last image in file.
			int newImageIndex = state.currImageIndex + moveOffset;
			if (newImageIndex < 0)
				moveOffset -= newImageIndex;
			else if (newImageIndex >= state.numImages)
				moveOffset -= (newImageIndex - state.numImages + 1);

			// Only proceed if user didn't attempt to go before the beginning or after the end.
			if (moveOffset)
			{
				state.currImageIndex += moveOffset;
				char rawData[IMAGE_SIZE * IMAGE_SIZE];
				int imageSize = IMAGE_SIZE;
				if (state.jzMode)
					imageSize = CLIP_IMAGE_SIZE;

				// Seek as needed to the appropriate record in file.
				// Decrement move offset because each read automatically moves us one record forward.
				// (If we're moving 1 record forward, normal sequential reading is appropriate, so no need to seek.)
				if (--moveOffset)
				{
					if (newImageIndex <= 0)
						state.ifData.seekg(0, state.ifData.beg);
					else
						state.ifData.seekg(moveOffset * imageSize * imageSize, state.ifData.cur);
				}

				// Read record and display it.
				state.ifData.read(rawData, imageSize * imageSize);
				Mat currImage(imageSize, imageSize, CV_8U, rawData);
				imshow(DCV_WINDOW_NAME, currImage);	
			}
		}

		// The following logic is used to time "sequences" in which we collect several snapshots.
		// In order to make it easier for the user, we play a beep sound seqCount number of times.
		// After the second to last beep we initiate capturing.
		// Capture will complete before the final beep.
		// With BEEP_COUNT = 4:
		// ... beep ... beep ... beep ... <capture> ... beep
		if (state.seqCount)
		{
			if (!state.seqLeft)
			{
				// Starting a new pass...
				PlaySound(TEXT("beep.wav"), NULL, SND_FILENAME | SND_ASYNC);
				if (--state.seqCount)
					state.seqLeft = state.seqSize;
			}
			else
				--state.seqLeft;

			// The seqLeft test here is to tune it so that actual capture starts when the beep
			// actually plays.  (It's not immediate after the call to PlaySound.)
			if ( (state.seqCount == 1) && (state.seqLeft == (state.seqSize - 4)) )
				ipc.StartCapture(state.captureKey, state.seqSize, state.jzMode);
		}
	}

	// Restore auto-exposure/gain:
	if (!capture.set(CV_CAP_PROP_EXPOSURE, 0.0))
	{
		cerr << "ERROR: Couldn't restore auto-exposure/gain." << endl;
	}

	return 0;
}
