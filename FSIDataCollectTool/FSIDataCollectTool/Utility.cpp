#include "stdafx.h"
#include "Utility.h"

using namespace cv;


BOOL FileExists(LPCTSTR szPath)
{
  DWORD dwAttrib = GetFileAttributes(szPath);

  return (dwAttrib != INVALID_FILE_ATTRIBUTES && 
         !(dwAttrib & FILE_ATTRIBUTE_DIRECTORY));
}

Rect GetBoundingBox(eBoundType boundType, const Rect & inputBB, bool relativeToInput)
{
	int xOff = 0;
	int yOff = 0;
	if (!relativeToInput)
	{
		xOff = inputBB.x;
		yOff = inputBB.y;
	}

	double offset;
	double factor;
	switch (boundType)
	{
	case BOUND_LOOSE:
		offset = -BOUND_LOOSE_OFFSET;
		factor = BOUND_LOOSE_FACTOR;
		break;

	case BOUND_TIGHT:
		offset = BOUND_TIGHT_OFFSET;
		factor = BOUND_TIGHT_FACTOR;
		break;

	case BOUND_FOCUS:
		offset = BOUND_FOCUS_OFFSET;
		factor = BOUND_FOCUS_FACTOR;
		break;
	}

	return Rect( xOff + static_cast<int>(inputBB.width * offset),
				 yOff + static_cast<int>(inputBB.height * offset),
				 static_cast<int>(inputBB.width * factor),
				 static_cast<int>(inputBB.height * factor) );
}

Mat GetFocusWeights(Rect & rect)
{
	// Build focus weights matrix following parabolic equation.
	// Used to give higher priority to pixels located closer to the center of the ROI.
	Mat focusWeights = Mat(rect.height, rect.width, CV_32FC1);
	double a, b, c, temp; // weight = a * sqr(x - b) + c
	b = rect.width / 2.0;
	a = -1.0 / (b * b);
	c = 1.0;
	Scalar val(0.0);
	int steps = (rect.width + 1) / 2;
	for (int i = 0; i < steps; ++i)
	{
		temp = i - b;
		val[0] = a * temp * temp + c;
		rectangle(focusWeights, rect, val);
		rect.x++;
		rect.y++;
		rect.width -= 2;
		rect.height -= 2;
	}

	return focusWeights;
}

void ConvertDistanceToWeight(Mat & distMatrix)
{
	// Translate raw distance-to-edge data into weight using parabolic equation.
	// Used to give higher priority to pixels located closer to an edge.
	int maxDist = distMatrix.rows / 30;
	float a = (-1.0f / (maxDist * maxDist));
	for (int i = 0; i < distMatrix.rows; ++i)
	{
		float * currRowPtr = distMatrix.ptr<float>(i);
		for (int j = 0; j < distMatrix.cols; ++j)
		{
			currRowPtr[j] = a * currRowPtr[j] * currRowPtr[j] + 1.0f;
			if (currRowPtr[j] < 0.0f)
				currRowPtr[j] = 0.0f;
		}
	}
}
