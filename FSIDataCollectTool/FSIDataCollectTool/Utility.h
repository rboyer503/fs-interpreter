#pragma once

#include "Globals.h"


BOOL FileExists(LPCTSTR szPath);
cv::Rect GetBoundingBox(eBoundType boundType, const cv::Rect & inputBB, bool relativeToInput);
cv::Mat GetFocusWeights(cv::Rect & rect);
void ConvertDistanceToWeight(cv::Mat & distMatrix);

/*
template <class Vector, class T>
void insert_into_vector(Vector & v, const T & t)
{
	typename Vector::iterator i = std::lower_bound(v.begin(), v.end(), t);
	v.insert(i, t);
}
*/
