#pragma once

class AdaptiveCanny
{
	static const double EDGE_FACTOR_EPSILON;

	double _lowThres, _highThres;
	double _lowEdgeFactor, _highEdgeFactor, _stepSize;
	bool _gradual;

public:
	AdaptiveCanny(double edgeFactor, double stepSize, bool gradual) :
		_lowThres(50.0), _highThres(100.0),
		_stepSize(stepSize), _gradual(gradual)
	{
		_lowEdgeFactor = edgeFactor * (1 - EDGE_FACTOR_EPSILON);
		_highEdgeFactor = edgeFactor * (1 + EDGE_FACTOR_EPSILON);
	};

	bool Process(const cv::Mat & image, cv::Mat & edges);

private:
	void CalcEdgeStats(const cv::Mat & edges, int & edgeCount, float & dispersion);
};
