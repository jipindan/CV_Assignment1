#include "Utilities.h"
#include "opencv2/core/utils/filesystem.hpp"

namespace {

// AllBalls trains, Table1-6 test
const int NUMBER_OF_IMAGES = 7;
const char* FILE_LOCATION = "Media/Pool/";
const char* RESULTS_LOCATION = "Results/";
const char* IMAGE_NAMES[NUMBER_OF_IMAGES] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

const int NUMBER_OF_BALLS = 16;
const char* BALL_NAMES[NUMBER_OF_BALLS] = {
	"W", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };

// 1 pixel = 1 mm
const int TABLE_COLS = 1434;
const int TABLE_ROWS = 683;
const double BALL_RADIUS = 21.5;

// Table corners (x, y)
const Point2f TABLE_CORNERS[4] = { Point2f(337, 35), Point2f(1570, 21), Point2f(1888, 451), Point2f(38, 472) };

// Ground truth (row, column), -1 = off the table
const int GROUND_TRUTH[NUMBER_OF_IMAGES][NUMBER_OF_BALLS][2] = {
	{ {293,797}, {253,908}, {250,959}, {255,1013}, {256,1072}, {254,1132}, {262,1184}, {256,1244}, {253,1307}, {329,901}, {330,961}, {335,1013}, {336,1067}, {334,1130}, {345,1187}, {342,1245} },
	{ {288,29}, {309,1092}, {429,1212}, {383,1131}, {335,1050}, {285,1211}, {261,1170}, {352,1173}, {331,1132}, {240,1210}, {310,1170}, {407,1169}, {281,1135}, {374,1212}, {360,1091}, {331,1213} },
	{ {507,1169}, {282,1103}, {620,1224}, {391,1121}, {315,1046}, {310,1305}, {202,621}, {361,1173}, {331,1132}, {177,725}, {185,1269}, {665,798}, {296,438}, {53,1235}, {362,1090}, {399,1205} },
	{ {434,29}, {23,908}, {616,1224}, {447,1199}, {336,29}, {305,1406}, {210,621}, {260,1294}, {226,1252}, {177,725}, {171,1270}, {665,798}, {249,351}, {53,1233}, {169,901}, {399,1201} },
	{ {88,1392}, {23,908}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {210,621}, {260,1294}, {226,1252}, {181,724}, {171,1270}, {665,798}, {249,351}, {58,1231}, {169,901}, {399,1201} },
	{ {608,26}, {535,23}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294}, {226,1252}, {538,72}, {171,1270}, {665,798}, {249,351}, {-1,-1}, {169,901}, {399,1201} },
	{ {629,947}, {535,23}, {477,1364}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294}, {226,1252}, {538,72}, {171,1270}, {-1,-1}, {249,351}, {-1,-1}, {169,901}, {399,1201} } };

// About a quarter of a ball
const int MINIMUM_BALL_AREA = 400;

// Cloth histogram bins
const int HUE_BINS = 18;
const int SATURATION_BINS = 16;

// Background difference threshold
const int DIFFERENCE_THRESHOLD = 30;

// Hough, accumulator threshold tried from first to last
const int HOUGH_SMOOTHING_SIZE = 5;
const double HOUGH_MIN_CENTRE_DISTANCE = 30;
const double HOUGH_CANNY_THRESHOLD = 100;
const int HOUGH_FIRST_ACCUMULATOR_THRESHOLD = 5;
const int HOUGH_LAST_ACCUMULATOR_THRESHOLD = 40;
const int HOUGH_MIN_RADIUS = 17;
const int HOUGH_MAX_RADIUS = 31;

// k-means clusters
const int KMEANS_CLUSTERS = 4;

// Watershed
const int WATERSHED_SMOOTHING_SIZE = 7;
const double WATERSHED_CANNY_LOW = 100;
const double WATERSHED_CANNY_HIGH = 200;
const double WATERSHED_MARKER_DISTANCE = 2.0;
const int WATERSHED_MAX_AREA = 3000;

// Ball histogram, HSV 12 x 4 x 4
const int BALL_HISTOGRAM_CHANNELS[3] = { 0, 1, 2 };
const int BALL_HISTOGRAM_SIZES[3] = { 12, 4, 4 };
const float BALL_HUE_RANGE[2] = { 0, 180 };
const float BALL_SATURATION_RANGE[2] = { 0, 256 };
const float BALL_VALUE_RANGE[2] = { 0, 256 };
const float* BALL_HISTOGRAM_RANGES[3] = { BALL_HUE_RANGE, BALL_SATURATION_RANGE, BALL_VALUE_RANGE };
// White pixel, failure analysis only
const int WHITE_MAX_SATURATION = 59;
const int WHITE_MIN_VALUE = 151;

// Save into Results
void Save(const string& folder, const string& name, const Mat& image){
	string directory = string(RESULTS_LOCATION) + folder;
	utils::fs::createDirectories(directory);
	string extension = (image.channels() == 3) ? ".jpg" : ".png";
	imwrite(directory + "/" + name + extension, image);
}

// Ball on this table
bool BallOnTable(int image_index, int ball){
	return GROUND_TRUTH[image_index][ball][0] >= 0;
}

// Ground truth as (x, y)
Point2f GroundTruthPosition(int image_index, int ball){
	return Point2f((float)GROUND_TRUTH[image_index][ball][1], (float)GROUND_TRUTH[image_index][ball][0]);
}

// Draw the ground truth
Mat DrawGroundTruth(const Mat& table_image, int image_index){
	Mat display = table_image.clone();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		if (!BallOnTable(image_index, ball)) continue;
		Point2f p = GroundTruthPosition(image_index, ball);
		circle(display, p, (int)BALL_RADIUS, Scalar(0, 255, 255), 2);
		putText(display, BALL_NAMES[ball], p + Point2f(24, -8), FONT_HERSHEY_SIMPLEX, 0.7, Scalar(0, 255, 255), 2);
	}
	return display;
}

// Closing then opening
Mat CloseAndOpen(const Mat& binary, int element_size, const string& folder, const string& name){
	Mat element(element_size, element_size, CV_8U, Scalar(1));
	Mat cleaned;
	morphologyEx(binary, cleaned, MORPH_CLOSE, element);
	morphologyEx(cleaned, cleaned, MORPH_OPEN, element);
	Save(folder, name + " closed and opened", cleaned);
	return cleaned;
}

// Region centres, holes and small regions ignored
vector<Point2f> RegionCentres(const Mat& binary, const string& name){
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(binary, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
	vector<Point2f> centres;
	int regions = 0;
	for (size_t i = 0; i < contours.size(); i++) {
		if (hierarchy[i][3] >= 0) continue;
		regions++;
		Moments m = moments(contours[i]);
		if (m.m00 < MINIMUM_BALL_AREA) continue;
		centres.push_back(Point2f((float)(m.m10 / m.m00), (float)(m.m01 / m.m00)));
	}
	printf("%-8s %3d regions, %3d large enough\n", name.c_str(), regions, (int)centres.size());
	return centres;
}

// Method 1: cloth histogram from AllBalls without the balls
MatND TableHistogram(const Mat& table_image){
	Mat hls_image;
	cvtColor(table_image, hls_image, COLOR_BGR2HLS);
	Mat cloth_mask(table_image.size(), CV_8U, Scalar(255));
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		circle(cloth_mask, GroundTruthPosition(0, ball), (int)(2 * BALL_RADIUS), Scalar(0), FILLED);
	Save("1_backprojection", "AllBalls cloth sample", cloth_mask);
	cout << "Cloth sample: " << countNonZero(cloth_mask) << " pixels" << endl;

	int channels[] = { 0, 2 };
	int sizes[] = { HUE_BINS, SATURATION_BINS };
	float hue_range[] = { 0, 180 };
	float saturation_range[] = { 0, 256 };
	const float* ranges[] = { hue_range, saturation_range };
	MatND histogram;
	calcHist(&hls_image, 1, channels, cloth_mask, histogram, 2, sizes, ranges);
	normalize(histogram, histogram, 0, 255, NORM_MINMAX);
	return histogram;
}

// Back projection, Otsu, white = not cloth
Mat BackProjectionMask(const Mat& table_image, const MatND& histogram, const string& name){
	Mat hls_image;
	cvtColor(table_image, hls_image, COLOR_BGR2HLS);
	int channels[] = { 0, 2 };
	float hue_range[] = { 0, 180 };
	float saturation_range[] = { 0, 256 };
	const float* ranges[] = { hue_range, saturation_range };
	Mat probabilities;
	calcBackProject(&hls_image, 1, channels, histogram, probabilities, ranges);
	Save("1_backprojection", name + " probabilities", probabilities);

	Mat binary;
	double otsu = threshold(probabilities, binary, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);
	Save("1_backprojection", name + " binary", binary);
	printf("%-8s Otsu threshold %5.1f\n", name.c_str(), otsu);
	return CloseAndOpen(binary, 5, "1_backprojection", name);
}

// Method 2: difference from AllBalls
Mat BackgroundMask(const Mat& table_image, const Mat& background, const string& name){
	Mat difference;
	absdiff(table_image, background, difference);
	cvtColor(difference, difference, COLOR_BGR2GRAY);
	Save("2_background", name + " difference", difference);
	Mat binary;
	threshold(difference, binary, DIFFERENCE_THRESHOLD, 255, THRESH_BINARY);
	Save("2_background", name + " binary", binary);
	return CloseAndOpen(binary, 3, "2_background", name);
}

// Original image to table, off-table points dropped
vector<Point2f> PointsOnTable(const vector<Point2f>& image_points, const Mat& homography){
	vector<Point2f> table_points, on_table;
	if (!image_points.empty()) perspectiveTransform(image_points, table_points, homography);
	for (size_t i = 0; i < table_points.size(); i++)
		if (table_points[i].x >= 0 && table_points[i].x < TABLE_COLS && table_points[i].y >= 0 && table_points[i].y < TABLE_ROWS)
			on_table.push_back(table_points[i]);
	return on_table;
}

// Grey with Gaussian smoothing
Mat SmoothedGrey(const Mat& image){
	Mat grey;
	cvtColor(image, grey, COLOR_BGR2GRAY);
	GaussianBlur(grey, grey, Size(HOUGH_SMOOTHING_SIZE, HOUGH_SMOOTHING_SIZE), 0);
	return grey;
}

// Method 3: Hough circles, centres in table coordinates
vector<Point2f> LocateByHough(const Mat& image, const Mat& homography, double accumulator_threshold, vector<Vec3f>& circles_on_table){
	vector<Vec3f> circles;
	HoughCircles(SmoothedGrey(image), circles, HOUGH_GRADIENT, 1, HOUGH_MIN_CENTRE_DISTANCE, HOUGH_CANNY_THRESHOLD,
		accumulator_threshold, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS);
	vector<Point2f> detections;
	circles_on_table.clear();
	for (size_t i = 0; i < circles.size(); i++) {
		vector<Point2f> on_table = PointsOnTable(vector<Point2f>(1, Point2f(circles[i][0], circles[i][1])), homography);
		if (on_table.empty()) continue;
		detections.push_back(on_table[0]);
		circles_on_table.push_back(circles[i]);
	}
	return detections;
}

// Save the Hough stages
void SaveHoughStages(const Mat& image, const vector<Vec3f>& circles_on_table, const string& name){
	Mat edges;
	Canny(SmoothedGrey(image), edges, HOUGH_CANNY_THRESHOLD / 2, HOUGH_CANNY_THRESHOLD);
	Save("3_hough", name + " edges", edges);
	Mat display = image.clone();
	for (size_t i = 0; i < circles_on_table.size(); i++)
		circle(display, Point2f(circles_on_table[i][0], circles_on_table[i][1]), (int)circles_on_table[i][2], Scalar(0, 0, 255), 2);
	Save("3_hough", name + " circles", display);
	printf("%-8s %3d circles on the table\n", name.c_str(), (int)circles_on_table.size());
}

// Method 4: k-means, largest cluster = cloth
Mat KMeansMask(const Mat& table_image, const string& name){
	Mat samples = table_image.reshape(1, table_image.rows * table_image.cols);
	samples.convertTo(samples, CV_32F);
	Mat labels, centres;
	kmeans(samples, KMEANS_CLUSTERS, labels, TermCriteria(TermCriteria::MAX_ITER | TermCriteria::EPS, 10, 1.0),
		3, KMEANS_PP_CENTERS, centres);
	labels = labels.reshape(1, table_image.rows);

	Mat clustered(table_image.size(), CV_8UC3);
	for (int row = 0; row < table_image.rows; row++)
		for (int col = 0; col < table_image.cols; col++) {
			int cluster = labels.at<int>(row, col);
			clustered.at<Vec3b>(row, col) = Vec3b((uchar)centres.at<float>(cluster, 0), (uchar)centres.at<float>(cluster, 1), (uchar)centres.at<float>(cluster, 2));
		}
	Save("4_kmeans", name + " clusters", clustered);

	int cloth = 0, cloth_size = 0;
	printf("%-8s", name.c_str());
	for (int cluster = 0; cluster < KMEANS_CLUSTERS; cluster++) {
		int size = countNonZero(labels == cluster);
		printf("  BGR (%3.0f,%3.0f,%3.0f) %7d pixels", centres.at<float>(cluster, 0), centres.at<float>(cluster, 1), centres.at<float>(cluster, 2), size);
		if (size > cloth_size) {
			cloth = cluster;
			cloth_size = size;
		}
	}
	printf("\n");
	Mat binary = (labels != cloth);
	Save("4_kmeans", name + " binary", binary);
	return CloseAndOpen(binary, 5, "4_kmeans", name);
}

// Method 5: watershed on the chamfer distance
vector<Point2f> LocateByWatershed(const Mat& image, const Mat& homography, const string& name){
	Mat grey;
	cvtColor(image, grey, COLOR_BGR2GRAY);
	blur(grey, grey, Size(WATERSHED_SMOOTHING_SIZE, WATERSHED_SMOOTHING_SIZE));
	Mat edges;
	Canny(grey, edges, WATERSHED_CANNY_LOW, WATERSHED_CANNY_HIGH);
	Save("5_watershed", name + " edges", edges);

	Mat not_edges, chamfer;
	threshold(edges, not_edges, 127, 255, THRESH_BINARY_INV);
	distanceTransform(not_edges, chamfer, DIST_L2, 3);
	Mat chamfer_display;
	normalize(chamfer, chamfer_display, 0, 255, NORM_MINMAX, CV_8U);
	Save("5_watershed", name + " chamfer distance", chamfer_display);

	Mat dilated, maxima;
	dilate(chamfer, dilated, Mat());
	compare(chamfer, dilated, maxima, CMP_EQ);
	maxima &= (chamfer > WATERSHED_MARKER_DISTANCE);
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;
	findContours(maxima, contours, hierarchy, RETR_CCOMP, CHAIN_APPROX_NONE);
	Mat markers(image.size(), CV_32S, Scalar(0));
	int regions = 0;
	for (size_t i = 0; i < contours.size(); i++)
		if (hierarchy[i][3] < 0)
			drawContours(markers, contours, (int)i, Scalar(++regions), FILLED);

	Mat inverse_chamfer = 255 - chamfer_display;
	cvtColor(inverse_chamfer, inverse_chamfer, COLOR_GRAY2BGR);
	watershed(inverse_chamfer, markers);
	Mat display = image.clone();
	display.setTo(Scalar(0, 0, 255), markers == -1);
	Save("5_watershed", name + " regions", display);

	vector<double> area(regions + 1, 0.0), sum_x(regions + 1, 0.0), sum_y(regions + 1, 0.0);
	for (int row = 0; row < markers.rows; row++)
		for (int col = 0; col < markers.cols; col++) {
			int region = markers.at<int>(row, col);
			if (region <= 0) continue;
			area[region]++;
			sum_x[region] += col;
			sum_y[region] += row;
		}
	vector<Point2f> centres;
	for (int region = 1; region <= regions; region++)
		if (area[region] >= MINIMUM_BALL_AREA && area[region] <= WATERSHED_MAX_AREA)
			centres.push_back(Point2f((float)(sum_x[region] / area[region]), (float)(sum_y[region] / area[region])));
	vector<Point2f> detections = PointsOnTable(centres, homography);
	printf("%-8s %5d markers, %3d ball sized, %3d on the table\n", name.c_str(), regions, (int)centres.size(), (int)detections.size());
	return detections;
}

// Location counts
struct LocationResult {
	int balls = 0, true_positives = 0, false_positives = 0, false_negatives = 0;
	double total_error = 0, max_error = 0;
};

// Nearest unmatched detection within one radius = TP
LocationResult EvaluateLocations(int image_index, const vector<Point2f>& detections, vector<int>& ball_for_detection, vector<bool>& ball_found){
	LocationResult result;
	ball_for_detection.assign(detections.size(), -1);
	ball_found.assign(NUMBER_OF_BALLS, false);
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		if (!BallOnTable(image_index, ball)) continue;
		result.balls++;
		double closest = BALL_RADIUS;
		int closest_detection = -1;
		for (size_t detection = 0; detection < detections.size(); detection++) {
			if (ball_for_detection[detection] >= 0) continue;
			double distance = norm(detections[detection] - GroundTruthPosition(image_index, ball));
			if (distance <= closest) {
				closest = distance;
				closest_detection = (int)detection;
			}
		}
		if (closest_detection < 0) continue;
		ball_for_detection[closest_detection] = ball;
		ball_found[ball] = true;
		result.true_positives++;
		result.total_error += closest;
		result.max_error = max(result.max_error, closest);
	}
	result.false_positives = (int)detections.size() - result.true_positives;
	result.false_negatives = result.balls - result.true_positives;
	return result;
}

// Precision, recall, F1, errors
void PrintLocationRow(const string& name, const LocationResult& r){
	double precision = (r.true_positives + r.false_positives) ? (double)r.true_positives / (r.true_positives + r.false_positives) : 0.0;
	double recall = r.balls ? (double)r.true_positives / r.balls : 0.0;
	double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
	double mean_error = r.true_positives ? r.total_error / r.true_positives : 0.0;
	printf("%-10s %6d %4d %4d %4d %10.3f %8.3f %6.3f %9.1f %9.1f\n", name.c_str(), r.balls, r.true_positives,
		r.false_positives, r.false_negatives, precision, recall, f1, mean_error, r.max_error);
}

// Green found, red false, blue missed
Mat DrawLocations(const Mat& table_image, int image_index, const vector<Point2f>& detections,
	const vector<int>& ball_for_detection, const vector<bool>& ball_found){
	Mat display = table_image.clone();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		if (!BallOnTable(image_index, ball) || ball_found[ball]) continue;
		Point2f p = GroundTruthPosition(image_index, ball);
		drawMarker(display, p, Scalar(255, 0, 0), MARKER_CROSS, 24, 3);
		putText(display, BALL_NAMES[ball], p + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
	}
	for (size_t detection = 0; detection < detections.size(); detection++) {
		Scalar colour = (ball_for_detection[detection] >= 0) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
		circle(display, detections[detection], (int)BALL_RADIUS, colour, 2);
	}
	return display;
}

void PrintLocationHeader(const string& first_column){
	printf("%-10s %6s %4s %4s %4s %10s %8s %6s %9s %9s\n", first_column.c_str(), "Balls", "TP", "FP", "FN", "Precision", "Recall", "F1", "MeanErr", "MaxErr");
}

// Evaluate one location method
LocationResult EvaluateMethod(const string& title, const string& folder, const Mat table_images[], const vector<Point2f> detections[]){
	cout << endl << title << " results:" << endl;
	PrintLocationHeader("Image");
	LocationResult total;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++) {
		vector<int> ball_for_detection;
		vector<bool> ball_found;
		LocationResult result = EvaluateLocations(i, detections[i], ball_for_detection, ball_found);
		PrintLocationRow(IMAGE_NAMES[i], result);
		Save(folder, string(IMAGE_NAMES[i]) + " locations", DrawLocations(table_images[i], i, detections[i], ball_for_detection, ball_found));
		total.balls += result.balls;
		total.true_positives += result.true_positives;
		total.false_positives += result.false_positives;
		total.false_negatives += result.false_negatives;
		total.total_error += result.total_error;
		total.max_error = max(total.max_error, result.max_error);
	}
	PrintLocationRow("All", total);
	return total;
}

// Disc mask
Mat BallMask(const Mat& image, const Vec3f& ball_circle){
	Mat mask(image.size(), CV_8U, Scalar(0));
	circle(mask, Point2f(ball_circle[0], ball_circle[1]), (int)ball_circle[2], Scalar(255), FILLED);
	return mask;
}

// HSV histogram in a disc
Mat BallHistogram(const Mat& hsv_image, const Vec3f& ball_circle){
	Mat histogram;
	calcHist(&hsv_image, 1, BALL_HISTOGRAM_CHANNELS, BallMask(hsv_image, ball_circle), histogram, 3, BALL_HISTOGRAM_SIZES, BALL_HISTOGRAM_RANGES);
	normalize(histogram, histogram, 1, 0, NORM_L1);
	return histogram;
}

// 16 reference histograms from AllBalls
vector<Mat> ReferenceHistograms(const Mat& hsv_image, const Mat& homography, const vector<Vec3f>& training_circles, vector<Vec3f>& reference_circles){
	vector<float> radii;
	for (size_t i = 0; i < training_circles.size(); i++)
		radii.push_back(training_circles[i][2]);
	sort(radii.begin(), radii.end());
	float radius = radii[radii.size() / 2];

	vector<Mat> references;
	reference_circles.clear();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		vector<Point2f> table_point(1, GroundTruthPosition(0, ball)), image_point;
		perspectiveTransform(table_point, image_point, homography.inv());
		reference_circles.push_back(Vec3f(image_point[0].x, image_point[0].y, radius));
		references.push_back(BallHistogram(hsv_image, reference_circles[ball]));
	}
	return references;
}

// Smallest Bhattacharyya distance to the references
double SmallestReferenceDistance(const Mat& hsv_image, const vector<Mat>& references, const Vec3f& ball_circle){
	Mat histogram = BallHistogram(hsv_image, ball_circle);
	double smallest = 2.0;
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
		smallest = min(smallest, compareHist(histogram, references[ball], HISTCMP_BHATTACHARYYA));
	return smallest;
}

// Choose both thresholds on AllBalls
double TrainOnAllBalls(const Mat& image, const Mat& hsv_image, const Mat& homography, double& max_ball_distance){
	cout << "Training on AllBalls:" << endl;
	printf("%-10s %4s %4s %4s %9s %9s %14s %14s\n", "Threshold", "TP", "FP", "FN", "MeanErr", "MaxErr", "BallDistance", "OtherDistance");
	vector<int> best_thresholds;
	vector<double> best_distances;
	int most_balls = 0;
	for (int threshold = HOUGH_FIRST_ACCUMULATOR_THRESHOLD; threshold <= HOUGH_LAST_ACCUMULATOR_THRESHOLD; threshold++) {
		vector<Vec3f> circles, reference_circles;
		vector<Point2f> detections = LocateByHough(image, homography, threshold, circles);
		vector<int> ball_for_detection;
		vector<bool> ball_found;
		LocationResult result = EvaluateLocations(0, detections, ball_for_detection, ball_found);
		vector<Mat> references = ReferenceHistograms(hsv_image, homography, circles, reference_circles);
		double largest_ball_distance = 0, smallest_other_distance = 2.0;
		for (size_t i = 0; i < circles.size(); i++) {
			double distance = SmallestReferenceDistance(hsv_image, references, circles[i]);
			if (ball_for_detection[i] >= 0) largest_ball_distance = max(largest_ball_distance, distance);
			else smallest_other_distance = min(smallest_other_distance, distance);
		}
		printf("%-10d %4d %4d %4d %9.1f %9.1f %14.3f ", threshold, result.true_positives, result.false_positives, result.false_negatives,
			result.total_error / result.true_positives, result.max_error, largest_ball_distance);
		if (result.false_positives) printf("%14.3f\n", smallest_other_distance);
		else printf("%14s\n", "none");
		if (result.true_positives < most_balls || largest_ball_distance >= smallest_other_distance) continue;
		if (result.true_positives > most_balls) {
			most_balls = result.true_positives;
			best_thresholds.clear();
			best_distances.clear();
		}
		best_thresholds.push_back(threshold);
		best_distances.push_back(smallest_other_distance);
	}
	size_t middle = best_thresholds.size() / 2;
	max_ball_distance = best_distances[middle];
	printf("%d balls kept and every other circle rejected for thresholds %d to %d\n", most_balls, best_thresholds.front(), best_thresholds.back());
	printf("Accumulator threshold %d, distance threshold %.3f\n", best_thresholds[middle], max_ball_distance);
	return best_thresholds[middle];
}

// Reject pockets
void RejectPockets(const Mat& image, const Mat& hsv_image, const vector<Mat>& references, double max_ball_distance, vector<Vec3f>& ball_circles, vector<Point2f>& detections, const string& name){
	vector<Vec3f> kept_circles;
	vector<Point2f> kept_detections;
	Mat display = image.clone();
	for (size_t i = 0; i < ball_circles.size(); i++) {
		double smallest = SmallestReferenceDistance(hsv_image, references, ball_circles[i]);
		Scalar colour = (smallest >= max_ball_distance) ? Scalar(0, 0, 255) : Scalar(0, 255, 0);
		circle(display, Point2f(ball_circles[i][0], ball_circles[i][1]), (int)ball_circles[i][2], colour, 2);
		if (smallest >= max_ball_distance) {
			printf("%-8s rejected (%4.0f,%4.0f) distance %.3f\n", name.c_str(), detections[i].y, detections[i].x, smallest);
			continue;
		}
		kept_circles.push_back(ball_circles[i]);
		kept_detections.push_back(detections[i]);
	}
	Save("6_recognition", name + " kept and rejected circles", display);
	ball_circles = kept_circles;
	detections = kept_detections;
}

// Histogram comparison costs
vector<vector<double>> HistogramCosts(const Mat& hsv_image, const vector<Mat>& references, const vector<Vec3f>& ball_circles, int metric){
	double sign = (metric == HISTCMP_CORREL || metric == HISTCMP_INTERSECT) ? -1.0 : 1.0;
	vector<vector<double>> costs;
	for (size_t i = 0; i < ball_circles.size(); i++) {
		Mat histogram = BallHistogram(hsv_image, ball_circles[i]);
		vector<double> ball_costs;
		for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
			ball_costs.push_back(sign * compareHist(histogram, references[ball], metric));
		costs.push_back(ball_costs);
	}
	return costs;
}

// Back projection costs
vector<vector<double>> BackProjectionCosts(const Mat& hsv_image, const vector<Mat>& references, const vector<Vec3f>& ball_circles, const string& name){
	vector<Mat> probabilities(NUMBER_OF_BALLS);
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		Mat histogram;
		normalize(references[ball], histogram, 0, 255, NORM_MINMAX);
		calcBackProject(&hsv_image, 1, BALL_HISTOGRAM_CHANNELS, histogram, probabilities[ball], BALL_HISTOGRAM_RANGES);
	}
	Save("6_recognition", name + " back projection of ball 7", probabilities[7]);
	Save("6_recognition", name + " back projection of ball 15", probabilities[15]);
	vector<vector<double>> costs;
	for (size_t i = 0; i < ball_circles.size(); i++) {
		Mat mask = BallMask(hsv_image, ball_circles[i]);
		vector<double> ball_costs;
		for (int ball = 0; ball < NUMBER_OF_BALLS; ball++)
			ball_costs.push_back(-mean(probabilities[ball], mask)[0]);
		costs.push_back(ball_costs);
	}
	return costs;
}

// Assign numbers, optionally each number once
vector<int> AssignNumbers(const vector<vector<double>>& costs, bool unique_numbers){
	vector<int> labels(costs.size(), -1);
	vector<bool> used(NUMBER_OF_BALLS, false);
	for (size_t round = 0; round < costs.size(); round++) {
		int best_detection = -1, best_ball = -1;
		for (size_t detection = 0; detection < costs.size(); detection++) {
			if (labels[detection] >= 0) continue;
			for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
				if (unique_numbers && used[ball]) continue;
				if (best_detection < 0 || costs[detection][ball] < costs[best_detection][best_ball]) {
					best_detection = (int)detection;
					best_ball = ball;
				}
			}
		}
		if (best_detection < 0) break;
		labels[best_detection] = best_ball;
		used[best_ball] = true;
	}
	for (size_t detection = 0; detection < costs.size(); detection++)
		if (labels[detection] < 0)
			labels[detection] = (int)(min_element(costs[detection].begin(), costs[detection].end()) - costs[detection].begin());
	return labels;
}

// Green correct, red wrong, orange not a ball
Mat DrawRecognition(const Mat& table_image, const vector<Point2f>& detections, const vector<int>& ball_for_detection, const vector<int>& labels){
	Mat display = table_image.clone();
	for (size_t detection = 0; detection < detections.size(); detection++) {
		int actual = ball_for_detection[detection];
		Scalar colour = (actual < 0) ? Scalar(0, 165, 255) : (actual == labels[detection]) ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
		circle(display, detections[detection], (int)BALL_RADIUS, colour, 2);
		putText(display, BALL_NAMES[labels[detection]], detections[detection] + Point2f(24, -8), FONT_HERSHEY_SIMPLEX, 0.8, colour, 2);
	}
	return display;
}

// White fraction in a disc
double WhiteFraction(const Mat& hsv_image, const Vec3f& ball_circle){
	Mat mask = BallMask(hsv_image, ball_circle), white;
	inRange(hsv_image, Scalar(0, 0, WHITE_MIN_VALUE), Scalar(180, WHITE_MAX_SATURATION, 255), white);
	return (double)countNonZero(white & mask) / countNonZero(mask);
}

// White fractions of the errors
void PrintWhiteFractions(const Mat hsv_images[], const vector<Vec3f> ball_circles[], const vector<Point2f> detections[], const vector<int> labels[], const vector<Vec3f>& reference_circles){
	cout << endl << "White fraction of the balls recognised wrongly:" << endl;
	printf("%-8s %8s %8s %16s %20s\n", "Image", "Error", "Ball", "Actual reference", "Recognised reference");
	for (int i = 1; i < NUMBER_OF_IMAGES; i++) {
		vector<int> ball_for_detection;
		vector<bool> ball_found;
		EvaluateLocations(i, detections[i], ball_for_detection, ball_found);
		for (size_t detection = 0; detection < detections[i].size(); detection++) {
			int actual = ball_for_detection[detection], recognised = labels[i][detection];
			if (actual < 0 || actual == recognised) continue;
			printf("%-8s %5s->%-2s %8.2f %16.2f %20.2f\n", IMAGE_NAMES[i], BALL_NAMES[actual], BALL_NAMES[recognised], WhiteFraction(hsv_images[i], ball_circles[i][detection]),
				WhiteFraction(hsv_images[0], reference_circles[actual]), WhiteFraction(hsv_images[0], reference_circles[recognised]));
		}
	}
}

// Recognition counts
struct RecognitionResult {
	int balls = 0, located = 0, correct = 0;
};

void PrintRecognitionHeader(const string& first_column, const string& last_column){
	printf("%-34s %6s %8s %8s %9s %9s  %s\n", first_column.c_str(), "Balls", "Located", "Correct", "Accuracy", "Overall", last_column.c_str());
}

void PrintRecognitionRow(const string& name, const RecognitionResult& r, const string& errors){
	printf("%-34s %6d %8d %8d %9.3f %9.3f %s\n", name.c_str(), r.balls, r.located, r.correct,
		(double)r.correct / r.located, (double)r.correct / r.balls, errors.c_str());
}

// Evaluate recognition
RecognitionResult EvaluateRecognition(const string& title, const Mat table_images[], const vector<Point2f> detections[], const vector<int> labels[]){
	cout << endl << title << " recognition:" << endl;
	PrintRecognitionHeader("Image", "Errors (actual->recognised)");
	RecognitionResult total;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++) {
		vector<int> ball_for_detection;
		vector<bool> ball_found;
		LocationResult location = EvaluateLocations(i, detections[i], ball_for_detection, ball_found);
		int correct = 0;
		string errors;
		for (size_t detection = 0; detection < detections[i].size(); detection++) {
			int actual = ball_for_detection[detection];
			if (actual < 0) continue;
			if (labels[i][detection] == actual) correct++;
			else errors += string(" ") + BALL_NAMES[actual] + "->" + BALL_NAMES[labels[i][detection]];
		}
		RecognitionResult result;
		result.balls = location.balls;
		result.located = location.true_positives;
		result.correct = correct;
		PrintRecognitionRow(IMAGE_NAMES[i], result, errors);
		Save("6_recognition", title + " " + IMAGE_NAMES[i], DrawRecognition(table_images[i], detections[i], ball_for_detection, labels[i]));
		total.balls += result.balls;
		total.located += result.located;
		total.correct += result.correct;
	}
	PrintRecognitionRow("All", total, "");
	return total;
}

// Without and with unique numbers
void EvaluateWithAndWithoutUniqueNumbers(const string& title, const vector<vector<double>> costs[], const Mat table_images[], const vector<Point2f> detections[], RecognitionResult results[2]){
	vector<int> labels[NUMBER_OF_IMAGES];
	for (int unique_numbers = 0; unique_numbers < 2; unique_numbers++) {
		for (int i = 1; i < NUMBER_OF_IMAGES; i++)
			labels[i] = AssignNumbers(costs[i], unique_numbers == 1);
		results[unique_numbers] = EvaluateRecognition(unique_numbers ? title : title + " without unique numbers", table_images, detections, labels);
	}
}

// Recognition totals
void PrintComparison(const string& title, const char* const names[], const RecognitionResult results[][2], int count){
	cout << endl << title << endl;
	PrintRecognitionHeader("", "");
	for (int i = 0; i < count; i++) {
		PrintRecognitionRow(string(names[i]) + " without unique", results[i][0], "");
		PrintRecognitionRow(string(names[i]) + " with unique", results[i][1], "");
	}
}

}

void MyApplication(){
	// Same k-means result on every run
	theRNG() = RNG();

	// Load images
	Mat images[NUMBER_OF_IMAGES];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
		string filename = string(FILE_LOCATION) + IMAGE_NAMES[i] + ".jpeg";
		images[i] = imread(filename, IMREAD_COLOR);
		if (images[i].empty()) {
			cout << "Could not open " << filename << endl;
			return;
		}
		cout << IMAGE_NAMES[i] << ": " << images[i].cols << " x " << images[i].rows << " pixels" << endl;
	}

	// Rectify
	Mat corners_image = images[0].clone();
	for (int corner = 0; corner < 4; corner++)
		line(corners_image, TABLE_CORNERS[corner], TABLE_CORNERS[(corner + 1) % 4], Scalar(0, 0, 255), 2);
	Save("0_data", "AllBalls corners", corners_image);

	Point2f table_corners[4] = { Point2f(0, 0), Point2f(TABLE_COLS - 1, 0), Point2f(TABLE_COLS - 1, TABLE_ROWS - 1), Point2f(0, TABLE_ROWS - 1) };
	Mat homography = getPerspectiveTransform(TABLE_CORNERS, table_corners);
	cout << endl << "Perspective transformation:" << endl << homography << endl;

	Mat table_images[NUMBER_OF_IMAGES];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
		warpPerspective(images[i], table_images[i], homography, Size(TABLE_COLS, TABLE_ROWS));
		Save("0_data", string(IMAGE_NAMES[i]) + " ground truth", DrawGroundTruth(table_images[i], i));
	}

	// Stage 1: location
	vector<Point2f> detections[NUMBER_OF_IMAGES];
	LocationResult totals[5];

	cout << endl << "Method 1: back projection" << endl;
	MatND histogram = TableHistogram(table_images[0]);
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(BackProjectionMask(table_images[i], histogram, IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	totals[0] = EvaluateMethod("Back projection", "1_backprojection", table_images, detections);

	cout << endl << "Method 2: background subtraction" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(BackgroundMask(table_images[i], table_images[0], IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	totals[1] = EvaluateMethod("Background subtraction", "2_background", table_images, detections);

	cout << endl << "Method 3: Hough circles" << endl;
	Mat hsv_images[NUMBER_OF_IMAGES];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++)
		cvtColor(images[i], hsv_images[i], COLOR_BGR2HSV);
	double max_ball_distance;
	double accumulator_threshold = TrainOnAllBalls(images[0], hsv_images[0], homography, max_ball_distance);
	vector<Vec3f> hough_circles[NUMBER_OF_IMAGES];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++) {
		detections[i] = LocateByHough(images[i], homography, accumulator_threshold, hough_circles[i]);
		SaveHoughStages(images[i], hough_circles[i], IMAGE_NAMES[i]);
	}
	vector<Point2f> hough_detections[NUMBER_OF_IMAGES];
	for (int i = 0; i < NUMBER_OF_IMAGES; i++)
		hough_detections[i] = detections[i];
	totals[2] = EvaluateMethod("Hough circles", "3_hough", table_images, detections);

	cout << endl << "Method 4: k-means clustering of colours" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(KMeansMask(table_images[i], IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	totals[3] = EvaluateMethod("k-means", "4_kmeans", table_images, detections);

	cout << endl << "Method 5: watershed" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = LocateByWatershed(images[i], homography, IMAGE_NAMES[i]);
	totals[4] = EvaluateMethod("Watershed", "5_watershed", table_images, detections);

	// Compare location methods
	const char* method_names[5] = { "Back proj", "Background", "Hough", "k-means", "Watershed" };
	cout << endl << "Comparison of the location methods (Table1 to Table6):" << endl;
	PrintLocationHeader("Method");
	for (int method = 0; method < 5; method++)
		PrintLocationRow(method_names[method], totals[method]);

	cout << endl << "Recognition: rejecting the circles unlike every reference ball" << endl;
	// Stage 2: recognition
	vector<Vec3f> reference_circles;
	vector<Mat> references = ReferenceHistograms(hsv_images[0], homography, hough_circles[0], reference_circles);
	cout << "Reference ball radius: " << reference_circles[0][2] << " pixels" << endl;
	Mat reference_discs = images[0].clone();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		Point2f centre(reference_circles[ball][0], reference_circles[ball][1]);
		circle(reference_discs, centre, (int)reference_circles[ball][2], Scalar(0, 255, 255), 2);
		putText(reference_discs, BALL_NAMES[ball], centre + Point2f(-10, -reference_circles[ball][2] - 4), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 255, 255), 2);
	}
	Save("6_recognition", "AllBalls reference discs", reference_discs);
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		RejectPockets(images[i], hsv_images[i], references, max_ball_distance, hough_circles[i], hough_detections[i], IMAGE_NAMES[i]);
	EvaluateMethod("Hough circles after rejection", "6_recognition", table_images, hough_detections);

	// Histogram metrics
	const char* metric_names[4] = { "Bhattacharyya", "Correlation", "Intersection", "Chi-square" };
	const int metrics[4] = { HISTCMP_BHATTACHARYYA, HISTCMP_CORREL, HISTCMP_INTERSECT, HISTCMP_CHISQR };
	vector<vector<double>> costs[NUMBER_OF_IMAGES];
	RecognitionResult metric_results[4][2], method_results[2][2];
	for (int metric = 0; metric < 4; metric++) {
		for (int i = 1; i < NUMBER_OF_IMAGES; i++)
			costs[i] = HistogramCosts(hsv_images[i], references, hough_circles[i], metrics[metric]);
		EvaluateWithAndWithoutUniqueNumbers(metric_names[metric], costs, table_images, hough_detections, metric_results[metric]);
		if (metric > 0) continue;
		vector<int> labels[NUMBER_OF_IMAGES];
		for (int i = 1; i < NUMBER_OF_IMAGES; i++)
			labels[i] = AssignNumbers(costs[i], false);
		PrintWhiteFractions(hsv_images, hough_circles, hough_detections, labels, reference_circles);
	}
	method_results[0][0] = metric_results[0][0];
	method_results[0][1] = metric_results[0][1];

	// Back projection
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		costs[i] = BackProjectionCosts(hsv_images[i], references, hough_circles[i], IMAGE_NAMES[i]);
	EvaluateWithAndWithoutUniqueNumbers("Back projection", costs, table_images, hough_detections, method_results[1]);

	// Compare recognition methods
	PrintComparison("Comparison of the histogram metrics (Table1 to Table6):", metric_names, metric_results, 4);
	const char* recognition_names[2] = { "Histogram", "Back projection" };
	PrintComparison("Comparison of the recognition methods, with the Bhattacharyya histogram metric (Table1 to Table6):", recognition_names, method_results, 2);
}
