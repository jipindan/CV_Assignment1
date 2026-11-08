#include "Utilities.h"
#include "opencv2/core/utils/filesystem.hpp"

namespace {

const int NUMBER_OF_IMAGES = 7;
const char* FILE_LOCATION = "Media/Pool/";
const char* RESULTS_LOCATION = "Results/";
const char* IMAGE_NAMES[NUMBER_OF_IMAGES] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

const int NUMBER_OF_BALLS = 16;
const char* BALL_NAMES[NUMBER_OF_BALLS] = {
	"W", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15" };

const int TABLE_COLS = 1434;
const int TABLE_ROWS = 683;
const double BALL_RADIUS = 21.5;

const Point2f TABLE_CORNERS[4] = { Point2f(337, 35), Point2f(1570, 21), Point2f(1888, 451), Point2f(38, 472) };

const int GROUND_TRUTH[NUMBER_OF_IMAGES][NUMBER_OF_BALLS][2] = {
	{ {293,797}, {253,908}, {250,959}, {255,1013}, {256,1072}, {254,1132}, {262,1184}, {256,1244}, {253,1307}, {329,901}, {330,961}, {335,1013}, {336,1067}, {334,1130}, {345,1187}, {342,1245} },
	{ {288,29}, {309,1092}, {429,1212}, {383,1131}, {335,1050}, {285,1211}, {261,1170}, {352,1173}, {331,1132}, {240,1210}, {310,1170}, {407,1169}, {281,1135}, {374,1212}, {360,1091}, {331,1213} },
	{ {507,1169}, {282,1103}, {620,1224}, {391,1121}, {315,1046}, {310,1305}, {202,621}, {361,1173}, {331,1132}, {177,725}, {185,1269}, {665,798}, {296,438}, {53,1235}, {362,1090}, {399,1205} },
	{ {434,29}, {23,908}, {616,1224}, {447,1199}, {336,29}, {305,1406}, {210,621}, {260,1294}, {226,1252}, {177,725}, {171,1270}, {665,798}, {249,351}, {53,1233}, {169,901}, {399,1201} },
	{ {88,1392}, {23,908}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {210,621}, {260,1294}, {226,1252}, {181,724}, {171,1270}, {665,798}, {249,351}, {58,1231}, {169,901}, {399,1201} },
	{ {608,26}, {535,23}, {546,1407}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294}, {226,1252}, {538,72}, {171,1270}, {665,798}, {249,351}, {-1,-1}, {169,901}, {399,1201} },
	{ {629,947}, {535,23}, {477,1364}, {444,1204}, {-1,-1}, {305,1406}, {131,25}, {260,1294}, {226,1252}, {538,72}, {171,1270}, {-1,-1}, {249,351}, {-1,-1}, {169,901}, {399,1201} } };

const int MINIMUM_BALL_AREA = 400;

const int HUE_BINS = 18;
const int SATURATION_BINS = 16;

const int DIFFERENCE_THRESHOLD = 30;

const int HOUGH_SMOOTHING_SIZE = 5;
const double HOUGH_MIN_CENTRE_DISTANCE = 30;
const double HOUGH_CANNY_THRESHOLD = 100;
const double HOUGH_ACCUMULATOR_THRESHOLD = 20;
const int HOUGH_MIN_RADIUS = 17;
const int HOUGH_MAX_RADIUS = 31;

const int KMEANS_CLUSTERS = 4;

void Save(const string& folder, const string& name, const Mat& image){
	string directory = string(RESULTS_LOCATION) + folder;
	utils::fs::createDirectories(directory);
	string extension = (image.channels() == 3) ? ".jpg" : ".png";
	imwrite(directory + "/" + name + extension, image);
}

bool BallOnTable(int image_index, int ball){
	return GROUND_TRUTH[image_index][ball][0] >= 0;
}

Point2f GroundTruthPosition(int image_index, int ball){
	return Point2f((float)GROUND_TRUTH[image_index][ball][1], (float)GROUND_TRUTH[image_index][ball][0]);
}

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

Mat CloseAndOpen(const Mat& binary, int element_size, const string& folder, const string& name){
	Mat element(element_size, element_size, CV_8U, Scalar(1));
	Mat cleaned;
	morphologyEx(binary, cleaned, MORPH_CLOSE, element);
	morphologyEx(cleaned, cleaned, MORPH_OPEN, element);
	Save(folder, name + " closed and opened", cleaned);
	return cleaned;
}

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

vector<Point2f> LocateByHough(const Mat& image, const Mat& homography, const string& name){
	Mat grey;
	cvtColor(image, grey, COLOR_BGR2GRAY);
	GaussianBlur(grey, grey, Size(HOUGH_SMOOTHING_SIZE, HOUGH_SMOOTHING_SIZE), 0);
	Mat edges;
	Canny(grey, edges, HOUGH_CANNY_THRESHOLD / 2, HOUGH_CANNY_THRESHOLD);
	Save("3_hough", name + " edges", edges);

	vector<Vec3f> circles;
	HoughCircles(grey, circles, HOUGH_GRADIENT, 1, HOUGH_MIN_CENTRE_DISTANCE, HOUGH_CANNY_THRESHOLD,
		HOUGH_ACCUMULATOR_THRESHOLD, HOUGH_MIN_RADIUS, HOUGH_MAX_RADIUS);
	Mat display = image.clone();
	vector<Point2f> centres, table_points, detections;
	for (size_t i = 0; i < circles.size(); i++) {
		circle(display, Point2f(circles[i][0], circles[i][1]), (int)circles[i][2], Scalar(0, 0, 255), 2);
		centres.push_back(Point2f(circles[i][0], circles[i][1]));
	}
	Save("3_hough", name + " circles", display);
	if (!centres.empty()) perspectiveTransform(centres, table_points, homography);
	for (size_t i = 0; i < table_points.size(); i++)
		if (table_points[i].x >= 0 && table_points[i].x < TABLE_COLS && table_points[i].y >= 0 && table_points[i].y < TABLE_ROWS)
			detections.push_back(table_points[i]);
	printf("%-8s %3d circles, %3d on the table\n", name.c_str(), (int)circles.size(), (int)detections.size());
	return detections;
}

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

struct LocationResult {
	int balls = 0, true_positives = 0, false_positives = 0, false_negatives = 0;
	double total_error = 0, max_error = 0;
};

LocationResult EvaluateLocations(int image_index, const vector<Point2f>& detections, vector<bool>& detection_correct, vector<bool>& ball_found){
	LocationResult result;
	detection_correct.assign(detections.size(), false);
	ball_found.assign(NUMBER_OF_BALLS, false);
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		if (!BallOnTable(image_index, ball)) continue;
		result.balls++;
		double closest = BALL_RADIUS;
		int closest_detection = -1;
		for (size_t detection = 0; detection < detections.size(); detection++) {
			double distance = norm(detections[detection] - GroundTruthPosition(image_index, ball));
			if (distance <= closest) {
				closest = distance;
				closest_detection = (int)detection;
			}
		}
		if (closest_detection < 0) continue;
		detection_correct[closest_detection] = true;
		ball_found[ball] = true;
		result.true_positives++;
		result.total_error += closest;
		result.max_error = max(result.max_error, closest);
	}
	result.false_positives = (int)detections.size() - result.true_positives;
	result.false_negatives = result.balls - result.true_positives;
	return result;
}

void PrintLocationRow(const string& name, const LocationResult& r){
	double precision = (r.true_positives + r.false_positives) ? (double)r.true_positives / (r.true_positives + r.false_positives) : 0.0;
	double recall = r.balls ? (double)r.true_positives / r.balls : 0.0;
	double f1 = (precision + recall) > 0 ? 2 * precision * recall / (precision + recall) : 0.0;
	double mean_error = r.true_positives ? r.total_error / r.true_positives : 0.0;
	printf("%-8s %6d %4d %4d %4d %10.3f %8.3f %6.3f %9.1f %9.1f\n", name.c_str(), r.balls, r.true_positives,
		r.false_positives, r.false_negatives, precision, recall, f1, mean_error, r.max_error);
}

Mat DrawLocations(const Mat& table_image, int image_index, const vector<Point2f>& detections,
	const vector<bool>& detection_correct, const vector<bool>& ball_found){
	Mat display = table_image.clone();
	for (int ball = 0; ball < NUMBER_OF_BALLS; ball++) {
		if (!BallOnTable(image_index, ball) || ball_found[ball]) continue;
		Point2f p = GroundTruthPosition(image_index, ball);
		drawMarker(display, p, Scalar(255, 0, 0), MARKER_CROSS, 24, 3);
		putText(display, BALL_NAMES[ball], p + Point2f(10, -10), FONT_HERSHEY_SIMPLEX, 0.6, Scalar(255, 0, 0), 2);
	}
	for (size_t detection = 0; detection < detections.size(); detection++) {
		Scalar colour = detection_correct[detection] ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
		circle(display, detections[detection], (int)BALL_RADIUS, colour, 2);
	}
	return display;
}

void EvaluateMethod(const string& title, const string& folder, const Mat table_images[], const vector<Point2f> detections[]){
	cout << endl << title << " results:" << endl;
	printf("%-8s %6s %4s %4s %4s %10s %8s %6s %9s %9s\n", "Image", "Balls", "TP", "FP", "FN", "Precision", "Recall", "F1", "MeanErr", "MaxErr");
	LocationResult total;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++) {
		vector<bool> detection_correct, ball_found;
		LocationResult result = EvaluateLocations(i, detections[i], detection_correct, ball_found);
		PrintLocationRow(IMAGE_NAMES[i], result);
		Save(folder, string(IMAGE_NAMES[i]) + " locations", DrawLocations(table_images[i], i, detections[i], detection_correct, ball_found));
		total.balls += result.balls;
		total.true_positives += result.true_positives;
		total.false_positives += result.false_positives;
		total.false_negatives += result.false_negatives;
		total.total_error += result.total_error;
		total.max_error = max(total.max_error, result.max_error);
	}
	PrintLocationRow("All", total);
}

}

void MyApplication(){
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

	vector<Point2f> detections[NUMBER_OF_IMAGES];

	cout << endl << "Method 1: back projection" << endl;
	MatND histogram = TableHistogram(table_images[0]);
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(BackProjectionMask(table_images[i], histogram, IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	EvaluateMethod("Back projection", "1_backprojection", table_images, detections);

	cout << endl << "Method 2: background subtraction" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(BackgroundMask(table_images[i], table_images[0], IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	EvaluateMethod("Background subtraction", "2_background", table_images, detections);

	cout << endl << "Method 3: Hough circles" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = LocateByHough(images[i], homography, IMAGE_NAMES[i]);
	EvaluateMethod("Hough circles", "3_hough", table_images, detections);

	cout << endl << "Method 4: k-means clustering of colours" << endl;
	for (int i = 1; i < NUMBER_OF_IMAGES; i++)
		detections[i] = RegionCentres(KMeansMask(table_images[i], IMAGE_NAMES[i]), IMAGE_NAMES[i]);
	EvaluateMethod("k-means", "4_kmeans", table_images, detections);
}
