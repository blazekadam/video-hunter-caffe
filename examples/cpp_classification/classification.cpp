#include <caffe/caffe.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <algorithm>
#include <iosfwd>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace caffe;  // NOLINT(build/namespaces)
using std::string;

/* Pair (label, confidence) representing a prediction. */
typedef std::pair<string, float> Prediction;

static bool PairCompare(const std::pair<float, int>& lhs,
	const std::pair<float, int>& rhs) {
	return lhs.first > rhs.first;
}

/* Return the indices of the top N values of vector v. */
static std::vector<int> Argmax(const std::vector<float>& v, int N) {
	std::vector<std::pair<float, int> > pairs;
	for (size_t i = 0; i < v.size(); ++i)
		pairs.push_back(std::make_pair(v[i], static_cast<int>(i)));
	std::partial_sort(pairs.begin(), pairs.begin() + N, pairs.end(), PairCompare);
	std::vector<int> result;
	for (int i = 0; i < N; ++i)
		result.push_back(pairs[i].second);
	return result;
}


class Classifier {
private:
	shared_ptr<Net<float> > net_;
	cv::Size input_geometry_;
	int num_channels_;
	cv::Mat mean_;
	std::vector<string> labels_;

private:

	void WrapInputLayer(std::vector<cv::Mat>* input_channels)  {
		Blob<float>* input_layer = net_->input_blobs()[0];
		int width = input_layer->width();
		int height = input_layer->height();
		float* input_data = input_layer->mutable_cpu_data();
		for (int i = 0; i < input_layer->channels(); ++i) {
			cv::Mat channel(height, width, CV_32FC1, input_data);
			input_channels->push_back(channel);
			input_data += width * height;
		}
	}

	void Preprocess(const cv::Mat& img, std::vector<cv::Mat>* input_channels)  {
		/* Convert the input image to the input image format of the network. */
		cv::Mat sample;
		if (img.channels() == 3 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGR2GRAY);
		else if (img.channels() == 4 && num_channels_ == 1)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2GRAY);
		else if (img.channels() == 4 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_BGRA2BGR);
		else if (img.channels() == 1 && num_channels_ == 3)
			cv::cvtColor(img, sample, cv::COLOR_GRAY2BGR);
		else
			sample = img;

		cv::Mat sample_resized;
		if (sample.size() != input_geometry_)
			cv::resize(sample, sample_resized, input_geometry_);
		else
			sample_resized = sample;

		cv::Mat sample_float;
		if (num_channels_ == 3)
			sample_resized.convertTo(sample_float, CV_32FC3);
		else
			sample_resized.convertTo(sample_float, CV_32FC1);

		cv::Mat sample_normalized;
		cv::subtract(sample_float, mean_, sample_normalized);

		/* This operation will write the separate BGR planes directly to the
		* input layer of the network because it is wrapped by the cv::Mat
		* objects in input_channels. */
		cv::split(sample_normalized, *input_channels);

		CHECK(reinterpret_cast<float*>(input_channels->at(0).data)
			== net_->input_blobs()[0]->cpu_data())
			<< "Input channels are not wrapping the input layer of the network.";
	}

	std::vector<float> Predict(const cv::Mat& img) {
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		Preprocess(img, &input_channels);

		net_->Forward();

		/* Copy the output layer to a std::vector */
		Blob<float>* output_layer = net_->output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		return std::vector<float>(begin, end);
	}

	std::vector<float> GetFeature(const cv::Mat& img) {
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		Preprocess(img, &input_channels);

		net_->Forward();

		/* Copy the output layer to a std::vector */
		Blob<float>* feature_layer = net_->blob_by_name("fc7").get();
		const float* begin = feature_layer->cpu_data();
		const float* end = begin + feature_layer->channels();
		return std::vector<float>(begin, end);
	}

	std::pair<std::vector<float>, std::vector<float>> GetBoth(const cv::Mat& img) {
		Blob<float>* input_layer = net_->input_blobs()[0];
		input_layer->Reshape(1, num_channels_,
			input_geometry_.height, input_geometry_.width);
		/* Forward dimension change to all layers. */
		net_->Reshape();

		std::vector<cv::Mat> input_channels;
		WrapInputLayer(&input_channels);

		Preprocess(img, &input_channels);

		net_->Forward();

		/* Copy the output layer to a std::vector */
		Blob<float>* feature_layer = net_->blob_by_name("fc7").get();
		const float* fbegin = feature_layer->cpu_data();
		const float* fend = fbegin + feature_layer->channels();

		Blob<float>* output_layer = net_->output_blobs()[0];
		const float* begin = output_layer->cpu_data();
		const float* end = begin + output_layer->channels();
		return std::pair<std::vector<float>, std::vector<float>>(std::vector<float>(fbegin, fend), std::vector<float>(begin, end));
	}

public:
	Classifier(const string& model_file, const string& trained_file, const string& label_file) {
		// Load the model
		Caffe::set_mode(Caffe::CPU);
		net_.reset(new Net<float>(model_file, TEST));
		net_->CopyTrainedLayersFrom(trained_file);

		Blob<float>* input_layer = net_->input_blobs()[0];
		Blob<float>* output_layer = net_->output_blobs()[0];
		num_channels_ = input_layer->channels();
		input_geometry_ = cv::Size(input_layer->width(), input_layer->height());

		// Prepare the normalization
		mean_ = cv::Mat(input_geometry_, CV_32FC3, cv::Scalar(103.939, 116.779, 123.68, 0));

		// Load the labels
		std::ifstream labels(label_file.c_str());
		string line;
		while (std::getline(labels, line))
			labels_.push_back(string(line));
	}

	std::vector<Prediction> Classify(const cv::Mat& img, int N = 5) {
		std::vector<float> output = Predict(img);

		N = std::min<int>(labels_.size(), N);
		std::vector<int> maxN = Argmax(output, N);
		std::vector<Prediction> predictions;
		for (int i = 0; i < N; ++i) {
			int idx = maxN[i];
			predictions.push_back(std::make_pair(labels_[idx], output[idx]));
		}

		return predictions;
	}

	std::string Feature(const cv::Mat& img) {
		std::vector<float> output = GetFeature(img);
		std::stringstream ss = std::stringstream();
		for (const auto & f : output)
			ss << f << ' ';
		ss << std::endl;
		return ss.str();
	}

	std::pair<std::string, std::vector<Prediction>> Both(const cv::Mat& img, int N = 5) {
		std::pair<std::vector<float>, std::vector<float>> output = GetBoth(img);
		auto & feature = output.first;
		auto & predict = output.second;
		std::stringstream ss = std::stringstream();
		for (const auto & f : feature)
			ss << f << ' ';
		ss << std::endl;

		N = std::min<int>(labels_.size(), N);
		std::vector<int> maxN = Argmax(predict, N);
		std::vector<Prediction> predictions;
		for (int i = 0; i < N; ++i) {
			int idx = maxN[i];
			predictions.push_back(std::make_pair(labels_[idx], predict[idx]));
		}

		return std::pair<std::string, std::vector<Prediction>>(ss.str(), predictions);
	}
};

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);

	string model_file   = "vgg.prototxt";
	string trained_file = "vgg.caffemodel";
	string label_file   = "synset_words.txt";
	Classifier classifier(model_file, trained_file, label_file);

	std::cout << "Type P, F or B to obtain either top 5 labels, fc7 features or both" << std::endl;
	std::cout << "READY" << std::endl;
	string type;
	std::getline(std::cin, type);
	
	bool both = type[0] == 'B';
	bool prediction = type[0] == 'P';

	for (string file; std::getline(std::cin, file);) {
		if (file == "EXIT")
			break;

		cv::Mat img = cv::imread(file, -1);
		CHECK(!img.empty()) << "Unable to decode image " << file;
		if (both) {
			std::pair<std::string, std::vector<Prediction>> result = classifier.Both(img);
			auto & predictions = result.second;
			auto & feature = result.first;
			for (size_t i = 0; i < predictions.size(); ++i) {
				Prediction p = predictions[i];
				std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
			}
			std::cout << feature;

		}
		else if (prediction) {
			std::cout << "---------- Prediction for " << file << " ----------" << std::endl;
			std::vector<Prediction> predictions = classifier.Classify(img);

			/* Print the top N predictions. */
			for (size_t i = 0; i < predictions.size(); ++i) {
				Prediction p = predictions[i];
				std::cout << std::fixed << std::setprecision(4) << p.second << " - \"" << p.first << "\"" << std::endl;
			}
		}
		else {
			std::string feature = classifier.Feature(img);
			std::cout << feature;
		}
	}
}