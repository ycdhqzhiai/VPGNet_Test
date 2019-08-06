#include <caffe/caffe.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string.h>
#include <algorithm> 
using namespace caffe;
using namespace cv;
#define MAX_INPUT_SIDE 640;
#define MIN_INPUT_SIDE 480;
//dump caffe feature map
class CaffeDump {
public:
    CaffeDump(const std::string& net_file, const std::string& weight_file, const int GPUID);
    ~CaffeDump();
    void caffe_forward(cv::Mat img);
private:
    void preprocess(cv::Mat cv_image);
    cv::Mat image_translation(cv::Mat & srcImage, int x0ffset, int y0ffset);
private:
    shared_ptr<Net<float> > net_;
    int num_channels_;
    float threshold_;
};
CaffeDump::CaffeDump(const std::string& net_file, const std::string& weights_file, const int GPUID)
{
    Caffe::SetDevice(GPUID);
    Caffe::set_mode(Caffe::GPU);
    net_.reset(new Net<float>(net_file, caffe::TEST));
    net_->CopyTrainedLayersFrom(weights_file);
    Blob<float>* input_layer = net_->input_blobs()[0];
    num_channels_ = input_layer->channels();
    CHECK_EQ(num_channels_, 3) << "Input layer should have 3 channels.";
}
CaffeDump::~CaffeDump() {}
void CaffeDump::caffe_forward(cv::Mat cv_image)
{
    if (cv_image.empty()) {
        std::cout << "Can not reach the image" << std::endl;
        return;
    }
    preprocess(cv_image);
    net_->ForwardFrom(0);
    shared_ptr<caffe::Blob<float>> layerData =net_->blob_by_name("binary-mask");
    shared_ptr<caffe::Blob<float>> labelData =net_->blob_by_name("multi-label");    
    int batch_size = layerData->num();  
    int dim_features = layerData->count() / batch_size; 
    int channels = layerData->channels();
    int height = layerData->height();
    int width = layerData->width();
    std::cout << "batch size:" << batch_size << std::endl;
    std::cout << "dimension:" << dim_features << std::endl;
    std::cout << "channels:" << channels << std::endl;
    std::cout << "height:" << height << std::endl;
    std::cout << "width:" << width << std::endl;
    std::cout << "channels*height*width:" << channels*height*width << std::endl;
    //CHECK_LT(channel, channels) << "Input channel number should small than channels.";
    float* feature_blob_data; 



    int l_features = labelData->count() / batch_size; 
    int l_channels = labelData->channels();
    int l_height = labelData->height();
    int l_width = labelData->width();
    std::cout << "l_dimension:" << l_features << std::endl;
    std::cout << "l_channels:" << l_channels << std::endl;
    std::cout << "l_height:" << l_height << std::endl;
    std::cout << "l_width:" << l_width << std::endl;
    std::cout << "l_channels*height*width:" << l_channels*l_height*l_width << std::endl;
    //CHECK_LT(channel, channels) << "Input channel number should small than channels.";
    float* label_blob_data; 
 
    for (int n = 0; n < batch_size; ++n)  
    { 

	printf("offset: %d\n", layerData->offset(n));
        feature_blob_data = layerData->mutable_cpu_data() +  layerData->offset(n); 
	label_blob_data = labelData->mutable_cpu_data() +  layerData->offset(n);
        float *arr = (float(*))malloc(height*width*sizeof(float));
        int idx = 0;

	float classion[l_channels][l_height][l_width] = {0};
	for(int c = 0; c < l_channels; c++)
	{
	    for(int h = 0; h < l_height; h++)
	    {
	        for(int w = 0; w < l_width; w++)
	        {
	    	    classion[c][h][w] = *label_blob_data++;
	        }
	    }
	}


	//std::vector<int> ll;
	int y_offset_class = 1; //offset for classification error
	int x_offset_class = 1;
	float grid_size = 8;
        for(int h = 0; h < l_height; h++)
        {
            for(int w = 0; w < l_width; w++)
            {  
		float max = 0;
		int label = 0;
	        for(int c = 0; c < l_channels; c++) 
		{
                    if (classion[c][h][w] > max)
		    {
			max = classion[c][h][w];
			label = c;
		    }
		}
		//ll.push_back(label);
		if(label != 0)
		{
                    cv::Point2f pt1 = cv::Point((float)(w + y_offset_class) * grid_size, (float)(h + x_offset_class) * grid_size);
                    cv::Point2f pt2 = cv::Point((float)(w + y_offset_class) * grid_size + grid_size, (float)(h + x_offset_class) * grid_size+grid_size);
            	    cv::rectangle(cv_image, pt1, pt2, Scalar(255,0,0),1,1,0);
		}
            }
        }
	cv::imwrite("result.jpg", cv_image);
	//for(int i = 0; i < ll.size(); i++)
	//{
	//    if(ll[i] != 0)
	//	std::cout << "label: " << ll[i] << std::endl;
	//}

        //for (int d = 0; d < dim_features; ++d)  
        //{ 
        //    if (d > height*width){
        //        arr[idx] = *(feature_blob_data + d);
        //        idx++;
        //    }
        //} 
        //int len = height*width; 
        //float min_val = *std::min_element (arr,arr+idx);
        //float max_val = *std::max_element(arr,arr+idx);  
        //std::cout << "size of feature:" << idx << ",max " << *std::max_element(arr,arr+len) << ",min " <<*std::min_element (arr,arr+len)<<std::endl;
        //for (int i=0;i<len;i++){
        //    arr[i] = 255*(arr[i]-min_val)/(max_val-min_val);
        //} 
        //cv::Mat img(cv::Size(width, height), CV_32FC1, arr);
	//cv::resize(img, img, cv::Size(640,480), 0, 0, CV_INTER_LINEAR);
	//cv::imwrite("mask.jpg", img);
	//float v[2][3] = {{1,0,16},{0,1,16}};
	//cv::Mat T_mat = cv::Mat(2, 3, CV_32F, v);
	//std::cout << T_mat << std::endl;
	//cv::Mat w_mask;
	//cv::warpAffine(img, w_mask, T_mat, cv::Size(640,480));
	//cv::imwrite("w_mask.jpg", w_mask);	
	 
    } 
}
cv::Mat CaffeDump::image_translation(cv::Mat & srcImage, int x0ffset, int y0ffset)
{
    int nRows = srcImage.rows;
    int nCols = srcImage.cols;
    cv::Mat resultImage(srcImage.size(), srcImage.type());
    //int nRows = srcImage.rows + abs(y0ffset);
    //int nCols = srcImage.cols + abs(x0ffset);
    //cv::Mat resultImage(nRows, nCols, srcImage.type());
    //遍历图像
    for (int i = 0; i < nRows; i++)
    {
        for (int j = 0; j < nCols; j++)
        {
            //映射变换
            int x = j - x0ffset;
            int y = i - y0ffset;
            //边界判断
            if (x >= 0 && y >= 0 && x < nCols && y < nRows)
            {
                resultImage.at<cv::Vec3b>(i, j) = srcImage.ptr<cv::Vec3b>(y)[x];
            }
        }
    }
    return resultImage;
}
void CaffeDump::preprocess(cv::Mat cv_image) {
  
    cv::resize(cv_image, cv_image, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);
    cv::Mat cv_new(cv_image.rows, cv_image.cols, CV_32FC3, cv::Scalar(0, 0, 0));
    int height = cv_image.rows;
    int width = cv_image.cols;
    /* Mean normalization (in this case it may not be the average of the training) */
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[0] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[0]);// - float(102.9801);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[1] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[1]);// - float(115.9465);
            cv_new.at<cv::Vec3f>(cv::Point(w, h))[2] = float(cv_image.at<cv::Vec3b>(cv::Point(w, h))[2]) ;//- float(122.7717);
        }
    }
    /* Max image size comparation to know if resize is needed */
    int max_side = MAX(height, width);
    int min_side = MIN(height, width);
    float max_side_scale = float(max_side) / MAX_INPUT_SIDE;
    float min_side_scale = float(min_side) / MIN_INPUT_SIDE;
    float max_scale = MAX(max_side_scale, min_side_scale);
    float img_scale = 1;
    if (max_scale > 1)
        img_scale = float(1) / max_scale;
    int height_resized = int(height * img_scale);
    int width_resized = int(width * img_scale);
    cv::Mat cv_resized;
    cv::resize(cv_new, cv_resized, cv::Size(width_resized, height_resized));
    std::cout << cv_resized.size() << std::endl;
    float data_buf[height_resized*width_resized * 3];
    for (int h = 0; h < height_resized; ++h)
    {
        for (int w = 0; w < width_resized; ++w)
        {
            data_buf[(0 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[0]);
            data_buf[(1 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[1]);
            data_buf[(2 * height_resized + h)*width_resized + w] = float(cv_resized.at<cv::Vec3f>(cv::Point(w, h))[2]);
        }
    }
    net_->blob_by_name("data")->Reshape(1, num_channels_, height_resized, width_resized);
    Blob<float> * input_blobs = net_->input_blobs()[0];
    switch (Caffe::mode()) {
        case Caffe::CPU:
            memcpy(input_blobs->mutable_cpu_data(), data_buf, sizeof(float) * input_blobs->count());
            break;
        case Caffe::GPU:
            caffe_gpu_memcpy(sizeof(float)* input_blobs->count(), data_buf, input_blobs->mutable_gpu_data());
            break;
        default:
            LOG(FATAL) << "Unknow Caffe mode";
    }
}
int main(int argc, char * argv[])
{
    if (argc < 2)
    {
        printf("Usage caffe_test <net.prototxt> <net.caffemodel> <inputFile_txt>\n");
        return -1;
    }
    int GPUID = 0;
    std::string  prototxt_file = argv[1];
    std::string caffemodel_file = argv[2];
    const char * input_files_path = argv[3];
    std::cout << "Reading the given prototxt file : " << prototxt_file << std::endl;
    std::cout << "Reading the given caffemodel file: " << caffemodel_file << std::endl;
    FILE * fs;
    char * image_path = NULL;
    size_t buff_size = 0;
    ssize_t read;
    fs = fopen(input_files_path, "r");
    if (!fs) {
        std::cout << "Unable to open the file." << input_files_path << std::endl;
        return -1;
    }
    CaffeDump dump(prototxt_file.c_str(), caffemodel_file.c_str(), GPUID);
    cv::Mat image = cv::imread(input_files_path, CV_LOAD_IMAGE_COLOR);
    cv::resize(image, image, cv::Size(640, 480), 0, 0, CV_INTER_LINEAR);
    dump.caffe_forward(image);
    return 0;
}
