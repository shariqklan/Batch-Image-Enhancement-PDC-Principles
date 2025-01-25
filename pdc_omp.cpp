//g++ -o pdc_omp pdc_omp.cpp `pkg-config --cflags --libs opencv4` -fopenmp
// ./pdc_omp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <cctype>
#include <omp.h> 
#include <unistd.h>

using namespace std;
using namespace cv;

const int MAX_IMAGES = 10000;


vector<omp_lock_t> guardians;
int operationSpecify[2]; 



Mat images[MAX_IMAGES];

int numLoadedImages = 0;  


void loadImage(int index, string filePath) {
    Mat img = imread(filePath);
    if (!img.empty()) {

        #pragma omp critical
        {
            if (numLoadedImages < MAX_IMAGES) {
                images[numLoadedImages++] = img;
            }
        }
        cout << "Loaded image: " << filePath << endl;
    } else {

        cerr << "Failed to load image: " << filePath << endl;
    }
}


void displayMenu() {
    cout << "\n\nWelcome to Image Processing Program!" << endl;
    cout << "Select the operations you want to perform: " << endl;
    cout << "g: Grayscale\n";
    cout << "f: Flip\n";
    cout << "r: Rotate\n";
    cout << "hc: High Contrast | lc: Low Contrast\n";
    cout << "hb: High Brightness | lb: Low Brightness\n";
    cout << "gb: Gaussian Blur | br: Background Remover | ed: Edge Detector\n\n";
}


vector<string> getUserOperations(int numOptions) {
    vector<string> selectedOperations;
    int index = 0;
    while (selectedOperations.size() < numOptions) { 
        string option;
        cout << "Enter option " << selectedOperations.size() + 1 << ": ";
        cin >> option;

        
        transform(option.begin(), option.end(), option.begin(), ::tolower);

        if (option == "g" || option == "f" || option == "r" || option == "hc" || option == "br" || option == "gb" || option == "ed" || option == "hb" || option == "lc" || option == "lb") {
            bool isDuplicate = false;
           
            for (auto iter = 0; iter < index; ++iter) {
                if (option == selectedOperations[iter]) {
                    isDuplicate = true;
                    break;
                }
            }

            if (isDuplicate) {
                cout << "\nOption already selected. Please try another one." << endl;
                continue;  
            }

            if (option == "f") {
                int reqFlip;
                cout << "Enter flip code (0 for vertical, 1 for horizontal, -1 for both): ";
                cin >> reqFlip;
                operationSpecify[1] = reqFlip;
            }
            if (option == "r") {
                int reqAngle;
                cout << "Enter angle to rotate all your images by (Anticlockwise): ";
                cin >> reqAngle;
                operationSpecify[0] = reqAngle;
            }

            selectedOperations.push_back(option); 
            index++;
        } else { 
            cout << "Invalid option! Please select from g, f, r, hc, lc, hb, lb, ed, br, gb" << endl;
        }
    }
    return selectedOperations; 
}




void imgRotater(int angle, int i) { 
    int height = images[i].rows;
    int width = images[i].cols;

    
    Point2f image_center(width / 2.f, height / 2.f);

    
    Mat rotation_mat = getRotationMatrix2D(image_center, angle, 1.0);

    
    double abs_cos = abs(rotation_mat.at<double>(0, 0));
    double abs_sin = abs(rotation_mat.at<double>(0, 1));

    
    int new_width = int(height * abs_sin + width * abs_cos);
    int new_height = int(height * abs_cos + width * abs_sin);

    
    rotation_mat.at<double>(0, 2) += new_width / 2.0 - image_center.x;
    rotation_mat.at<double>(1, 2) += new_height / 2.0 - image_center.y;

    
    
    Mat rotated_mat;
    warpAffine(images[i], rotated_mat, rotation_mat, Size(new_width, new_height));

    images[i] = rotated_mat;
}

// Flip image
void imgFlipper(int flipCode, int i) {
    flip(images[i], images[i], flipCode);
}

// Grayscale image
void imgGrayer(int i) {
    cvtColor(images[i], images[i], COLOR_BGR2GRAY);
}

// High contrast image
void histogramContraster(int i) { 
    cvtColor(images[i], images[i], COLOR_BGR2YCrCb); 

    vector<Mat> channels; 
    split(images[i], channels);

    
    equalizeHist(channels[0], channels[0]);

    
    merge(channels, images[i]);

    
    cvtColor(images[i], images[i], COLOR_YCrCb2BGR);
}

// Low contrast image
void lowContraster(int i) {
    images[i].convertTo(images[i], -1, 0.5, 0); 
}

// High and Low Brightness image
void brightnesser(int i, string req) {
    if (req == "lb") {
        images[i].convertTo(images[i], -1, 1, -100);
    } else { // high brightness
        images[i].convertTo(images[i], -1, 1, 100);
    }
}

// Edge Detection image
void edger(int i) {
    Mat resultCanny;
    Canny(images[i], resultCanny, 80, 240);
    images[i] = resultCanny;
}

// Gaussian Blur image
void imgGaussBlur(int i) {
    int kernel = 5; 
    int sigMax = 13; 
    GaussianBlur(images[i], images[i], Size(kernel, kernel), sigMax);
}

// Background Remove image
void imgBGRemover(int i) { 
    Mat hsvImage;
    cvtColor(images[i], hsvImage, COLOR_BGR2HSV); 

    Scalar lowerBound(0, 100, 100); 
    Scalar upperBound(10, 255, 255);

    
    Mat mask;
    inRange(hsvImage, lowerBound, upperBound, mask);

    Mat foreground;
    images[i].copyTo(foreground, mask);

    images[i] = foreground;
}


int getOptimalChunkSize(int numImages) {
    int optimal, minimum = 2;
    int coreCount = omp_get_max_threads(); 
    optimal = max(minimum, (numImages / coreCount));

    if (numImages < (coreCount * 2)) return 1;  
    return optimal;
}


void runner(int id, vector<string> Op, int num, int startIndex, int endIndex) {
    // Iterate through each image in the chunk and apply each operation
    #pragma omp parallel for
    for (int i = startIndex; i < endIndex; i++) {
        for (int j = 0; j < num; j++) {
            
            string task = Op[j];
            omp_set_lock(&guardians[i]);
            if (task == "r") {
                imgRotater(operationSpecify[0], i);
            } else if (task == "f") {
                imgFlipper(operationSpecify[1], i);
            } else if (task == "g") {
                imgGrayer(i);
            } else if (task == "hc") {
                histogramContraster(i);
            } else if (task == "lc") {
                lowContraster(i);
            } else if (task == "lb" || task == "hb") {
                brightnesser(i, task);
            } else if (task == "ed") {
                edger(i);
            } else if (task == "br") {
                imgBGRemover(i);
            } else if (task == "gb") {
                imgGaussBlur(i);
            }
            omp_unset_lock(&guardians[i]);
        }
    }
}

int main() {
    displayMenu();
    int num;
    cout << "Enter the number of options you want to perform: ";
    cin >> num;

    if (num < 0 || num > 10) {
        cout << "Please enter correct number of options\n";
        return 0;
    }

    vector<string> selectedOperations = getUserOperations(num);

    cout << "Selected operations: " << endl;
    for (string op : selectedOperations) {
        cout << op << " ";
    }
    cout << endl;

    
    string folderPath = "images/";
    vector<string> fileList;
    glob(folderPath, fileList);  // Fill the fileList with image files from the folder

    int numImages = min((int)fileList.size(), MAX_IMAGES);

    guardians.resize(numImages);
    for (int i = 0; i < numImages; ++i) {
        omp_init_lock(&guardians[i]);
    }

    double processStartload = omp_get_wtime();
    // Load images concurrently
    #pragma omp parallel for
    for (int i = 0; i < numImages; ++i) {
        loadImage(i, fileList[i]);
    }
    double processEndload = omp_get_wtime();

    int optimalChunkSize = getOptimalChunkSize(numImages);

    double processStart = omp_get_wtime();

    // Process images in parallel
    #pragma omp parallel
    {
        int numThreads = omp_get_num_threads();
        int chunkSize = max(optimalChunkSize, numImages / numThreads);
        
        #pragma omp for
        for (int i = 0; i < numImages; i += chunkSize) {
            runner(omp_get_thread_num(), selectedOperations, num, i, min(i + chunkSize, numImages));
        }
    }

    double processEnd = omp_get_wtime();
    

    string outputPath = "output/";

    double processStartsave = omp_get_wtime();
     #pragma omp parallel for
    for (int i = 0; i < numImages; i++) {
        
        string outputFilePath = outputPath + "image_" + to_string(i) + ".jpg";
        
        // Save the image to the output path
        if (imwrite(outputFilePath, images[i])) {
            cout << "Saved image: " << outputFilePath << endl;
        } else {
            cerr << "Error saving image: " << outputFilePath << endl;
        }
    }
    double processEndsave = omp_get_wtime();

    cout << "Number of images processed: " << numImages << endl;
    cout << "All images saved to output directory." << endl;
    cout << "Total time taken for processing images: " << (processEnd - processStart) << " seconds" << endl;
    cout << "Total time taken for saving images to output file: " << (processEndsave - processStartsave) << " seconds" << endl;
    cout << "Total time taken for loading images from images file to array: " << (processEndload - processStartload) << " seconds" << endl;
    cout << "\n\nThank You for choosing our application!\n";
    return 0;
}
