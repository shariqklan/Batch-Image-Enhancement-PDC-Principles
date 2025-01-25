//mpic++ -o image_process pdc_mpi.cpp `pkg-config --cflags --libs opencv4` -lmpi
//mpirun -np 4 ./image_process

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm>
#include <mpi.h>
#include <string>
#include <sys/statvfs.h>

using namespace std;
using namespace cv;

const int MAX_IMAGES = 30000; // Max number of images we expect to handle
const int MAX_IMAGE_SIZE = 50000; // Max size for images 

vector<Mat> images(MAX_IMAGES);
int numLoadedImages = 0; // to track how many images are loaded 


int operationSpecify[2]; // [0] -> Rotate angle, [1] -> Flip code


bool loadImage(int rank, const string& filePath) {
    Mat img = imread(filePath);
    if (img.empty()) {
        cerr << "Process " << rank << " failed to load image: " << filePath << endl;
        return false;
    }
    // Check if image size exceeds maximum size
    if (img.rows > MAX_IMAGE_SIZE || img.cols > MAX_IMAGE_SIZE) {
        cerr << "Process " << rank << " skipped large image: " << filePath << endl;
        return false;
    }

    images[numLoadedImages++] = img;
    cout << "Process " << rank << " loaded image: " << filePath << endl;
    return true;
}


void displayMenu() {
    cout << "\nWelcome to the Image Processing Program!" << endl;
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
    string option;

    while (selectedOperations.size() < numOptions) {
        cout << "Enter option " << selectedOperations.size() + 1 << ": ";
        cin >> option;
        transform(option.begin(), option.end(), option.begin(), ::tolower);

        if (option == "g" || option == "f" || option == "r" || option == "hc" ||
            option == "br" || option == "gb" || option == "ed" || option == "hb" ||
            option == "lc" || option == "lb") {

            if (find(selectedOperations.begin(), selectedOperations.end(), option) == selectedOperations.end()) {
                selectedOperations.push_back(option);
            } else {
                cout << "Option already selected. Please choose a different one." << endl;
            }
        } else {
            cout << "Invalid option! Please select from g, f, r, hc, lc, hb, lb, ed, gb, or br." << endl;
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

void imgFlipper(int flipCode, int i) {
    flip(images[i], images[i], flipCode);
}

void imgGrayer(int i) {
    cvtColor(images[i], images[i], COLOR_BGR2GRAY);
}

void histogramContraster(int i) {
    cvtColor(images[i], images[i], COLOR_BGR2YCrCb);
    vector<Mat> channels;
    split(images[i], channels);
    equalizeHist(channels[0], channels[0]);
    merge(channels, images[i]);
    cvtColor(images[i], images[i], COLOR_YCrCb2BGR);
}

void lowContraster(int i) {
    images[i].convertTo(images[i], -1, 0.5, 0);
}

void brightnesser(int i, const string& req) {
    if (req == "lb") {
        images[i].convertTo(images[i], -1, 1, -100);
    } else {
        images[i].convertTo(images[i], -1, 1, 100);
    }
}

void edger(int i) {
    Mat resultCanny;
    Canny(images[i], resultCanny, 80, 240);
    images[i] = resultCanny;
}

void imgGaussBlur(int i) {
    int kernel = 5;
    int sigMax = 13;
    GaussianBlur(images[i], images[i], Size(kernel, kernel), sigMax);
}

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


void saveImages(int rank, int startIndex, int endIndex) {
    string outputPath = "output/";
    for (int i = startIndex; i < endIndex; i++) {
        string outputFilePath = outputPath + "image_" + to_string(i) + ".jpg";
        if (!images[i].empty() && imwrite(outputFilePath, images[i])) {
            cout << "Process " << rank << " saved image: " << outputFilePath << endl;
        } else {
            cerr << "Process " << rank << " failed to save image: " << outputFilePath << endl;
        }
    }
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        displayMenu();
    }

    
    int numOperations;
    if (rank == 0) {
        cout << "Enter the number of options you want to perform: ";
        cin >> numOperations;
    }
    MPI_Bcast(&numOperations, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Get user operations and broadcast to all processes
    
    vector<string> selectedOperations;
    if (rank == 0) {
        selectedOperations = getUserOperations(numOperations);
    }
    int opCount = selectedOperations.size();
    MPI_Bcast(&opCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        selectedOperations.resize(opCount);
    }
    MPI_Bcast(selectedOperations.data(), opCount, MPI_CHAR, 0, MPI_COMM_WORLD);

    double startTime = MPI_Wtime(); 
    // Image loading and distribution across processes
    string folderPath = "images/";
    vector<string> fileList;
    glob(folderPath, fileList);
    int numImages = min((int)fileList.size(), MAX_IMAGES);

    // Check we don't exceed the number of images
    if (numImages >= MAX_IMAGES) {
        cerr << "Error: Exceeding maximum image limit of " << MAX_IMAGES << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Adjust chunk size to ensure all images are processed
    int chunkSize = (numImages + size - 1) / size; // Ceiling division
    int startIndex = rank * chunkSize;
    int endIndex = min((rank + 1) * chunkSize, numImages);

    // Load images into the images array for local processes
    for (int i = startIndex; i < endIndex; ++i) {
        if (!loadImage(rank, fileList[i])) {
            break; 
        }
    }

    // Synchronizing all processes
    MPI_Barrier(MPI_COMM_WORLD);

    
    for (int i = startIndex; i < endIndex; ++i) {
        for (const auto& op : selectedOperations) {
            if (op == "r") {
                imgRotater(operationSpecify[0], i);
            } else if (op == "f") {
                imgFlipper(operationSpecify[1], i);
            } else if (op == "g") {
                imgGrayer(i);
            } else if (op == "hc") {
                histogramContraster(i);
            } else if (op == "lc") {
                lowContraster(i);
            } else if (op == "hb" || op == "lb") {
                brightnesser(i, op);
            } else if (op == "gb") {
                imgGaussBlur(i);
            } else if (op == "br") {
                imgBGRemover(i);
            } else if (op == "ed") {
                edger(i);
            }
        }
    }

    
    saveImages(rank, startIndex, endIndex);

    double endTime = MPI_Wtime(); // End the timer on all ranks
    double elapsedTime = endTime - startTime;
    cout << endl << "Time Elapsed: "<<elapsedTime << " second" << endl;

    MPI_Finalize();
    return 0;
}