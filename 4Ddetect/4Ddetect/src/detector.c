#include "network.h"
#include "region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
//#include "demo.h"
#include "option_list.h"
#include <string.h>

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/core/core_c.h"
#include "opencv2/core/version.hpp" 
#include "opencv2/imgproc/imgproc_c.h"

#ifndef CV_VERSION_EPOCH
#include "opencv2/videoio/videoio_c.h"
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)"" CVAUX_STR(CV_VERSION_REVISION)
#pragma comment(lib, "opencv_world" OPENCV_VERSION ".lib")
#else
#define OPENCV_VERSION CVAUX_STR(CV_VERSION_EPOCH)"" CVAUX_STR(CV_VERSION_MAJOR)"" CVAUX_STR(CV_VERSION_MINOR)
#pragma comment(lib, "opencv_core" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_imgproc" OPENCV_VERSION ".lib")
#pragma comment(lib, "opencv_highgui" OPENCV_VERSION ".lib")
#endif

IplImage* draw_train_chart(float max_img_loss, int max_batches, int number_of_lines, int img_size);
void draw_train_loss(IplImage* img, int img_size, float avg_loss, float max_img_loss, int current_batch, int max_batches);
#endif    

static int coco_ids[] = {1,2,3,4,5,6,7,8,9,10,11,13,14,15,16,17,18,19,20,21,22,23,24,25,27,28,31,32,33,34,35,36,37,38,39,40,41,42,43,44,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,67,70,72,73,74,75,76,77,78,79,80,81,82,84,85,86,87,88,89,90};

static int get_coco_image_id(char *filename)
{
    char *p = strrchr(filename, '/');
    char *c = strrchr(filename, '_');
    if (c) p = c;
    return atoi(p + 1);
}

static void print_cocos(FILE *fp, char *image_path, detection *dets, int num_boxes, int classes, int w, int h)
{
    int i, j;
    int image_id = get_coco_image_id(image_path);
    for (i = 0; i < num_boxes; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        float bx = xmin;
        float by = ymin;
        float bw = xmax - xmin;
        float bh = ymax - ymin;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fp, "{\"image_id\":%d, \"category_id\":%d, \"bbox\":[%f, %f, %f, %f], \"score\":%f},\n", image_id, coco_ids[j], bx, by, bw, bh, dets[i].prob[j]);
        }
    }
}

void print_detector_detections(FILE **fps, char *id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2. + 1;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2. + 1;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2. + 1;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2. + 1;

        if (xmin < 1) xmin = 1;
        if (ymin < 1) ymin = 1;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            if (dets[i].prob[j]) fprintf(fps[j], "%s %f %f %f %f %f\n", id, dets[i].prob[j],
                xmin, ymin, xmax, ymax);
        }
    }
}

void print_imagenet_detections(FILE *fp, int id, detection *dets, int total, int classes, int w, int h)
{
    int i, j;
    for (i = 0; i < total; ++i) {
        float xmin = dets[i].bbox.x - dets[i].bbox.w / 2.;
        float xmax = dets[i].bbox.x + dets[i].bbox.w / 2.;
        float ymin = dets[i].bbox.y - dets[i].bbox.h / 2.;
        float ymax = dets[i].bbox.y + dets[i].bbox.h / 2.;

        if (xmin < 0) xmin = 0;
        if (ymin < 0) ymin = 0;
        if (xmax > w) xmax = w;
        if (ymax > h) ymax = h;

        for (j = 0; j < classes; ++j) {
            int class = j;
            if (dets[i].prob[class]) fprintf(fp, "%d %d %f %f %f %f %f\n", id, j + 1, dets[i].prob[class],
                xmin, ymin, xmax, ymax);
        }
    }
}

typedef struct {
    box b;
    float p;
    int class_id;
    int image_index;
    int truth_flag;
    int unique_truth_index;
} box_prob;

int detections_comparator(const void *pa, const void *pb)
{
    box_prob a = *(box_prob *)pa;
    box_prob b = *(box_prob *)pb;
    float diff = a.p - b.p;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

#ifdef OPENCV
typedef struct {
    float w, h;
} anchors_t;

int anchors_comparator(const void *pa, const void *pb)
{
    anchors_t a = *(anchors_t *)pa;
    anchors_t b = *(anchors_t *)pb;
    float diff = b.w*b.h - a.w*a.h;
    if (diff < 0) return 1;
    else if (diff > 0) return -1;
    return 0;
}

void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show)
{
    printf("\n num_of_clusters = %d, width = %d, height = %d \n", num_of_clusters, width, height);
    if (width < 0 || height < 0) {
        printf("Usage: darknet detector calc_anchors data/voc.data -num_of_clusters 9 -width 416 -height 416 \n");
        printf("Error: set width and height \n");
        return;
    }
    float *rel_width_height_array = calloc(1000, sizeof(float));

    list *options = read_data_cfg(datacfg);
    char *train_images = option_find_str(options, "train", "data/train.list");
    list *plist = get_paths(train_images);
    int number_of_images = plist->size;
    char **paths = (char **)list_to_array(plist);

    int number_of_boxes = 0;
    printf(" read labels from %d images \n", number_of_images);

    int i, j;
    for (i = 0; i < number_of_images; ++i) {
        char *path = paths[i];
        char labelpath[4096];
        replace_image_to_label(path, labelpath);

        int num_labels = 0;
        box_label *truth = read_boxes(labelpath, &num_labels);
        char buff[1024];
        for (j = 0; j < num_labels; ++j)
        {
            if (truth[j].x > 1 || truth[j].x <= 0 || truth[j].y > 1 || truth[j].y <= 0 ||
                truth[j].w > 1 || truth[j].w <= 0 || truth[j].h > 1 || truth[j].h <= 0)
            {
                printf("\n\nWrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f \n",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                sprintf(buff, "echo \"Wrong label: %s - j = %d, x = %f, y = %f, width = %f, height = %f\" >> bad_label.list",
                    labelpath, j, truth[j].x, truth[j].y, truth[j].w, truth[j].h);
                system(buff);
            }
            number_of_boxes++;
            rel_width_height_array = realloc(rel_width_height_array, 2 * number_of_boxes * sizeof(float));
            rel_width_height_array[number_of_boxes * 2 - 2] = truth[j].w * width;
            rel_width_height_array[number_of_boxes * 2 - 1] = truth[j].h * height;
            printf("\r loaded \t image: %d \t box: %d", i+1, number_of_boxes);
        }
    }
    printf("\n all loaded. \n");

    CvMat* points = cvCreateMat(number_of_boxes, 2, CV_32FC1);
    CvMat* centers = cvCreateMat(num_of_clusters, 2, CV_32FC1);
    CvMat* labels = cvCreateMat(number_of_boxes, 1, CV_32SC1);

    for (i = 0; i < number_of_boxes; ++i) {
        points->data.fl[i * 2] = rel_width_height_array[i * 2];
        points->data.fl[i * 2 + 1] = rel_width_height_array[i * 2 + 1];
    }


    const int attemps = 10;
    double compactness;

    enum {
        KMEANS_RANDOM_CENTERS = 0,
        KMEANS_USE_INITIAL_LABELS = 1,
        KMEANS_PP_CENTERS = 2
    };

    printf("\n calculating k-means++ ...");
    // Should be used: distance(box, centroid) = 1 - IoU(box, centroid)
    cvKMeans2(points, num_of_clusters, labels,
        cvTermCriteria(CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 10000, 0), attemps,
        0, KMEANS_PP_CENTERS,
        centers, &compactness);

    // sort anchors
    qsort(centers->data.fl, num_of_clusters, 2*sizeof(float), anchors_comparator);

    printf("\n");
    float avg_iou = 0;
    for (i = 0; i < number_of_boxes; ++i) {
        float box_w = points->data.fl[i * 2];
        float box_h = points->data.fl[i * 2 + 1];
        //int cluster_idx = labels->data.i[i];
        int cluster_idx = 0;
        float min_dist = FLT_MAX;
        for (j = 0; j < num_of_clusters; ++j) {
            float anchor_w = centers->data.fl[j * 2];
            float anchor_h = centers->data.fl[j * 2 + 1];
            float w_diff = anchor_w - box_w;
            float h_diff = anchor_h - box_h;
            float distance = sqrt(w_diff*w_diff + h_diff*h_diff);
            if (distance < min_dist) min_dist = distance, cluster_idx = j;
        }

        float anchor_w = centers->data.fl[cluster_idx * 2];
        float anchor_h = centers->data.fl[cluster_idx * 2 + 1];
        float min_w = (box_w < anchor_w) ? box_w : anchor_w;
        float min_h = (box_h < anchor_h) ? box_h : anchor_h;
        float box_intersect = min_w*min_h;
        float box_union = box_w*box_h + anchor_w*anchor_h - box_intersect;
        float iou = box_intersect / box_union;
        if (iou > 1 || iou < 0) { // || box_w > width || box_h > height) {
            printf(" Wrong label: i = %d, box_w = %d, box_h = %d, anchor_w = %d, anchor_h = %d, iou = %f \n",
                i, box_w, box_h, anchor_w, anchor_h, iou);
        }
        else avg_iou += iou;
    }
    avg_iou = 100 * avg_iou / number_of_boxes;
    printf("\n avg IoU = %2.2f %% \n", avg_iou);

    char buff[1024];
    FILE* fw = fopen("anchors.txt", "wb");
    printf("\nSaving anchors to the file: anchors.txt \n");
    printf("anchors = ");
    for (i = 0; i < num_of_clusters; ++i) {
        sprintf(buff, "%2.4f,%2.4f", centers->data.fl[i * 2], centers->data.fl[i * 2 + 1]);
        printf("%s", buff);
        fwrite(buff, sizeof(char), strlen(buff), fw);
        if (i + 1 < num_of_clusters) {
            fwrite(", ", sizeof(char), 2, fw);
            printf(", ");
        }
    }
    printf("\n");
    fclose(fw);

    if (show) {
        size_t img_size = 700;
        IplImage* img = cvCreateImage(cvSize(img_size, img_size), 8, 3);
        cvZero(img);
        for (j = 0; j < num_of_clusters; ++j) {
            CvPoint pt1, pt2;
            pt1.x = pt1.y = 0;
            pt2.x = centers->data.fl[j * 2] * img_size / width;
            pt2.y = centers->data.fl[j * 2 + 1] * img_size / height;
            cvRectangle(img, pt1, pt2, CV_RGB(255, 255, 255), 1, 8, 0);
        }

        for (i = 0; i < number_of_boxes; ++i) {
            CvPoint pt;
            pt.x = points->data.fl[i * 2] * img_size / width;
            pt.y = points->data.fl[i * 2 + 1] * img_size / height;
            int cluster_idx = labels->data.i[i];
            int red_id = (cluster_idx * (uint64_t)123 + 55) % 255;
            int green_id = (cluster_idx * (uint64_t)321 + 33) % 255;
            int blue_id = (cluster_idx * (uint64_t)11 + 99) % 255;
            cvCircle(img, pt, 1, CV_RGB(red_id, green_id, blue_id), CV_FILLED, 8, 0);
            //if(pt.x > img_size || pt.y > img_size) printf("\n pt.x = %d, pt.y = %d \n", pt.x, pt.y);
        }
        cvShowImage("clusters", img);
        cvWaitKey(0);
        cvReleaseImage(&img);
        cvDestroyAllWindows();
    }

    free(rel_width_height_array);
    cvReleaseMat(&points);
    cvReleaseMat(&centers);
    cvReleaseMat(&labels);
}
#else
void calc_anchors(char *datacfg, int num_of_clusters, int width, int height, int show) {
    printf(" k-means++ can't be used without OpenCV, because there is used cvKMeans2 implementation \n");
}
#endif // OPENCV

void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh,
                   float hier_thresh, int dont_show, int ext_output, int save_labels, char *photo)
{
    list *options = read_data_cfg(datacfg);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    char **names = get_labels_custom(name_list, &names_size); //get_labels(name_list);

    image **alphabet = load_alphabet();
    network net = parse_network_cfg_custom(cfgfile, 1); // set batch=1
    if(weightfile){
        load_weights(&net, weightfile);
    }
    //set_batch_network(&net, 1);
    fuse_conv_batchnorm(net);
    calculate_binary_weights(net);
    if (net.layers[net.n - 1].classes != names_size) {
        printf(" Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
            name_list, names_size, net.layers[net.n - 1].classes, cfgfile);
        if(net.layers[net.n - 1].classes > names_size) getchar();
    }
    srand(2222222);
    double time;
    char buff[256];
    char *input = buff;
    int j;
    float nms=.45;    // 0.4F
    while(1){
        if(filename){
            strncpy(input, filename, 256);
            if(strlen(input) > 0)
                if (input[strlen(input) - 1] == 0x0d) input[strlen(input) - 1] = 0;
        } else {
            printf("Enter Image Path: ");
            fflush(stdout);
            input = fgets(input, 256, stdin);
            if(!input) return;
            strtok(input, "\n");
        }
        image im = load_image(input,0,0,net.c);
        int letterbox = 0;
        //image sized = resize_image(im, net.w, net.h);
        image sized = letterbox_image(im, net.w, net.h); letterbox = 1;
        layer l = net.layers[net.n-1];

        box *boxes = calloc(l.w*l.h*l.n, sizeof(box));
        float **probs = calloc(l.w*l.h*l.n, sizeof(float *));
        for(j = 0; j < l.w*l.h*l.n; ++j) probs[j] = calloc(l.classes, sizeof(float *));

        float *X = sized.data;
        time= what_time_is_it_now();
        network_predict(net, X);
        network_predict_image(&net, im); letterbox = 1;
        printf("%s: Predicted in %f seconds.\n", input, (what_time_is_it_now()-time));
        get_region_boxes(l, 1, 1, thresh, probs, boxes, 0, 0);
		
        int nboxes = 0;
        detection *dets = get_network_boxes(&net, im.w, im.h, thresh, hier_thresh, 0, 1, &nboxes, letterbox);
		if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		draw_detection(im, dets, nboxes, thresh, names, alphabet, l.classes);
		//char a[100] = "C:\\Users\\24221\\Desktop\\";
		//strcat(a, "data");
		//strcat(a, "\\output");
        save_image(im, "output");
        //if (!dont_show) {
          //  show_image(im, "output");
       // }
        if(save_labels)
        {
            char labelpath[4096];
            replace_image_to_label(input, labelpath);

            FILE* fw = fopen(labelpath, "wb");
            int i;
            for (i = 0; i < nboxes; ++i) {
                char buff[1024];
                int class_id = -1;
                float prob = 0;
                for (j = 0; j < l.classes; ++j) {
                    if (dets[i].prob[j] > thresh && dets[i].prob[j] > prob) {
                        prob = dets[i].prob[j];
                        class_id = j;
                    }
                }
                if (class_id >= 0) {
                    sprintf(buff, "%d %2.4f %2.4f %2.4f %2.4f\n", class_id, dets[i].bbox.x, dets[i].bbox.y, dets[i].bbox.w, dets[i].bbox.h);
                    fwrite(buff, sizeof(char), strlen(buff), fw);
                }
            }
            fclose(fw);
        }

        free_detections(dets, nboxes);
        free_image(im);
        free_image(sized);

#ifdef OPENCV
        if (!dont_show) {
            cvWaitKey(0);
            cvDestroyAllWindows();
        }
#endif
        if (filename) break;
    }

    free_ptrs(names, net.layers[net.n - 1].classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i;
    const int nsize = 8;
    for (j = 0; j < nsize; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);
    free_network(net);
}

void run_detector(int argc, char **argv)
{
    int dont_show = find_arg(argc, argv, "-dont_show");
    int show = find_arg(argc, argv, "-show");
    int http_stream_port = find_int_arg(argc, argv, "-http_port", -1);
    char *out_filename = find_char_arg(argc, argv, "-out_filename", 0);
    char *outfile = find_char_arg(argc, argv, "-out", 0);
    char *prefix = find_char_arg(argc, argv, "-prefix", 0);
    float thresh = find_float_arg(argc, argv, "-thresh", .25);    // 0.24
    float hier_thresh = find_float_arg(argc, argv, "-hier", .5);
    int cam_index = find_int_arg(argc, argv, "-c", 0);
    int frame_skip = find_int_arg(argc, argv, "-s", 0);
    int num_of_clusters = find_int_arg(argc, argv, "-num_of_clusters", 5);
    int width = find_int_arg(argc, argv, "-width", -1);
    int height = find_int_arg(argc, argv, "-height", -1);

    int ext_output = find_arg(argc, argv, "-ext_output");
    int save_labels = find_arg(argc, argv, "-save_labels");
   /* if(argc < 4){
        fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
        return;
    }*/
    char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
    int *gpus = 0;
    int gpu = 0;
    int ngpus = 0;
    if(gpu_list){
        printf("%s\n", gpu_list);
        int len = strlen(gpu_list);
        ngpus = 1;
        int i;
        for(i = 0; i < len; ++i){
            if (gpu_list[i] == ',') ++ngpus;
        }
        gpus = calloc(ngpus, sizeof(int));
        for(i = 0; i < ngpus; ++i){
            gpus[i] = atoi(gpu_list);
            gpu_list = strchr(gpu_list, ',')+1;
        }
    } else {
        gpu = gpu_index;
        gpus = &gpu;
        ngpus = 1;
    }
    int clear = find_arg(argc, argv, "-clear");
    char *datacfg = "coco.data";
    char *cfg = "yolov3.cfg";
    char *weights = "yolov3.weights";
    if(weights)
        if(strlen(weights) > 0)
            if (weights[strlen(weights) - 1] == 0x0d) weights[strlen(weights) - 1] = 0;
    char *filename = argv[1];
    test_detector(datacfg, cfg, weights, filename, thresh, hier_thresh, dont_show, ext_output, save_labels, argv[1]);
}
