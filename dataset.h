#ifndef _DATASET_H
#define _DATASET_H
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <dirent.h>
#include <stdlib.h>
#include <string.h>
using namespace std;

class DataSet;
typedef void (DataSet::*File_func)(const char* filename);


class DataSet{
    public:
    int         m_num; /* number of the feature */
    int         m_dim; /* dimension of the feature */

    /*! feature -> filename */
    vector<string> m_vfilename;/* file names */
    int*        m_featureFileIdx;
    float*      m_features;

    /* ground_truth files */
    int dir_num;             // the number of directory
    int* sort_file_assign;   // the feature file index sorted in increasing order
    int* file_elmt_num;      // the number of ftr_file in each dir
    int* file_start_idx;     // the start index of each dir (used in search)

    public:
    DataSet():m_num(0),m_dim(0),\
              m_featureFileIdx(NULL),m_features(NULL){
    }
    void readData(char* feature_dir);
    
    // Reads feature name files, generates ground truth file
    // Initial instance vars with memory
    bool initDataSet(const char* feature_dir, const char *feature_names);
    
    // Reads in features from files
    // Generates query feature file by randomly pick up one ftr in each dir
    void readFeatureFile(const char *filepath);
    
    float* getFtrs(){
        return m_features;
    }
    
    void freeFtrs() {
        if(m_features){
            free(m_features);
            m_features=NULL;
        }
    }

    string getFtrFileName(int fileIdx) {
        return m_vfilename[fileIdx];
    }
    int getFtrFileIdx(int ftrIdx){
        return m_featureFileIdx[ftrIdx];
    }

    int getFeatureDim() {
        return m_dim;
    }
    int getFeatureNum() {
        return m_num;
    }
    
    ~DataSet(){
        if(m_features){
            free(m_features);
            m_features=NULL;
        }
        if(m_featureFileIdx) {
            free(m_featureFileIdx);
            m_featureFileIdx=NULL;
        }
	if(sort_file_assign) {
            free(sort_file_assign);
            sort_file_assign=NULL;
        }
	if(file_elmt_num) {
            free(file_elmt_num);
            file_elmt_num=NULL;
        }
	if(file_start_idx) {
            free(file_start_idx);
            file_start_idx=NULL;
        }
 
 
 
    }
    private:
    void getHeader(const char* filename);
    // void readFeatureFile(const char *filepath);
    void explore_dir(const char*,File_func pfunc);
};

#endif /*_DATASET_H*/
