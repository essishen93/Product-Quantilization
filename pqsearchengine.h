#ifndef _PQSEARCHENGINE
#define _PQSEARCHENGINE
#include <vector>
#include <string>
#include <stdlib.h>
#include <ctype.h>
#include "featureflfmt.h"
#include <iostream>

#include "dataset.h"
#include "model.h"

using namespace std;

class PqSearchEngine {
    public:
        DataBase*     m_db;
        PqModel*      m_model;

        /* search parameter */
        int         m_sift_neighbor_num;
        int         m_pic_neighbor_num;
        int         m_w;            /*bucket neighbor */
    
        /* ground_truth files */
        int dir_num;
        int* sort_file_assign;
        int* file_elmt_num;
        int* file_start_idx;

    public:
        PqSearchEngine();  //vector reserved
        /*
        void trainModel(DataSet* dataset,char* cenroid_file=NULL);
        void addAdditionalDataSet(DataSet* dataset);
        void init(const char* base_data_dir,int coarsek,char* centorid_file,char*engine_file){
            m_ds.initDataSet(base_data_dir);
            m_ds.readData(base_data_dir);
            
            m_model.setCoarseK(coarsek);
            //char *centorid_file="centroids.txt";
            m_model.trainModel(&m_ds,centorid_file);
            //char *engine_file="search_engine.txt";
            writeEngine(engine_file);
        }
        */

        ~PqSearchEngine() {
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

        void setDataBase(DataBase*db){
            m_db=db;
        }
        DataBase* getDataBase(){
            return m_db;
        }
        void setModel(PqModel* model) {
            m_model=model;
        }
        PqModel* getModel() {
            return m_model;
        }
        void setSiftNeigNum(int num){
            m_sift_neighbor_num=num;
        }
        int getSiftNeigNum(){
            return m_sift_neighbor_num;
        }
        void setPicNeigNum(int num){
            m_pic_neighbor_num=num;
        }
        int getPicNeigNum(){
            return m_pic_neighbor_num;
        }
        void setBuckNeigNum(int num) {
            m_w = num;            /*bucket neighbor */
        }
        int getBuckNeigNum() {
            return m_w ;            /*bucket neighbor */
        }

        void writeEngine(const char*);
        bool loadEngine(const char *);
        void getDistances(int cluster_idx,float**dis_tab);
};


typedef struct search_thread_para{
    string search_result_file;
    PqSearchEngine* se;
    vector<string> files;
}SearchThreadPara;
// k nearest neighbour search.
void search_withfile(PqSearchEngine* se,char* query_file);
void search_withdir(PqSearchEngine* se,char* query_dir);
void* search_file(void* searchThreadPara);
//bool search_gistquery(PqSearchEngine* search_engine, char* queryfile,char*resultfile,char* ground_truefile=NULL);
void search_img(PqSearchEngine* search_engine, const char *ftr_file, const char *grdTruth_file);
bool isTrueIdx(int* truth_idx, int idx, int num);
#endif /* PQSEARCHENGINE */

