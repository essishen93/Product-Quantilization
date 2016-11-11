#include "database.h"
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>
#include <fstream>
#include "util.h"
extern "C" {
#include <yael/kmeans.h>
#include <yael/nn.h>
#include <yael/vector.h>
#include <yael/sorting.h>
#include <yael/machinedeps.h>
}

using namespace std;
DataBase::DataBase(){
    m_feature_num=0;
    m_coarse_k=0;
    m_coarse_assign=NULL; 
    m_coarse_cluster_start_idx=NULL; 
    m_coarse_cluster_element_num=NULL;
    m_coarse_cluster_assign_idx=NULL;
    for (int i=0;i<NSQ;i++) {
        m_pq_assign[i]=NULL;
    }
    m_featureFileIdx=NULL;
}

void DataBase::setFtrNum(int feature_num) {
    m_feature_num = feature_num;
}
int DataBase::getFtrNum() {
    return m_feature_num;
}
void DataBase::setCoarseK(int coarse_k){
    m_coarse_k=coarse_k;
}
int  DataBase::getCoarseK(){
    return m_coarse_k;
}
int DataBase::getFtrFileNum() {
    return m_vfilename.size();
}
string DataBase::getFtrFileName(int fileIdx) {
    return m_vfilename[fileIdx];
}
void DataBase::setFtrFileName(vector<string> ftrfilename) {
    m_vfilename=ftrfilename;
}
int DataBase::getFtrFileIdx(int ftrIdx){
    return m_featureFileIdx[ftrIdx];
}
void DataBase::setFtrFileIdx(int* ftrfileidx) {
    m_featureFileIdx = ftrfileidx;
}

void DataBase::setPqAssign(int idx,int* assign) {
    FREE(m_pq_assign[idx]);
    m_pq_assign[idx]=assign;
}

int  DataBase::getPqAssign(int sub_idx,int base_idx){
    return m_pq_assign[sub_idx][base_idx];
}

void DataBase::setCoarseAssign(int* coarse_assign){
    FREE(m_coarse_assign);
    m_coarse_assign=coarse_assign;
    buildIvf();
}
int DataBase::getCoarseAssign(int feature_idx){
    return m_coarse_assign[feature_idx];
}

void DataBase::setCoarseClusterStartIdx(int* idx) {
    FREE(m_coarse_cluster_start_idx);
    m_coarse_cluster_start_idx=idx;
}
int DataBase::getCoarseClusterStartIdx(int idx) {
    return m_coarse_cluster_start_idx[idx];
}

void DataBase::setCoarseClusterEleNum(int *en) {
    FREE(m_coarse_cluster_element_num);
    m_coarse_cluster_element_num=en;
}
int DataBase::getCoarseClusterEleNum(int cluster_idx) {
    return m_coarse_cluster_element_num[cluster_idx];
}

void DataBase::setCoarseClusterAssignIdx(int *idx) {
    FREE(m_coarse_cluster_assign_idx);
    m_coarse_cluster_assign_idx=idx;
}

int DataBase::getCoarseClusterAssignIdx(int idx) {
    return m_coarse_cluster_assign_idx[idx];
}

bool DataBase::merge(DataBase* db) {
    assert(m_coarse_k==db->getCoarseK());
    int new_feature_num=m_feature_num + db->getFtrNum();
    int new_coarse_k = m_coarse_k;
    int* new_pq_assign[NSQ];
    for (int i = 0;i<NSQ;i++) {
        new_pq_assign[i]= ivec_new_0(new_feature_num);
        memcpy(new_pq_assign[i],m_pq_assign[i],m_feature_num);
        memcpy(new_pq_assign[i]+m_feature_num,m_pq_assign[i],db->getFtrNum());
    }

    int* new_coarse_assign=ivec_new_0(new_feature_num);
    memcpy(new_coarse_assign,m_coarse_assign,m_feature_num);
    memcpy(new_coarse_assign+m_feature_num,db->m_coarse_assign,db->getFtrNum());
    
    vector<string> filenames;
    filenames.reserve(1000);
    filenames.insert(filenames.begin(),m_vfilename.begin(),m_vfilename.end());
    filenames.insert(filenames.end(),db->m_vfilename.begin(),m_vfilename.end());

    int* new_featureFileIdx = ivec_new_0(filenames.size());
    memcpy(new_featureFileIdx,m_featureFileIdx,m_feature_num);
    for (int i =0 ;i<db->m_vfilename.size();i++) {
        new_featureFileIdx[i+m_vfilename.size()]=db->m_featureFileIdx[i];
    }

    setFtrNum(new_feature_num);
    for (int i = 0;i<NSQ;i++) {
        setPqAssign(i,new_pq_assign[i]);
    }
    setCoarseAssign(new_coarse_assign);
    setFtrFileName(filenames);
    setFtrFileIdx(new_featureFileIdx);
    return true;
}

void DataBase::buildIvf(){
    FREE(m_coarse_cluster_element_num);
    m_coarse_cluster_element_num = ivec_new_0(m_coarse_k);
    int i = 0;
    for (i = 0;i<m_feature_num;i++) {
        m_coarse_cluster_element_num[m_coarse_assign[i]]++;
    }
    /*
    ofstream ofile("result_database_elmt_num.txt", ios_base::out);
    if (!ofile) {
        cout << "elmt_num cannot open" << endl;
	return;
    }
    for (i = 0; i < m_coarse_k; i++) {
      ofile << i << " " << m_coarse_cluster_element_num[i] <<endl;
    }
    ofile.close();
    */
    FREE(m_coarse_cluster_assign_idx);
    m_coarse_cluster_assign_idx = ivec_new(m_feature_num);
    ivec_sort_index(m_coarse_assign,m_feature_num, m_coarse_cluster_assign_idx);
    /*
    ofstream bfile("result_database_assign_idx.txt", ios_base::out);
    if (!bfile) {
        cout << "assgin_idx cannot open" << endl;
	return;
    }
    for (i = 0; i < m_feature_num; i++) {
      bfile << i << " " << m_coarse_cluster_assign_idx[i] <<endl;
    }
    bfile.close();
    */
    FREE(m_coarse_cluster_start_idx);
    m_coarse_cluster_start_idx = ivec_new(m_coarse_k);
    m_coarse_cluster_start_idx[0]=0;
    for (i =0;i<m_coarse_k;i++) {
       m_coarse_cluster_start_idx[i] =  m_coarse_cluster_start_idx[i-1] +  m_coarse_cluster_element_num[i-1];
    }
    /*
    ofstream qfile("result_database_start_idx.txt", ios_base::out);
    if (!qfile) {
        cout << "start_idx cannot open" << endl;
	return;
    }
    for (i = 0; i < m_coarse_k; i++) {
      qfile << i << " " << m_coarse_cluster_start_idx[i] <<endl;
    }
    qfile.close();
    */
}

void DataBase::loadDataBase(char* dbfile) {
  
    cout << "->start loading database..." << endl;
    ifstream ifile(dbfile,ios::in);
    if(!ifile.good()){
        cout<<"[ERROR]: No database file:"<<dbfile<<endl;
        ifile.close();
        exit(0);
    }
    ifile.read((char*)&m_feature_num,sizeof(int));
    /*
    ifile.read((char*)&m_coarse_k,sizeof(int));
    if (m_coarse_assign) free(m_coarse_assign);
    FREE(m_coarse_assign);
    assert(m_coarse_assign==NULL);
    m_coarse_assign=ivec_new(m_feature_num);
    ifile.read((char*)(m_coarse_assign), m_feature_num*sizeof(int));
    buildIvf();
    */
    /* read pq assign */
    if (m_coarse_assign) free(m_coarse_assign);
    for(int i=0;i<NSQ;i++) {
        if (m_pq_assign[i]) free(m_pq_assign[i]);
        m_pq_assign[i]=ivec_new(m_feature_num);
        ifile.read((char*)m_pq_assign[i],sizeof(int)*m_feature_num);
    }

    /* read feature file index */
    /*
    if (m_featureFileIdx) free(m_featureFileIdx);
    m_featureFileIdx=ivec_new(m_feature_num);
    ifile.read((char*)m_featureFileIdx,sizeof(int)*m_feature_num);
    */
    /* read feature file names */
    /*
    int feature_file_num = 0;
    ifile.read((char*)&feature_file_num ,sizeof(int));
    for(int i = 0;i<feature_file_num;i++){
        string tmp;
        ifile>>tmp;
        m_vfilename.push_back(tmp);
    }
    */
    ifile.close();
    return;
}

void DataBase::saveDataBase(char* dbfile) {
    
    cout << "->start saving database..." << endl;
    ofstream ofile(dbfile,ios::out);
    if(!ofile.good()){
        cout<<"[ERROR]: cann't open database file:"<<dbfile<<endl;
        exit(0);
    }
    ofile.write((char*)&m_feature_num,sizeof(int));
    
    //ofile.write((char*)&m_coarse_k,sizeof(int));
    //ofile.write((char*)m_coarse_assign,m_feature_num*sizeof(int));
    for(int i=0;i<NSQ;i++) {
        ofile.write((char*)m_pq_assign[i], sizeof(int)*m_feature_num);
    }
    

    //ofile.write((char*)m_featureFileIdx, m_feature_num*sizeof(int));
    /*    int filename_size = m_vfilename.size();

    ofile.write((char*)&filename_size,sizeof(int));
    for(int i=0;i<filename_size;i++) {
        ofile<<m_vfilename[i]<<endl;
    }
    */
    ofile.close();
    return;
}

DataBase::~DataBase(){
    for (int i=0;i<NSQ;i++) {
        FREE( m_pq_assign[i]);
    }
    FREE(m_coarse_assign);
    FREE(m_coarse_cluster_element_num); 
    FREE(m_coarse_cluster_assign_idx);
    FREE(m_coarse_cluster_start_idx);
    FREE(m_featureFileIdx);
}

