#include <iostream>
#include <time.h>
#include "featureflfmt.h"
#include "dataset.h"
#include "model.h"
#include "util.h"

extern "C" {
#include <yael/kmeans.h>
#include <yael/nn.h>
#include <yael/vector.h>
#include <yael/sorting.h>
#include <yael/machinedeps.h>
//#include <vl/kmeans.h>
}

using namespace std;
PqModel::PqModel(): m_coarse_k(0), m_coarse_centroids(NULL) {
    for(int i=0;i<NSQ;i++) {
        m_pq_centroids[i]=NULL;
    }
}

void PqModel::setCoarseK(int coarse_k) {
    m_coarse_k=coarse_k;
}

int PqModel::getCoarseK() {
    return m_coarse_k;
}
float* PqModel::getCoarseCentroid(){
    return m_coarse_centroids;
}
float* PqModel::getPqCentroid(int k) {
    return m_pq_centroids[k];
}

DataBase* PqModel::trainModel(DataSet*ds,char* centroid_file) {
    
    //cout << "->start trainning searching model..." << endl;
    //VL_PRINT ("<------vlfeat install!------>");
    DataBase *db = new DataBase();
    db->setFtrNum(ds->getFeatureNum());
    //db->setCoarseK(m_coarse_k);
    /*
    ifstream ifile(centroid_file,ios::binary);
    if(!ifile.good()){
        ifile.close();
        cout<<"->coarse index without coarse.centroids file !"<<endl;
        trainAndCoarseIdx(ds,db);// find m_coarse_assign
        writeCentroids(centroid_file);
    }
    else{
        ifile.close();
        cout<<"->coarse index with coarse.centroids file !"<<endl;
        trainAndCoarseIdxWithCentroid(ds,db,centroid_file);
    }
    */
  
    trainAndPq(ds,db);
  
    /*
    db->setFtrFileName(ds->m_vfilename);
    int* ftrFileIdx= ivec_new_0(ds->getFeatureNum());
    memcpy(ftrFileIdx,ds->m_featureFileIdx,ds->getFeatureNum());
    db->setFtrFileIdx(ftrFileIdx);
    */
    return db;
}

void PqModel::trainAndCoarseIdx(DataSet* dataset,DataBase* db) {
    if(m_coarse_k<=0) {
        cout<<"E-->m_coarse_k should not be:"<<m_coarse_k<<"\t";
        cout<<"please set m_coarse_k first"<<endl;
        exit(0);
    }

    int d       = dataset->getFeatureDim();
    int n       = dataset->getFeatureNum();
  
    if (2*n<m_coarse_k) {
        cout<<"->base feature num < coarse_k/2"<<endl;
        m_coarse_k=n/2;
    }
   
    /* cluster number should not be bigger than samples */
    int coarse_k= m_coarse_k;
    int niter   = 50;
    float * v   = dataset->getFtrs();
    
    cout<<"->train and coarse index..."<<endl;
    cout<<"->coarse k="<<coarse_k<<endl;
    cout<<"->dataset feature dim: "<<d<<", num: "<<n<<", coarsek: "<<coarse_k<<endl;
    
    /*! parameters for kmeans */
    //int nt      = count_cpu();
    int nt      = 1;
    cout<<"->cpu number:"<<nt<<endl;
    //int flags = 1;
    int flags   = nt | KMEANS_INIT_BERKELEY; //
    //int flags   = nt | KMEANS_INIT_RANDOM  ; 
    
    int* coarse_assign = ivec_new(n);// the index of every sample
    int* coarse_cluster_element_num = ivec_new_0(coarse_k); // the number of vec in every cluster
    if (m_coarse_centroids) free(m_coarse_centroids);
    m_coarse_centroids=fvec_new(d*coarse_k);// the centroids

    int redo    = 1;
    int seed    = 0;
    
    float err = kmeans(d, n, coarse_k, niter, v, flags, seed, redo, m_coarse_centroids, NULL, coarse_assign, coarse_cluster_element_num);
    cout<< "->coarse index kmeans err = "<<err<<endl;
    /*
    ofstream ofile("result_kmeans_1.txt", ios_base::out);
    if (!ofile) {
        cout << "File cannot open!" << endl;
	return;
    }
    int i = 0;
    for (i =0; i < n; i++) {
      ofile << i << " " << coarse_assign[i] << endl;
    }
    ofile.close(); 
    free(coarse_cluster_element_num);
    */
    //int* coarse_cluster_assign_idx = ivec_new(n); /* inverted file elements */
    //ivec_sort_index(coarse_assign,n,coarse_cluster_assign_idx );

    /* inverted file elements */
    //int* coarse_cluster_start_idx=ivec_new(m_coarse_k);
    /* compute the coarse cluster index */
    //coarse_cluster_start_idx[0]=0;
    //for (i=1;i<m_coarse_k;i++) {
    //   coarse_cluster_start_idx[i]=coarse_cluster_start_idx[i-1]+ coarse_cluster_element_num[i-1];
    //}

    db->setCoarseAssign(coarse_assign);
    //db->setCoarseClusterEleNum(coarse_cluster_element_num);
    //db->setCoarseClusterAssignIdx(coarse_cluster_assign_idx);
    //db->setCoarseClusterStartIdx(coarse_cluster_start_idx);
	//fprintf (stderr, "->kmeans err = %.3f\n", err);

     cout<<"->coarse index end!"<<endl;
}

void PqModel::trainAndCoarseIdxWithCentroid(DataSet* ds,DataBase*db,char* modelfile) {
    loadModel(modelfile);

    int* coarse_assign = ivec_new(ds->m_num);
    int* coarse_cluster_element_num = ivec_new_0(m_coarse_k); 

    int nq = ds->m_num;
    int nb = m_coarse_k;
    int d  = FEATURE_DIM;
    int w  = 1;
    int *assign = coarse_assign;
    float *base = m_coarse_centroids;
    float *query = ds->getFtrs();
    float* dis = fvec_new(w*nq);
    int distype = 2;
    /*! number of threads */
    //int nt=count_cpu();
    int nt=1;

    knn_full_thread(distype,nq,nb,d,w,base,query,NULL,assign,dis,nt);
    free(dis);

   // for (int i=0;i<ds->m_num;i++) {
   //     coarse_cluster_element_num[assign[i]]++;
   // }
   // int* coarse_cluster_assign_idx = ivec_new(ds->m_num); /* inverted file elements */
   // ivec_sort_index(coarse_assign,ds->m_num,coarse_cluster_assign_idx );

   // /* inverted file elements */
   // int* coarse_cluster_start_idx=ivec_new(m_coarse_k);
   // /* compute the coarse cluster index */
   // coarse_cluster_start_idx[0]=0;
   // for (int i=1;i<m_coarse_k;i++) {
   //     coarse_cluster_start_idx[i]=coarse_cluster_start_idx[i-1]+ coarse_cluster_element_num[i-1];
   // }
    db->setCoarseAssign(coarse_assign);
    //db->setCoarseClusterStartIdx(coarse_cluster_start_idx);
    //db->setCoarseClusterEleNum(coarse_cluster_element_num);
    //db->setCoarseClusterAssignIdx(coarse_cluster_assign_idx);
    return;
}

/*! product quantization */
void PqModel::trainAndPq(DataSet* ds,DataBase* db){
    cout << "->product quantization..." << endl;
    /*
    if(m_coarse_k==0) {
        cout<<"error:m_coarse_k should not be 0"<<"\t";
        cout<<"please set m_coarse_k"<<endl;
        exit(0);
    }
    */
    /*! compute the residual vector. */
    /*
    for (int i=0;i<ds->getFeatureNum();i++) {
        for (int j=0;j<ds->getFeatureDim();j++) { // the original vec will not be used
            (ds->getFtrs())[i*FEATURE_DIM+j] -= m_coarse_centroids[db->getCoarseAssign(i)*FEATURE_DIM+j];
        }
    }
    */
  

    /*! parameter for kmeans */
    
    int redo    = 1;
    int seed    = 0;
    int nt      =count_cpu();
  
    int flags   = nt | KMEANS_INIT_BERKELEY | KMEANS_NORMALIZE_CENTS; 
    int niter   = 50;
    
    float* subv = fvec_new(ds->getFeatureNum() * LSQ);

    for (int k = 0; k < NSQ; k++) {
        int* pq_assign      = ivec_new(ds->getFeatureNum()); // the centroid index of each vec
        m_pq_centroids[k]   = fvec_new(LSQ*KS);              // the centroids in this subquantizer, the number is KS, the dimension is LSQ
        for (int i = 0; i < ds->getFeatureNum(); i++) {      // get sub feature vec of each vec in this subquantizer
            for(int j = 0;j < LSQ; j++) {
                subv[i*LSQ + j]=(ds->getFtrs())[i*FEATURE_DIM + k*LSQ + j];
            }
        }
	time_t start = time(NULL);
	float err = kmeans(LSQ, ds->getFeatureNum(), KS, niter, subv, flags, seed, redo, m_pq_centroids[k], NULL, pq_assign, NULL);
	time_t end = time(NULL);
  	cout << "subquantizer" << k << ": err=" << err << endl;
	cout << "Using time: " << difftime(end, start)/3600.0 << " hour" << endl;
        db->setPqAssign(k,pq_assign);
    }
    free(subv);
    subv=NULL;
    cout<<"->product quantization ends. "<<endl;
    
}

DataBase* PqModel::getDataBase(DataSet* ds ){
    DataBase* db = new DataBase();
    db->setFtrNum(ds->getFeatureNum());
    db->setCoarseK(m_coarse_k);
    coarseIdx(ds,db);
    pq(ds,db);

    db->setFtrFileName(ds->m_vfilename);
    int* ftrFileIdx= ivec_new_0(ds->getFeatureNum());
    memcpy(ftrFileIdx,ds->m_featureFileIdx,ds->getFeatureNum());
    db->setFtrFileIdx(ftrFileIdx);
    return db;
}
void PqModel::coarseIdx(DataSet*ds ,DataBase*db){

    int feature_num=ds->getFeatureNum();
    int *coarse_assign = ivec_new(feature_num);
    int *coarse_cluster_element_num = ivec_new_0(m_coarse_k); 

    int nq = feature_num;
    int nb = m_coarse_k;
    int d  = FEATURE_DIM;
    int w  = 1;
    int *assign = coarse_assign;
    float *base = m_coarse_centroids;
    float *query = ds->getFtrs();
    float* dis = fvec_new(w*nq);
    int distype = 2;
    /* number of threads */
    //int nt=count_cpu();
    int nt=1;

    knn_full_thread(distype,nq,nb,d,w,base,query,NULL,assign,dis,nt);
    free(dis);

    //for (int i=0;i<ds->m_num;i++) {
    //    coarse_cluster_element_num[assign[i]]++;
    //}
    //int *coarse_cluster_assign_idx = ivec_new(ds->m_num); /* inverted file elements */
    //ivec_sort_index(coarse_assign,ds->m_num,coarse_cluster_assign_idx );

    ///* inverted file elements */
    //int* coarse_cluster_start_idx=ivec_new(m_coarse_k);
    ///* compute the coarse cluster index */
    //coarse_cluster_start_idx[0]=0;
    //for (int i=1;i<m_coarse_k;i++) {
    //   coarse_cluster_start_idx[i]=coarse_cluster_start_idx[i-1]+coarse_cluster_element_num[i-1];
    //}

    db->setCoarseAssign(assign);
    //db->setCoarseClusterStartIdx(coarse_cluster_start_idx);
    //db->setCoarseClusterEleNum(coarse_cluster_element_num);
    //db->setCoarseClusterAssignIdx(coarse_cluster_assign_idx);
    return ;
}

void PqModel::pq(DataSet* ds, DataBase* db) {
    cout<<"->product quantization..."<<endl;

    /*! compute the residual vector. */
    for (int i=0;i<ds->getFeatureNum();i++) {
        for (int j=0;j<ds->getFeatureDim();j++) {
            int coarse_assign_idx=db->getCoarseAssign(i);
            (ds->getFtrs())[i*FEATURE_DIM+j]-=m_coarse_centroids[coarse_assign_idx*FEATURE_DIM+j];
        }
    }

    /*! parameter for kmeans */
    float* subv = fvec_new(ds->getFeatureNum()*LSQ);

    for (int k=0;k<NSQ;k++) {
        for (int i=0;i<ds->getFeatureNum();i++) {
            for(int j=0;j<LSQ;j++) {
                subv[i*LSQ+j]=(ds->getFtrs())[i*FEATURE_DIM+k*LSQ+j];
            }
        }
        int nq = ds->getFeatureNum();
        int nb = KS;
        int d  = LSQ;
        int w  = 1;
        int *assign = ivec_new(nq);
        float *base = m_pq_centroids[k];
        float *query = subv;
        float* dis = fvec_new(w*nq);
        int distype = 2;
        /* number of threads */
        //int nt=count_cpu();
        int nt=1;

        knn_full_thread(distype,nq,nb,d,w,base,query,NULL,assign,dis,nt);
        free(dis);

        db->setPqAssign(k,assign);
    }
    free(subv);
    subv=NULL;
    cout<<"->product quantization end "<<endl;
}


void PqModel::saveModel(char*centroid_file){
  
    cout << "start saving model..." << endl;
    ofstream ofile(centroid_file);
    if(!ofile.good()){
        cout<<"[ERROR]: cann't open centroid_file:"<<centroid_file<<endl;
        exit(0);
    }
    //ofile.write((char*)&m_coarse_k,sizeof(int));
    //ofile.write((char*)m_coarse_centroids,m_coarse_k*FEATURE_LEN);
    for(int i=0;i<NSQ;i++) {//not scalable,but is OK.
        ofile.write((char*)m_pq_centroids[i],sizeof(float)*LSQ*KS);
    }
    ofile.close();
    return;
}

void PqModel::loadModel(char*centroid_file){

    cout << "start loading model..." << endl;
    ifstream ifile(centroid_file,ios::binary);
    if(!ifile.good()){
        cout<<"[ERROR]: No centroids file !"<<endl;
        ifile.close();
        exit(0);
    }
    //ifile.read((char*)&m_coarse_k,sizeof(int));
    //std::cout<<"m_coarse_k="<<m_coarse_k<<endl;
    //FREE(m_coarse_centroids);
    //m_coarse_centroids=fvec_new(FEATURE_DIM*m_coarse_k);

    //for(int i=0;i<m_coarse_k;i++) {
    //    ifile.read((char*)(m_coarse_centroids+i*FEATURE_DIM), FEATURE_LEN);
    //}
    //ifile.read((char*)(m_coarse_centroids),m_coarse_k*FEATURE_LEN);

    for(int i=0;i<NSQ;i++) {
        FREE(m_pq_centroids[i]);
        m_pq_centroids[i] = fvec_new(LSQ*KS*sizeof(float));
        ifile.read((char*)m_pq_centroids[i],sizeof(float)*LSQ*KS);
    }
    ifile.close();
    return;
}

void PqModel::predict(float* q_ar,int q_num,int neig_num,int *assign,int distype){//assign is q_num*neig_num
    /*since we use predict in multithread, 
     * multithread here is just a burden;
     */
    int thread_num = 1;
    float *dis = fvec_new(neig_num*q_num);//just 100*1000 for gist.
    knn_full_thread(distype,q_num,m_coarse_k,FEATURE_DIM,neig_num,m_coarse_centroids,q_ar,NULL,assign,dis,thread_num);
/*
   cout<<"neig_num="<<neig_num<<endl;//number of assigned centroids.
  //output the dist .ll
    
    int q =4;//the qth query feature.
    float *dist2 = new float[m_coarse_k]; 
    memset(dist2,0,sizeof(float)*m_coarse_k);
    for(int i=0;i<m_coarse_k;i++){
	//calculate the dist between qth and ith centroid.
	for(int d=0;d<FEATURE_DIM;d++)
	{
	      dist2[i] += (q_ar[FEATURE_DIM*q+d]-m_coarse_centroids[i*FEATURE_DIM+d])*(q_ar[FEATURE_DIM*q+d]-m_coarse_centroids[i*FEATURE_DIM+d]);
	}
   }
   for(int i=0;i<neig_num;i++){//bigger than bigger.
	int idx1 = assign[neig_num*q+i];
	cout<<idx1<<"  "<<dist2[idx1]<<" "<<dis[neig_num*q+i]<<endl;
   }
    delete[]dist2;
*/
    free(dis);
}

PqModel::~PqModel() {

    if(m_coarse_centroids) {
        //cout<<"free coarse centroids"<<endl;
        free( m_coarse_centroids);
        m_coarse_centroids=NULL;
    }
    int i=0;
    for (i=0;i<NSQ;i++) {
        free( m_pq_centroids[i]);
        m_pq_centroids[i]=NULL;
    }
}

