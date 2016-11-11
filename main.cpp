#include <time.h>
#include <iostream>
#include <sys/stat.h>
#include <stdlib.h>

#include "database.h"
#include "model.h"
#include "pqsearchengine.h"

void usage() {
    cout<<"usage:pqsearch [query_ftr_dir|query_list_file]"<<endl;
}

int main(int argc, char* argv[]){
  /*
	if (argc != 4) {
		std::cout << "./pqsearch gist_query.fvecs result_file gist_groundtruth.ivecs";
		return -1;
	}
  */
	//parameter
	int bucket_neig = 0;//multiple assignment
	int neighbor_num = 0;
	PqModel* model = new PqModel();
	char* model_file = "model_1280.dat";
	model->loadModel(model_file);
	std::cout<<"load model success!\n";
	DataBase* db = new DataBase();
	char* database_file = "database_1280.dat";
	db->loadDataBase(database_file);
	std::cout<<"load database success!\n";
	
	PqSearchEngine se;
	se.setDataBase(db);
	se.setModel(model);
	time_t start,end;

	bucket_neig = 1;
	se.setBuckNeigNum(bucket_neig);
	neighbor_num = 1280;
	se.setSiftNeigNum(neighbor_num);
	search_img(&se, argv[1], argv[2]);


	/*
	int a[3]={0,3,6};
	for(int i=0;i<3;i++){
		bucket_neig = pow(2,a[i]);
		std::cout<<"m_w="<<bucket_neig<<" ";
                se.setBuckNeigNum(bucket_neig);
		for(int i=0;i<=5;i++){
		    start = clock();
		    neighbor_num = pow(10,i);
		    cout<<"R="<<neighbor_num<<" ";
		    se.setSiftNeigNum(neighbor_num);
		    search_gistquery(&se, argv[1], argv[2],argv[3]);
		    end = clock();
		    std::cout<<" use time:"<<(end-start)/CLOCKS_PER_SEC<<std::endl;
		}
		std::cout<<endl;
	}
	

	if (model!=NULL){
		delete model;
		model = NULL;
	}
	std::cout<<"model mem released"<<endl;
	if (db!=NULL){
	  cout << "db != NULL" << endl;
		delete db;
		db = NULL;
	}
	std::cout<<"database mem released"<<endl;
	*/
	return 0;
}
