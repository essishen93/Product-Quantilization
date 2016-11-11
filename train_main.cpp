#include "database.h"
#include "dataset.h"
#include "model.h"
#include <iostream>
#include <time.h>
using namespace std;

void usage() {
    cout<<"usage:pqtrain feature_dir"<<endl;
}

int main(int args, char* argv[]) {
    /*
    if (args!=2){
        usage();
        return -1;
    }
    */
  
 
    DataSet* ds = new DataSet();
    ds->initDataSet(argv[1], argv[2]); // argv[1] = file path, argv[2] = image names (txt)
    
    char filePath[512];  // feature.bin
    sprintf(filePath, "%s/%s", argv[1], argv[3]);    // argv[3] = feature_vec (bin)
    ds->readFeatureFile(filePath);
    
    
    PqModel model;
    //int coarse_k = 320;
    //model.setCoarseK(coarse_k);

    char* centroid_file = NULL;


    time_t start = time(NULL);
    DataBase* db = model.trainModel(ds, centroid_file);
    time_t end = time(NULL);


    cout << "Database indexing is done. Time is " << difftime(end, start)/3600.0 << " hour"  << endl;
    
    char* model_file = "model_1280.dat";
    char* database_file = "database_1280.dat";
    model.saveModel(model_file);
    db->saveDataBase(database_file);
    
    delete db;
    
    delete ds;
 
    

    return 0;
}
