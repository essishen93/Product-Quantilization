#include "dataset.h"
#include <unistd.h>
#include <cstring>
#include <cstdlib>
#include "featureflfmt.h"

extern "C" {
#include <yael/kmeans.h>
#include <yael/nn.h>
#include <yael/vector.h>
#include <yael/sorting.h>
#include <yael/machinedeps.h>
}

void DataSet::explore_dir(const char* dir,File_func pfunc) {
    DIR             *pDir;
    struct dirent   *ent;

    pDir=opendir(dir);

    char filepath[512];

    while(NULL!=(ent=readdir(pDir))) {
      cout << "in loop" << endl;
        if (ent->d_type&DT_DIR) {
            continue;
        }
        else {
            sprintf(filepath,"%s/%s",dir,ent->d_name);
            (this->*pfunc)(filepath);
        }
    }
}
bool DataSet::initDataSet(const char* feature_dir, const char *feature_names) {

    char ftr_names[512];
    sprintf(ftr_names,"%s/%s",feature_dir, feature_names);
    cout << "->init dataset from "<< ftr_names << " ..." << endl; 
    ifstream ifile(ftr_names);
    if (!ifile.good()){
        cout<<"[ERROR]: image_names file do not exist!"<<endl;
        exit(0);
    }
    cout << "->start building ground truth file..." << endl;
    /* build ground truth file */
    dir_num = 0;
    sort_file_assign = ivec_new_0(FEATURE_NUM);  // the dir_assign of each feature
    file_elmt_num = ivec_new_0(FEATURE_NUM);     // the number of features in each dir

    string first_name;
    getline(ifile, first_name);
    string curr_dir = first_name.substr(0, 32);
    int file_idx = 0, dir_idx = 0;
    sort_file_assign[file_idx] = file_idx;
    file_idx++;
    file_elmt_num[dir_idx]++;
    
    while (!ifile.eof()) {
	string file_name;
	getline(ifile, file_name);
	string dir_name = file_name.substr(0, 32);
	if (dir_name != curr_dir) {
	    curr_dir = dir_name;
	    dir_idx++;
	}
	sort_file_assign[file_idx] = file_idx;
	file_idx++;
	file_elmt_num[dir_idx]++;	
    }
    dir_num = dir_idx + 1;
    cout << "  The total number of dir is " << dir_num << endl;
    ifile.close();
    
    file_start_idx = ivec_new_0(dir_num);  // the start index of each dir
    file_start_idx[0] = 0;
    for (int i = 1; i < dir_num; i++) {
        file_start_idx[i] = file_start_idx[i-1] + file_elmt_num[i-1];
    }
    /* save ground truth file */
    ofstream ofile("groundTruth_file_1280.dat");
    ofile.write((char*)&dir_num, sizeof(int));
    ofile.write((char*)sort_file_assign, FEATURE_NUM * sizeof(int));
    ofile.write((char*)file_elmt_num, dir_num * sizeof(int));
    ofile.write((char*)file_start_idx, dir_num * sizeof(int));
    ofile.close();
       
    m_num = FEATURE_NUM;
    m_dim = FEATURE_DIM;
    cout << "->feature number:" << m_num << "\tfeature dim:" << m_dim << endl;  
  
    m_features=fvec_new(m_dim*m_num);
    m_featureFileIdx=ivec_new(m_num);
    if (!m_features||!m_featureFileIdx) {
        cout<<"[ERROR]: can't malloc memory for DataSet m_features or m_featureFileIdx!"<<endl;
        exit(0);
    }
    return true;
}

void DataSet::getHeader(const char* filepath) {
    if (strstr(filepath,"summery.txt")) return;
    ifstream ifile(filepath,ios::binary);
    if(ifile.good()){
        // read feature file header
        FtrFileHeader header;
        ifile.read((char*)&header,sizeof(FtrFileHeader));
        m_num+=header.m_ftr_num;
        if(!m_dim) {
            m_dim = header.m_ftr_dim;
        }
    }
    ifile.close();
}

void DataSet::readData(char* feature_dir) {
    cout<<"->read data set:"<<feature_dir<<endl;
    const char *pdir=feature_dir;
    explore_dir(pdir,&DataSet::readFeatureFile);
}

void DataSet::readFeatureFile(const char *filepath) {
    
    cout << "->start reading feature from files..." << endl;
    ifstream ifile;
    ifile.open(filepath, ios::binary);
    if (!ifile.good()) {
        cout << "[EEROR]: Feature file cannot open!" << endl; 
	exit(0);
    }
  
    ofstream ofile;
    ofile.open("query_ftr_file_1280.bin", ios_base::binary);
    if (!ofile.good()) {
        cout << "[ERROR]: Query feature file cannot open!" << endl;
	exit(0);
    }
    
    ofstream outfile("query_ftr_num_1280.txt");
    
    m_features = new float[m_num * m_dim];
    //char *buffer;
    int i = 0, offset = 0;
    int total_num = 0;
    int true_query_num = 0;
    while (!ifile.eof() && i < dir_num) {
        int num = file_elmt_num[i];
	if (num == 0) {cout << "[ERROR]: dir_num cannot equal 0!" << endl;}
        size_t nbytes = num * m_dim * sizeof(float);
        char *buffer = new char[nbytes];
        ifile.read(buffer, nbytes);
        srand((unsigned)time(NULL));
        // int query_idx = rand() % num;
        int query_idx = num - 1;
	if (query_idx != 0) {true_query_num++;}
	outfile << (total_num + query_idx) << endl;
        total_num += num;
        int n = 0;
	for (n = 0; n < num; n++) {
	    memcpy(m_features + offset + n * m_dim, buffer + n * m_dim * sizeof(float), m_dim * sizeof(float));
	    if (n == query_idx) {
	      ofile.write((char*)(buffer + n * m_dim *sizeof(float)),  m_dim *sizeof(float));
	    }
	}
	i++;
	offset += num * m_dim;
	delete[] buffer;
	buffer = NULL;
    }
    cout << "true_query_num = " << true_query_num << endl; 
    ifile.close();
    ofile.close();
    outfile.close();
    //delete[] buffer;
    //buffer = NULL;
    cout << "->reading file succeed!" << endl;  
}
