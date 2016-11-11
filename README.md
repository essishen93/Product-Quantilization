# Product-Quantilization
The implementation of the paper " Product quantization for nearest neighbor search"

These codes are the modified version of codes from the following author:

/**
 * @file config_file.h
 * @Synopsis   read configure file
 * @author xuzuoxin@ict.ac.cn
 * @version 0.1
 * @date 2012-06-02
 */
 
 This project contains all the tools to find stable points in a 
picture.Additionally, it can also be used as an picture search 
engine.

The project mainly impletes the algorithm introduced by:

 "Product quantization for nearest neighbor search"
 Hervé Jégou, Matthijs Douze and Cordelia Schmid, 2011 TPAMI.

and also use a library(yael) from their project, all rights own to them.
  
1. Feature Extraction
  Run "extracter picture_dir feature_dir"
  picture_dir: the directory storing the original images
  feature_dir: the directory storing the output feature files
  
2.Train and build the search dataset
  Run "pqtrain base_pic_feature_dir"
  output files: model.dat, database.dat
  
3. Query a image
  Run "pqsearch  query_pic_feature_dir/query_pic_feature_list_file"
  The program will aotumatically load model.dat and database.dat to query
