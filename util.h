#ifndef _UTIL_H
#define _UTIL_H
#include <iostream>
#include <ostream>

using namespace std;

#define FREE(p) do{if(p){ free(p);p=NULL;}}while(0)

struct Associator {
    float dis;
    int   idx;
    Associator(){
        dis=-1.0;
        idx=-1;
    }
    Associator(float _dis,int _idx) {
        dis=_dis;
        idx=_idx;
    }

    inline bool operator < (Associator &a)const {
        return dis<a.dis;
    }
    inline bool operator > ( Associator &a)const {
        return dis>a.dis;
    }
    inline bool operator == ( Associator &a)const {
        //return dis == a.dis;
        return idx == a.idx;
    }
    inline bool operator <= (Associator &a)const {
        return dis <= a.dis;
    }
    inline bool operator >= (Associator &a)const {
        return dis >= a.dis;
    }
  
    friend ostream& operator << (ostream& os,const Associator&a){
        os<<a.dis<<" "<<a.idx<<endl;
        return os;
    }

};

template<class T>
class MaxHeap {
    private:
        T *heap;
        int m_cursize;
        int m_maxsize;
    public:
        MaxHeap() {
            m_cursize=0;
            m_maxsize=0;
            heap = NULL;
        }
        void init(int maxsize) {
            m_cursize=0;
            setMaxsize(maxsize);
        }

        ~MaxHeap(){
            delete[] heap;
        }
        int size() const {
            return m_cursize;
        }
        void setMaxsize(int maxsize) {
            m_maxsize=maxsize;
            if(heap)
                delete[] heap;
            heap = new T[m_maxsize+1];
        }
    
    
        void remove(int i) {
            for (int j = (i+1); j < m_cursize; j++){
                heap[j-1] = heap[j];
            }
            m_cursize--;
            if (m_cursize < 0) {m_cursize = 0;}
        }
    
        bool isInsert(const T x) {
            int i = 0;
            while (i < m_cursize) {
                if (x == heap[i]) {
                    if (x < heap[i]) {remove(i);}
                    else             {return false;}
                }
                else {i++;}
            }
            return true;
        }
    
        void insert_help(const T x, int i) {
            if (i < m_maxsize) {
                for (int j = (m_cursize-1); j > (i-1); j--){
                    heap[j+1] = heap[j];
                }
                heap[i] = x;
                m_cursize++;
                if (m_cursize > m_maxsize) {m_cursize = m_maxsize;}
                return;
            }
            return;
        }
    
        void insert(const T x) {
            if (isInsert(x)) {
                //find the position
                int i = 0;
                while (i < m_cursize) {
                    if (x < heap[i]) {
                        insert_help(x, i);
                        return;
                    }
                    else {i++;}
                }
                insert_help(x, i);
                return;
                
            }
            /*
            //find the position
            if (m_cursize == m_maxsize) {
                int i = 0;  // the position
                while (i < m_cursize) {
                    if (x < heap[i]) {
                        for (int j = (m_cursize - 2); j >= i; j--) {
                            heap[j+1] = heap[j];
                        }
                        heap[i] = x;
                        return;
                    }
                    else {
                        i++;
                    }
                }
                
            }
            else {
                int i = 0;  // the position
                while (i < m_cursize) {
                    if (x < heap[i]) {
                        for (int j = (m_cursize - 1); j >= i; j--) {
                            heap[j+1] = heap[j];
                        }
                        heap[i] = x;
                        m_cursize++;
                        return;
                    }
                    else {
                        i++;
                    }
                }
                heap[i] = x;
                m_cursize++;
                return;
            }*/
            return ;
        }

        T *getData()const {
            return heap;
        }
        friend std::ostream& operator << (ostream& os,const MaxHeap<T>& mh){
            os<<"mh size:"<<mh.size()<<endl;
            for(int i = 0;i < mh.size();i++) {
                //os<<mh.heap[i]<<endl;
                os<<(mh.getData())[i];
            }
            return os;
        }
};


#ifdef _TEST
int main() {
    MaxHeap< Associator >* sift_point=new MaxHeap<Associator >[2](23);
    //MaxHeap< int > sift_point(100);

    for (int i =1000;i> 0;i--) {
        sift_point[0].insert(Associator(i*1.0,i));
        //sift_point.insert(i);
    }
    cout<<sift_point[0]<<endl;
    return 0;
}

#endif /*_TEST*/
#endif /* _UTIL_H*/

