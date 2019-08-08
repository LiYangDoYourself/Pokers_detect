#include <iostream>
//#include <opencv2/opencv.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include <cstring>
#include <cstdlib>
#include <cmath>
#include <list>
#include <vector>
#include <sys/time.h>
#include  "opencv2/objdetect/objdetect.hpp"


#define pi 3.14159


using namespace std;
using namespace cv;

int Maxval=0;
int Minval=256;
vector<Rect> NumEdgeVec;
vector<Rect> FlowerEdgeVec;

typedef struct _tsl_calibrate_threshold_contours{
	int thresVal;//当前阈值
	vector< vector <Point> > thresContours;//此阈值下所有符合要求的轮廓
}tsl_thres_and_conts;

typedef struct _mark_contours{
    int mark_num;
    vector<Point> vec_con;
}mark_contours;

struct match_same_contour{
                int Contourserialnumber;
                vector<int> MatchContourserialnumber;
};

struct ResContour{
            int tmpflag;
            vector<int> tmpotherflag;
};

struct LineParam{
    double k;
    double b;

} ;

struct Intersection{
    int x;
    int y;
};


// caculate the min_max_val
void caculate_val(string imagename){

    Mat srcImg=imread(imagename,0);

    Mat grayImg=srcImg.clone();/* = srcImg*/;
		//双边滤波bilateralFilter  1 srcimg 2 grayimg 3 zhijing 4 这个参数的值月大，表明该像素邻域内有月宽广的颜色会被混合到一起 5
		//bilateralFilter(srcImg, grayImg, 3, 3 * 2, 10 / 2);
		//
		//Mat element = getStructuringElement(MORPH_RECT, Size(3, 3));
		// 膨胀  use 3*3 replace the light point
		//dilate(grayImg, grayImg, element);


    int width =grayImg.cols;
    int height =grayImg.rows;


 	int Histogram[256] = { 0 };//灰度直方图
		for (int i = 0; i<height; i++)
		{
			uchar *pt = grayImg.ptr<uchar>(i);
			for (int j = 0; j<width; j++)
			{
				int temp = pt[j];
				temp = temp<0 ? 0 : temp;
				temp = temp>255 ? 255 : temp;
				Histogram[(int)temp]++;
			}
		}


    cout<<width<<"///"<<height<<endl;


    int validnum=1000;

    for(int k=0;k<256;k++)
    {
        if(Histogram[k]>validnum)
        {
            Minval=k;
            break;
        }

    }

    for (int k = 255; k >= 0; k--)
		{
			if (Histogram[k] > validnum)
			{
				Maxval = k;
				break;
			}
		}

		  cout<<"max:"<<Maxval<<"min:"<<Minval<<endl;

}


inline int time_cost(char* str,int flag)
{
    static struct  timeval beginTime = {0,0};
    static struct  timeval endTime = {0,0};

    if(flag==0)
    {
        gettimeofday(&beginTime, NULL);
    }
    else
    {
        gettimeofday(&endTime, NULL);
        long long tm_begin = beginTime.tv_sec*1000+beginTime.tv_usec/1000;
        long long tm_end = endTime.tv_sec*1000+endTime.tv_usec/1000;
        printf("%s cost: %lldms\n",str,tm_end-tm_begin);
    }

    return 0;
}

void detectnum(Mat &Dstpic,Mat src){

    string xmlpath="E:\\train2\\xml2424_20190729\\cascade.xml";
    //string xmlpath="E:\\train2\\xml2424_20190729\\cascade.xml";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-31-10-17-17\\2019-07-31-10-17-17_0001.jpg";
    //string imagename="E:\\0003.jpeg";
    cv::CascadeClassifier num_cascade;
    if(!num_cascade.load(xmlpath)){
        cout<<"error loading\n"<<endl;

    }

    //Mat frame = imread(imagename);
    Mat frame = src.clone();
    if(frame.empty())
    {
        cout<<"read fail"<<endl;
    }
    else{
        std::vector<Rect> pokers;
        Mat frame_gray;
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
        equalizeHist(frame_gray,frame_gray);

        //num_cascade.detectMultiScale(frame_gray,pokers,1.1,3,0|cv::CASCADE_SCALE_IMAGE,Size(20,20),Size(40,40));
        //缩放三次
        num_cascade.detectMultiScale(frame_gray,pokers,1.1,1,1,Size(24,24),Size(45,45));

        for(size_t i=0;i<pokers.size();i++)
        {
            //rectangle(frame,pokers[i],Scalar(0,0,255),1);
            //rectangle(Dstpic,pokers[i],Scalar(0,0,255),1);
            NumEdgeVec.push_back(pokers[i]);
            //cout<<pokers[i].width<<endl;
        }

        //imshow("123",frame);
        //imshow("123",Dstpic);
        //waitKey(0);

    }

}

void detectflower(Mat &Dstpic,Mat src){

    //string xmlpath="E:\\train2\\xml1616_20190801\\cascade.xml";
    string xmlpath="E:\\train2\\xml1616_20190801\\cascade.xml";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-31-10-17-17\\2019-07-31-10-17-17_0001.jpg";
    //string imagename="E:\\0003.jpeg";
    cv::CascadeClassifier num_cascade;
    if(!num_cascade.load(xmlpath)){
        cout<<"error loading\n"<<endl;

    }

    //Mat frame = imread(imagename);
    Mat frame = src.clone();
    if(frame.empty())
    {
        cout<<"read fail"<<endl;
    }
    else{

        std::vector<Rect> pokers;
        Mat frame_gray;
        cvtColor(frame,frame_gray,COLOR_BGR2GRAY);
        equalizeHist(frame_gray,frame_gray);


        //num_cascade.detectMultiScale(frame_gray,pokers,1.1,3,0|cv::CASCADE_SCALE_IMAGE,Size(20,20),Size(40,40));
        //缩放三次
        num_cascade.detectMultiScale(frame_gray,pokers,1.1,1,1,Size(16,16),Size(32,32));

        for(size_t i=0;i<pokers.size();i++)
        {
            //rectangle(frame,pokers[i],Scalar(0,0,255),1);
            //rectangle(Dstpic,pokers[i],Scalar(0,0,255),1);
            FlowerEdgeVec.push_back(pokers[i]);
            //cout<<pokers[i].width<<endl;
        }

        //imshow("123",frame);
        //imshow("123",Dstpic);
        //waitKey(0);

    }


}

Point section(float k,Point SrcRect,Point DstRect)
{
    LineParam line_fir,line_sec;
    Intersection section_fir;
    line_fir.k=k;
    line_fir.b=SrcRect.y-line_fir.k*SrcRect.x;
    line_sec.k=-1/k;
    line_sec.b=DstRect.y-line_sec.k*DstRect.x;
    section_fir.y=(line_sec.k*line_fir.b-line_fir.k*line_sec.b)/(line_sec.k-line_fir.k);
    section_fir.x=(line_fir.b-line_sec.b)/(line_sec.k-line_fir.k);
    return Point(section_fir.x,section_fir.y);
}

void readvideo(string videopath){

    VideoCapture cap(videopath);
    Mat src;
    int cnum=0;
    while(cnum<3)
    {
        Mat frame;
        cap>>frame;
        if(frame.empty())
        {
            cout<<"read error"<<endl;
        }

        detectnum(src,frame);
        detectflower(src,frame);
        cnum++;
    }


}

Mat readthirdpic()
{
    string dirpath="E:\\cut_pic\\20190716poker\\2019-07-31-10-21-48";
    vector<cv::String> picvec;
    glob(dirpath,picvec);
    Mat src;
    Mat frame;
    for(size_t i=0;i<4;i++)
    {
        frame=imread(picvec[i]);
        detectnum(src,frame);
        detectflower(src,frame);
    }
    return frame;
}

void combinatenumflower(Mat &Dstpic)
{
    // 数字排序 小->大
    for(size_t i=0;i<NumEdgeVec.size();i++)
    {
        for(size_t j=i;j<NumEdgeVec.size();j++)
        {
            int srcx=NumEdgeVec[i].x;
            int dstx=NumEdgeVec[j].x;
            if(srcx>=dstx)
            {
                Rect a;
                a=NumEdgeVec[i];
                NumEdgeVec[i]=NumEdgeVec[j];
                NumEdgeVec[j]=a;
            }

        }
    }


    // 花色排序
     for(size_t i=0;i<FlowerEdgeVec.size();i++)
    {
        for(size_t j=i;j<FlowerEdgeVec.size();j++)
        {
            int srcx=FlowerEdgeVec[i].x;
            int dstx=FlowerEdgeVec[j].x;
            if(srcx>=dstx)
            {
                Rect a;
                a=FlowerEdgeVec[i];
                FlowerEdgeVec[i]=FlowerEdgeVec[j];
                FlowerEdgeVec[j]=a;
            }

        }
    }

    cout<<NumEdgeVec.size()<<endl;
    cout<<FlowerEdgeVec.size()<<endl;

    #if 0
    for(size_t i=0;i<NumEdgeVec.size();i++)
    {
        Rect NumEdgeRect=NumEdgeVec[i];
        rectangle(Dstpic,NumEdgeRect,Scalar(0,255,0),1,8);
    }
    for(size_t j=0;j<FlowerEdgeVec.size();j++)
    {
        Rect FlowEdgeRect=FlowerEdgeVec[j];
        rectangle(Dstpic,FlowEdgeRect,Scalar(0,0,255),1,8);
    }

    imshow("dstpic",Dstpic);
    waitKey(0);
    #endif // 1

    // 取出不重合的数字边框
    // 保证每一个位置上就一个边框
    // 剔除较远的数字 合并面积相交大于Src_width*Dst_width
    int FlagFlower[NumEdgeVec.size()]={0};
    for(size_t i=0;i<NumEdgeVec.size();i++)
    {
        Rect SrcEdgeBox=NumEdgeVec[i];
        Point SrcCenter=Point(SrcEdgeBox.x+SrcEdgeBox.width/2,SrcEdgeBox.y+SrcEdgeBox.height/2);
        int flag=0;
        for(size_t j=0;j<NumEdgeVec.size();j++)
        {
            if(FlagFlower[i]==0)
            {
                if(i!=j)
                {
                Rect DstEdgeBox=NumEdgeVec[j];
                Point DstCenter=Point(DstEdgeBox.x+DstEdgeBox.width/2,DstEdgeBox.y+DstEdgeBox.height/2);
                Rect r3=NumEdgeVec[i]&NumEdgeVec[j];
                double S_D_Distance=sqrt((SrcCenter.x-DstCenter.x)*(SrcCenter.x-DstCenter.x)+(SrcCenter.y-DstCenter.y)*(SrcCenter.y-DstCenter.y));
                double ps=r3.area();

                if(S_D_Distance<6*(SrcEdgeBox.width/2))
                {
                    flag=1;
                }

                if(ps>(SrcEdgeBox.width/2)*(DstEdgeBox.width/2))
                {
                    FlagFlower[j]=1;
                }
                }
            }

        }
        if(flag==0)
        {
            FlagFlower[i]=1;
        }
    }

    vector<Rect> NewNumEdgeVec;
    for(size_t i=0;i<NumEdgeVec.size();i++)
    {
            if(FlagFlower[i]==0)
            {
                NewNumEdgeVec.push_back(NumEdgeVec[i]);
            }
    }

    // 取出不重合的花色边框
    int FlagFlower2[FlowerEdgeVec.size()]={0};
    for(size_t i=0;i<FlowerEdgeVec.size();i++)
    {
        Rect SrcEdgeBox=FlowerEdgeVec[i];
        Point SrcCenter=Point(SrcEdgeBox.x+SrcEdgeBox.width/2,SrcEdgeBox.y+SrcEdgeBox.height/2);
        int flag=0;
        for(size_t j=0;j<FlowerEdgeVec.size();j++)
        {
            if(FlagFlower2[i]==0)
            {
                if(i!=j)
                {
                Rect DstEdgeBox=FlowerEdgeVec[j];
                Point DstCenter=Point(DstEdgeBox.x+DstEdgeBox.width/2,DstEdgeBox.y+DstEdgeBox.height/2);
                Rect r3=FlowerEdgeVec[i]&FlowerEdgeVec[j];
                double S_D_Distance=sqrt((SrcCenter.x-DstCenter.x)*(SrcCenter.x-DstCenter.x)+(SrcCenter.y-DstCenter.y)*(SrcCenter.y-DstCenter.y));
                double ps=r3.area();

                if(S_D_Distance<9*(SrcEdgeBox.width/2))
                {
                    flag=1;
                }

                //if(ps>250)
                if(ps>(SrcEdgeBox.width/2)*(DstEdgeBox.width/2))
                {
                    FlagFlower2[j]=1;
                }

                if(DstEdgeBox.area()>=30*30)
                   {
                       //FlagFlower2[j]=1;
                   }

                }
            }

        }
        if(flag==0)
        //if(flag==0 || FlowerEdgeVec[i].area()>625)
        {
            FlagFlower2[i]=1;
        }




    }

    vector<Rect> NewFlowEdgeVec;
    for(size_t i=0;i<FlowerEdgeVec.size();i++)
    {
            if(FlagFlower2[i]==0)
            {
                NewFlowEdgeVec.push_back(FlowerEdgeVec[i]);
            }
    }

     cout<<NewNumEdgeVec.size()<<endl;
     cout<<NewFlowEdgeVec.size()<<endl;

    // 匹配距离小于Src_width+Dst_Width 最好的花色
    vector<ResContour> NumFlowerCombitionVec;
    for(size_t i=0;i<NewNumEdgeVec.size();i++)
    {
        ResContour tmpobj;
        Rect SrcEdgeBox=NewNumEdgeVec[i];
        Point SrcCenter=Point(SrcEdgeBox.x+SrcEdgeBox.width/2,SrcEdgeBox.y+SrcEdgeBox.height/2);
        for(size_t j=0;j<NewFlowEdgeVec.size();j++)
        {
           Rect DstEdgeBox=NewFlowEdgeVec[j];
           Point DstCenter=Point(DstEdgeBox.x+DstEdgeBox.width/2,DstEdgeBox.y+DstEdgeBox.height/2);
           double S_D_Distance=sqrt((SrcCenter.x-DstCenter.x)*(SrcCenter.x-DstCenter.x)+(SrcCenter.y-DstCenter.y)*(SrcCenter.y-DstCenter.y));
           //if(S_D_Distance<=40)
           if(S_D_Distance<=(SrcEdgeBox.width+DstEdgeBox.width))
           {
               tmpobj.tmpflag=i;
               tmpobj.tmpotherflag.push_back(j);
           }
        }

        if(tmpobj.tmpotherflag.size()>0)
        {
             NumFlowerCombitionVec.push_back(tmpobj);
        }

    }


    cout<<NumFlowerCombitionVec.size()<<endl;
    // 显示未组合的数字和花色
    #if 1
    for(size_t i=0;i<NewNumEdgeVec.size();i++)
    {
        Rect NumEdgeRect=NewNumEdgeVec[i];
        rectangle(Dstpic,NumEdgeRect,Scalar(0,255,0),1,8);
    }
    for(size_t j=0;j<NewFlowEdgeVec.size();j++)
    {
        Rect FlowEdgeRect=NewFlowEdgeVec[j];
        rectangle(Dstpic,FlowEdgeRect,Scalar(0,0,255),1,8);
    }

    imshow("dstpic",Dstpic);
    waitKey(0);
    #endif // 0

    #if 0
    for(size_t i=0;i<NumFlowerCombitionVec.size();i++)
    {
        Rect SrcEdgeRect=NewNumEdgeVec[NumFlowerCombitionVec[i].tmpflag];
        Point SrcCenter=Point(SrcEdgeRect.x+SrcEdgeRect.width/2,SrcEdgeRect.y+SrcEdgeRect.height/2);
        //circle(Dstpic,SrcCenter,2,Scalar(0,0,255),1,8);
        //rectangle(Dstpic,SrcEdgeRect,Scalar(0,255,0),1,8);
        int MinDis=1000;
        int Location=0;
        Rect TmpMinRect;
        Point TmpDstCenter;
        vector<Point> FourPoints;
        for(size_t j=0;j<NumFlowerCombitionVec[i].tmpotherflag.size();j++)
        {
            Rect DstEdgeRect=NewFlowEdgeVec[NumFlowerCombitionVec[i].tmpotherflag[j]];
            Point DstCenter=Point(DstEdgeRect.x+DstEdgeRect.width/2,DstEdgeRect.y+DstEdgeRect.height/2);
            double S_D_Distance=sqrt((SrcCenter.x-DstCenter.x)*(SrcCenter.x-DstCenter.x)+(SrcCenter.y-DstCenter.y)*(SrcCenter.y-DstCenter.y));
            string num=to_string(S_D_Distance);
            //line(Dstpic,SrcCenter,DstCenter,Scalar(255,255,255),1,8);
            //putText(Dstpic,num,Point((SrcCenter.x+DstCenter.x)/2,(SrcCenter.y+DstCenter.y)/2),FONT_HERSHEY_SIMPLEX,1,Scalar(0,0,255),2,8);
            //if(S_D_Distance>8)
                //{rectangle(Dstpic,DstEdgeRect,Scalar(255,0,0),1,8);}
            if(S_D_Distance<MinDis)
            {
                MinDis=S_D_Distance;
                Location=j;
                TmpMinRect=DstEdgeRect;
                TmpDstCenter=DstCenter;
            }

            cout<<S_D_Distance<<" ";
            cout<<"area:"<<DstEdgeRect.area();
        }

        //rectangle(Dstpic,TmpMinRect,Scalar(255,0,0),1,8);
        if(MinDis>8)
        {
            Rect CombineRect=TmpMinRect|SrcEdgeRect;
            rectangle(Dstpic,CombineRect,Scalar(255,0,0),1,8);
            line(Dstpic,SrcCenter,TmpDstCenter,Scalar(255,255,255),1,8);
            #if 0
            Point Numfirpoint=SrcEdgeRect.tl();
            Point Numsecpoint=Point(SrcEdgeRect.x+SrcEdgeRect.width,SrcEdgeRect.y);
            Point Numthipoint=Point(SrcEdgeRect.x,SrcEdgeRect.y+SrcEdgeRect.height);
            Point Numforpoint=SrcEdgeRect.br();
            Point Flowerfirpoint=TmpMinRect.tl();
            Point Flowersecpoint=Point(TmpMinRect.x+TmpMinRect.width,TmpMinRect.y);
            Point Flowerthipoint=Point(TmpMinRect.x,TmpMinRect.y+TmpMinRect.height);
            Point Flowerforpoint=TmpMinRect.br();
            if(SrcCenter.x==TmpDstCenter.x)
            {
                if(SrcCenter.y<TmpDstCenter.y)
                {
                    Point firsec=Numfirpoint;
                    Point secsec=Numsecpoint;
                    Point thisec=Point(Numfirpoint.x,Flowerthipoint.y);
                    Point forsec=Point(Numsecpoint.x,Flowerforpoint.y);
                    FourPoints.push_back(firsec);
                    FourPoints.push_back(secsec);
                    FourPoints.push_back(thisec);
                    FourPoints.push_back(forsec);
                }
            }

            else{
                float k=(SrcCenter.y-TmpDstCenter.y)/(SrcCenter.x-TmpDstCenter.x);
                if(k<0)
                {
                    //LineParam line_fir,line_sec;
                    //Intersection section_fir;
                    //line_fir.k=k;
                    //line_fir.b=Numfirpoint.y-line_fir.k*Numfirpoint.x;
                    //line_sec.k=-1/k;
                    //line_sec.b=Numsecpoint.y-line_sec.k*Numsecpoint.x;
                    //section_fir.y=(line_sec.k*line_fir.b-line_fir.k*line_sec.b)/(line_sec.k-line_fir.k);
                    //section_fir.x=(line_fir.b-line_sec.b)/(line_sec.k-line_fir.k);
                    Point firsec=section(k,Numfirpoint,Numsecpoint);
                    Point secsec=section(k,Numforpoint,Numsecpoint);
                    Point thisec=section(k,Numforpoint,Flowerthipoint);
                    Point forsec=section(k,Numfirpoint,Flowerthipoint);
                    FourPoints.push_back(firsec);
                    FourPoints.push_back(secsec);
                    FourPoints.push_back(thisec);
                    FourPoints.push_back(forsec);
                }

                else if(k>0)
                {
                    Point firsec=section(k,Numsecpoint,Numfirpoint);
                    Point secsec=section(k,Numsecpoint,Flowerforpoint);
                    Point thisec=section(k,Numthipoint,Flowerforpoint);
                    Point forsec=section(k,Numthipoint,Numfirpoint);
                    FourPoints.push_back(firsec);
                    FourPoints.push_back(secsec);
                    FourPoints.push_back(thisec);
                    FourPoints.push_back(forsec);
                }
                #if 1
                else
                {
                    cout<<k<<endl;
                    int flag=0;
                    if (SrcCenter.x>TmpDstCenter.x)
                    {
                        flag=1;
                    }

                    Point firsec=(flag==1)?Point(Flowerfirpoint.x,Numfirpoint.y):Numfirpoint;
                    Point secsec=(flag==1)?Numsecpoint:Point(Flowersecpoint.x,Numfirpoint.y);
                    Point thisec=(flag==1)?Point(Flowerthipoint.x,Numforpoint.y):Numthipoint;
                    Point forsec=(flag==1)?Numforpoint:Point(Flowerforpoint.x,Numforpoint.y);
                    FourPoints.push_back(firsec);
                    FourPoints.push_back(secsec);

                    FourPoints.push_back(forsec);
                     FourPoints.push_back(thisec);

                }
                #endif // 0

            }
            #endif // 0
        }

        //if(FourPoints.size()>0)
        {
            for(size_t i=0;i<4;i++)
            {
            //line(Dstpic,FourPoints[i],FourPoints[(i+1)%4],Scalar(255,0,0),1,8);
            }
            cout<<endl;
            imshow("dstpic",Dstpic);
            waitKey(0);
        }


    }
    #endif // 0

}



int main()
{

    //string imagename="E:\\0003.jpeg";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-30-17-31-00\\2019-07-30-17-31-00_0001.jpg";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-30-17-32-23\\2019-07-30-17-32-23_0003.jpg";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-30-17-33-10\\2019-07-30-17-33-10_0001.jpg";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-30-17-34-07\\2019-07-30-17-34-07_0001.jpg";
    //string imagename="E:\\cut_pic\\testpokers\\2019-08-02-11-10-33\\2019-08-02-11-10-33_0036.jpg";
    //string imagename="E:\\cut_pic\\20190716poker\\2019-07-31-10-18-50\\2019-07-31-10-18-50_0001.jpg";
    string imagename="E:\\cut_pic\\20190716poker\\2019-07-31-10-21-48\\2019-07-31-10-21-48_0004.jpg";

    //Mat Src=imread(imagename);

    //Mat Src(Size(1080,720),CV_8UC3,Scalar::all(0));
    //time_cost("s",0);
    //caculate_val(imagename);
    //poker_contour_draw(imagename);
    //poker_contour_extract(imagename);

    Mat Src=readthirdpic();

    //detectnum(Src,Src);
    //detectflower(Src,Src);
    combinatenumflower(Src);


    cout<<NumEdgeVec.size()<<endl;
    cout<<FlowerEdgeVec.size()<<endl;

   // time_cost("e",1);
    //string ImagePath="E:\\0341.jpeg";
    //caculate_val(ImagePath);
    //detectcolor(ImagePath);
    return 0;
}
