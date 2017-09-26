package com.example.s.movevector;

// Java
import java.util.List;
import java.util.Vector;

// OpenCV
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Size;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.MatOfPoint3f;
import org.opencv.core.Point;
import org.opencv.core.Point3;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.TermCriteria;
import org.opencv.features2d.Features2d;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.video.Video;
import org.opencv.android.CameraBridgeViewBase;

// UI
import android.app.Activity;
import android.os.Bundle;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Toast;

public class MainActivity extends Activity implements CameraBridgeViewBase.CvCameraViewListener2 {
    private static final String TAG = "OCVSample::Activity";

    // カメラ
    private CameraBridgeViewBase mOpenCvCameraView;

    // カメラ画像
    private Mat image, image_small;

    // オプティカルフロー用
    private Mat image_prev, image_next;
    private MatOfPoint2f pts_prev, pts_next;

    // OpenCVライブラリのロード
    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS: {
                    mOpenCvCameraView.enableView();
                    pts_prev = new MatOfPoint2f();
                    pts_next = new MatOfPoint2f();
                    break;
                }
                default: {
                    super.onManagerConnected(status);
                    break;
                }
            }
        }
    };

    // コンストラクタ
    public MainActivity() {
    }

    // 起動時
    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.camera_view);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
    }

    // 停止時
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    // 再開時
    @Override
    public void onResume() {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_2_0, this, mLoaderCallback);
        Toast.makeText(MainActivity.this, "onResume", Toast.LENGTH_SHORT).show();
    }

    // 破棄時
    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    // カメラ開始時
    @Override
    public void onCameraViewStarted(int width, int height) {
        image = new Mat(height, width, CvType.CV_8UC3);
        image_small = new Mat(height/8, width/8, CvType.CV_8UC3);
        image_prev = new Mat(image_small.rows(), image_small.cols(), image_small.type());
        image_next = new Mat(image_small.rows(), image_small.cols(), image_small.type());
    }

    // カメラ停止時
    @Override
    public void onCameraViewStopped() {
    }

    // 画像取得時
    @Override
    public Mat onCameraFrame(CvCameraViewFrame inputFrame) {
        // 縮小
        image = inputFrame.rgba();
        Imgproc.resize(image, image_small, image_small.size(), 0, 0, Imgproc.INTER_NEAREST);

        // グレースケール
        Mat gray = new Mat(image_small.rows(), image_small.cols(), CvType.CV_8UC1);
        Imgproc.cvtColor(image_small, gray, Imgproc.COLOR_RGB2GRAY);

        // 特徴点抽出
        MatOfPoint features = new MatOfPoint();
        Imgproc.goodFeaturesToTrack(gray, features, 50, 0.01, 10);

        // 特徴点が見つかった
        if (features.total() > 0) {
            // 過去のデータが存在する
            if (pts_prev.total() > 0) {
                // 現在のデータ
                gray.copyTo(image_next);
                pts_next = new MatOfPoint2f(features.toArray());

                // オプティカルフロー算出
                MatOfByte status = new MatOfByte();
                MatOfFloat err = new MatOfFloat();
                Video.calcOpticalFlowPyrLK(image_prev, image_next, pts_prev, pts_next, status, err);

                // 表示
                long flow_num = status.total();
                if (flow_num > 0) {
                    List<Byte>  list_status = status.toList();
                    List<Point> list_features_prev = pts_prev.toList();
                    List<Point> list_features_next = pts_next.toList();
                    double scale_x = image.cols() / image_small.cols();
                    double scale_y = image.rows() / image_small.rows();
                    for (int i = 0; i < flow_num; i++) {
                        if (list_status.get(i) == 1) {
                            Point p1 = new Point();
                            p1.x = list_features_prev.get(i).x * scale_x;
                            p1.y = list_features_prev.get(i).y * scale_y;
                            //Core.circle(image, p1, 3, new Scalar(255,0,0), -1, 8, 0 );
                            Point p2 = new Point();
                            p2.x = list_features_next.get(i).x * scale_x;
                            p2.y = list_features_next.get(i).y * scale_y;
                            //Core.circle(image, p2, 3, new Scalar(255,255,0), -1, 8, 0 );

                            // フロー描画
                            int thickness = 5;
                            Imgproc.line(image, p1, p2, new Scalar(0,255,0), thickness);
                        }
                    }
                }
            }

            // 過去のデータ
            gray.copyTo(image_prev);
            pts_prev = new MatOfPoint2f(features.toArray());
        }

        return image;
    }
}