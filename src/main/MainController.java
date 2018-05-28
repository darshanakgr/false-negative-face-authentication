package main;

import javafx.application.Platform;
import javafx.event.ActionEvent;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.Label;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;
import javafx.scene.paint.Color;
import org.opencv.core.*;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;


public class MainController {

    private final VideoCapture capture;
    private ScheduledExecutorService timer;
    private CascadeClassifier faceCascade;
    private int absoluteFaceSize;
    private boolean cameraActive;
    private boolean trained;

    @FXML
    public Button cameraButton;
    @FXML
    private ImageView currentFrame;
    @FXML
    public Label statusLabel;

    public MainController() {
        // load the opencv native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        // get capturing devices
        this.capture = new VideoCapture();
        // load the haar classifier
        this.faceCascade = new CascadeClassifier();
        this.faceCascade.load("./src/resources/haarcascades/haarcascade_frontalface_alt.xml");
    }

    public void startCamera(ActionEvent actionEvent) {
        if (!this.cameraActive) {
            // start the video capture
            this.capture.open(0);

            // is the video stream available?
            if (this.capture.isOpened()) {
                this.cameraActive = true;

                // grab a frame every 33 ms (30 frames/sec)
                Runnable frameGrabber = () -> {
                    // effectively grab and process a single frame
                    Mat frame = grabFrame();
                    // convert and show the frame
                    Image imageToShow = Utils.mat2Image(frame);
                    updateImageView(currentFrame, imageToShow);
                };

                this.timer = Executors.newSingleThreadScheduledExecutor();
                this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);

                // update the button content
                this.cameraButton.setText("Stop Camera");
            } else {
                // log the error
                System.err.println("Failed to open the camera connection...");
            }
        } else {
            // the camera is not active at this point
            this.cameraActive = false;
            // update again the button content
            this.cameraButton.setText("Authenticate");
            // stop the timer
            this.stopAcquisition();
        }
    }

    private Mat grabFrame() {
        Mat frame = new Mat();

        // check if the capture is open
        if (this.capture.isOpened()) {
            try {
                // read the current frame
                this.capture.read(frame);

                // if the frame is not empty, process it
                if (!frame.empty()) {
                    // face detection
                    this.detectAndDisplay(frame);
                }

            } catch (Exception e) {
                // log the (full) error
                System.err.println("Exception during the image elaboration: " + e);
            }
        }

        return frame;
    }

    private void detectAndDisplay(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();

        // convert the frame in gray scale
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        // equalize the frame histogram to improve the result
        Imgproc.equalizeHist(grayFrame, grayFrame);

        // compute minimum face size (20% of the frame height, in our case)
        if (this.absoluteFaceSize == 0) {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0) {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }

        // detect faces
        this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 1, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());

        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            // run face detection algorithm
            showFace(frame, facesArray[i]);
            // each rectangle in faces is a face: draw them!
            Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 0, 0), 3);
        }
    }

    private void showFace(Mat frame, Rect rect) {
        // gray scale image of face
        Mat grayface = new Mat();
        // crop the face out of the frame
        Imgproc.cvtColor(frame.submat(rect), grayface, Imgproc.COLOR_BGR2GRAY);
        // resize the face to 50x50
        Imgproc.resize(grayface, grayface, new Size(40, 40));
        // face train
        if (!trained) {
            if (FaceRecognition.trainFace(grayface)) {
                // training done
                this.trained = true;
                // start authentication
                Platform.runLater(() -> {
                    this.cameraActive = false;
                    // update again the button content
                    this.cameraButton.setText("Authenticate");
                    // update status
                    this.statusLabel.setText("Face Training Done. Now Press Authenticate");
                    // stop the timer
                    this.stopAcquisition();
                });
            }
        } else {
            Platform.runLater(() -> {
                double match = FaceRecognition.checkFace(grayface) * 100;
                this.statusLabel.setText(String.format("Access %s (Face Match %.2f%%)\n", match > 75 ? "Granted" : "Denied",match));
                this.statusLabel.setTextFill(match > 75 ? Color.web("#27ae60") : Color.web("#c0392b"));
            });
        }
    }

    private void stopAcquisition() {
        if (this.timer != null && !this.timer.isShutdown()) {
            try {
                // stop the timer
                this.timer.shutdown();
                this.timer.awaitTermination(33, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                // log any exception
                System.err.println("Exception in stopping the frame capture, trying to release the camera now... " + e);
            }
        }

        if (this.capture.isOpened()) {
            // release the camera
            this.capture.release();
        }
    }

    private void updateImageView(ImageView view, Image image) {
        Utils.onFXThread(view.imageProperty(), image);
    }

}